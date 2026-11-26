import logging
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from uuid import UUID, uuid4

from aidial_client import DialException
from aidial_client.types.metadata import FileItem
from sqlalchemy import and_, delete, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.auth.schemas import User
from statgpt.admin.services.exceptions import (
    PreparedDownloadChannelNotFoundError,
    PreparedDownloadExpiredError,
    PreparedDownloadNotFoundError,
    PreparedDownloadStorageError,
)
from statgpt.common import models, schemas
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.utils.async_utils import await_cancellation_safe
from statgpt.common.utils.dial.attachments import AttachmentsStorage, attachments_storage_factory

_log = logging.getLogger(__name__)

PREPARED_DOWNLOAD_LIFETIME = timedelta(hours=1)
PREPARED_DOWNLOAD_PREFIX = "prepared-downloads"
_PREPARED_DOWNLOAD_ORPHAN_GRACE = timedelta(hours=24)

_FOREIGN_KEY_VIOLATION_SQLSTATE = "23503"


class PreparedDownloadService:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def ensure_channel_exists(self, channel_id: int) -> None:
        channel = await self._session.scalar(
            select(models.Channel.id).where(models.Channel.id == channel_id)
        )
        if channel is None:
            raise PreparedDownloadChannelNotFoundError(channel_id)

    async def prepare(
        self,
        channel_id: int,
        user: User,
        auth_context: AuthContext,
        content: bytes,
        filename: str,
        media_type: str,
    ) -> schemas.PreparedDownloadResponse:
        download_id = uuid4()
        extension = filename.rsplit(".", 1)[1]
        name = f"{PREPARED_DOWNLOAD_PREFIX}/{download_id}.{extension}"
        file_path: str | None = None
        try:
            try:
                async with attachments_storage_factory(auth_context.api_key) as storage:
                    uploaded = await storage.put_file(name, media_type, content)
                    file_path = uploaded.url
            except Exception as exc:
                raise PreparedDownloadStorageError() from exc

            prepared_at = datetime.now(UTC)
            expires_at = prepared_at + PREPARED_DOWNLOAD_LIFETIME
            await self.ensure_channel_exists(channel_id)
            record = models.PreparedDownload(
                id=download_id,
                owner_user_id=user.id,
                channel_id=channel_id,
                file_path=file_path,
                filename=filename,
                media_type=media_type,
                created_at=prepared_at,
                updated_at=prepared_at,
                expires_at=expires_at,
            )
            self._session.add(record)
            try:
                await self._session.commit()
            except IntegrityError as exc:
                await self._session.rollback()
                if getattr(exc.orig, "sqlstate", None) == _FOREIGN_KEY_VIOLATION_SQLSTATE:
                    await self.ensure_channel_exists(channel_id)
                raise
            return schemas.PreparedDownloadResponse(id=download_id, expires_at=expires_at)
        except BaseException:
            await await_cancellation_safe(self._compensate(file_path, auth_context))
            raise

    async def get_download(
        self, download_id: UUID, channel_id: int, user: User
    ) -> models.PreparedDownload:
        record = await self._session.scalar(
            select(models.PreparedDownload).where(
                models.PreparedDownload.id == download_id,
                models.PreparedDownload.channel_id == channel_id,
                models.PreparedDownload.owner_user_id == user.id,
            )
        )
        if record is None:
            raise PreparedDownloadNotFoundError()
        if record.expires_at <= datetime.now(UTC):
            raise PreparedDownloadExpiredError()
        return record

    async def cleanup(self, auth_context: AuthContext, batch_size: int = 100) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cutoff = datetime.now(UTC)
        async with attachments_storage_factory(auth_context.api_key) as storage:
            await self._cleanup_expired_rows(storage, cutoff, batch_size)
            await self._cleanup_orphans(storage, cutoff, batch_size)

    async def _cleanup_expired_rows(
        self, storage: AttachmentsStorage, cutoff: datetime, batch_size: int
    ) -> None:
        cursor: tuple[datetime, UUID] | None = None
        while True:
            query = (
                select(
                    models.PreparedDownload.expires_at,
                    models.PreparedDownload.id,
                    models.PreparedDownload.file_path,
                )
                .where(models.PreparedDownload.expires_at <= cutoff)
                .order_by(models.PreparedDownload.expires_at, models.PreparedDownload.id)
                .limit(batch_size)
            )
            if cursor is not None:
                expires_at, download_id = cursor
                query = query.where(
                    or_(
                        models.PreparedDownload.expires_at > expires_at,
                        and_(
                            models.PreparedDownload.expires_at == expires_at,
                            models.PreparedDownload.id > download_id,
                        ),
                    )
                )
            async with self._session.begin():
                rows = (await self._session.execute(query)).all()
            if not rows:
                return

            for expires_at, download_id, file_path in rows:
                cursor = (expires_at, download_id)
                try:
                    await self._delete_file(storage, file_path)
                    async with self._session.begin():
                        await self._session.execute(
                            delete(models.PreparedDownload).where(
                                models.PreparedDownload.id == download_id,
                                models.PreparedDownload.file_path == file_path,
                                models.PreparedDownload.expires_at <= cutoff,
                            )
                        )
                except Exception:
                    _log.exception("Failed to clean up prepared download %s.", download_id)
            del rows

    async def _cleanup_orphans(
        self, storage: AttachmentsStorage, cutoff: datetime, batch_size: int
    ) -> None:
        oldest_allowed = int((cutoff - _PREPARED_DOWNLOAD_ORPHAN_GRACE).timestamp() * 1000)
        async for file in storage.iter_files_in_folder(
            PREPARED_DOWNLOAD_PREFIX, page_size=batch_size
        ):
            try:
                if not self._is_generated_file(file):
                    continue
                timestamps = [t for t in (file.created_at, file.updated_at) if t is not None]
                if not timestamps:
                    _log.warning("Skipping prepared-download file with unknown age: %s", file.url)
                    continue
                if max(timestamps) >= oldest_allowed:
                    continue
                async with self._session.begin():
                    reference = await self._session.scalar(
                        select(models.PreparedDownload.id).where(
                            models.PreparedDownload.file_path == file.url
                        )
                    )
                if reference is None:
                    await self._delete_file(storage, file.url)
            except Exception:
                _log.exception("Failed to clean up orphan prepared-download file %s.", file.url)

    @staticmethod
    def _is_generated_file(file: FileItem) -> bool:
        if file.node_type != "ITEM" or not file.url.startswith("files/"):
            return False
        parent, _, filename = file.url.rpartition("/")
        if not parent.endswith(f"/{PREPARED_DOWNLOAD_PREFIX}"):
            return False
        stem, _, extension = filename.rpartition(".")
        if extension not in {"csv", "xlsx"}:
            return False
        try:
            return str(UUID(stem)) == stem
        except ValueError:
            return False

    @staticmethod
    async def _delete_file(storage: AttachmentsStorage, file_path: str) -> None:
        try:
            await storage.delete_file(file_path)
        except DialException as exc:
            if exc.status_code != HTTPStatus.NOT_FOUND:
                raise

    async def _compensate(self, file_path: str | None, auth_context: AuthContext) -> None:
        try:
            await self._session.rollback()
        except Exception:
            _log.exception("Failed to roll back prepared-download metadata.")
        if file_path is not None:
            try:
                async with attachments_storage_factory(auth_context.api_key) as storage:
                    await storage.delete_file(file_path)
            except Exception:
                _log.exception("Failed to remove an uploaded file after preparation failed.")
