import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from aidial_client.types.metadata import FileItem
from fastapi import APIRouter, FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, event, inspect, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.auth.schemas import User
from statgpt.admin.auth.user import require_jwt_auth
from statgpt.admin.exception_handlers import register_exception_handlers
from statgpt.admin.routers import discovery_datasets as router
from statgpt.admin.services import prepared_downloads
from statgpt.admin.services.exceptions import PreparedDownloadNotFoundError
from statgpt.admin.services.prepared_downloads import PreparedDownloadService
from statgpt.common import models
from statgpt.common.config.versions import Versions
from statgpt.common.utils import MediaTypes
from statgpt.common.utils.dial.attachments import AttachmentsStorage


@pytest_asyncio.fixture
async def database(async_session_factory) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = async_session_factory.kw["bind"]
    try:
        async with async_session_factory() as session:
            assert await session.scalar(text("SELECT current_database()")) == "test_db"
        yield async_session_factory
    finally:
        await engine.dispose()


@pytest.fixture
def user() -> User:
    identifier = f"prepared-download-test-{uuid4()}"
    return User(id=identifier, username=identifier, name="Integration Test")


@pytest_asyncio.fixture
async def channel_id(database, user: User) -> AsyncIterator[int]:
    async with database() as session:
        channel = models.Channel(
            title="Prepared download integration test",
            description="Integration test",
            deployment_id=f"prepared-download-test-{uuid4()}",
            details={},
        )
        session.add(channel)
        await session.commit()
        identifier = channel.id
    try:
        yield identifier
    finally:
        async with database() as session:
            await session.execute(
                delete(models.PreparedDownload).where(
                    models.PreparedDownload.owner_user_id == user.id
                )
            )
            await session.execute(delete(models.Channel).where(models.Channel.id == identifier))
            await session.commit()


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    storage = MagicMock(spec=AttachmentsStorage)

    async def upload(name: str, media_type: str, content: bytes) -> FileItem:
        return FileItem(
            bucket="integration-tests",
            url=f"files/integration-tests/{name}",
            node_type="ITEM",
            resource_type="FILE",
        )

    async def files(folder: str, page_size: int) -> AsyncIterator[FileItem]:
        for file in ():
            yield file

    storage.put_file.side_effect = upload
    storage.delete_file.return_value = None
    storage.iter_files_in_folder.side_effect = files

    @asynccontextmanager
    async def context(api_key: str):
        yield storage

    monkeypatch.setattr(prepared_downloads, "attachments_storage_factory", context)
    return storage


@pytest_asyncio.fixture
async def client(database, user: User, monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    channels = APIRouter(prefix="/channels")
    channels.include_router(router.channel_discovery_datasets_router)
    app.include_router(channels)
    register_exception_handlers(app)
    app.dependency_overrides[require_jwt_auth] = lambda: user

    @asynccontextmanager
    async def session_context():
        async with database() as session:
            yield session

    monkeypatch.setattr(router, "get_session_context_manager", session_context)
    monkeypatch.setattr(
        router.DiscoveryDatasetService,
        "export_discovery_datasets",
        AsyncMock(return_value=b"prepared snapshot"),
    )
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.parametrize(
    "filename,media_type",
    [("discovery_datasets.csv", MediaTypes.CSV), ("discovery_datasets.xlsx", MediaTypes.XLSX)],
)
async def test_prepared_metadata_survives_session_changes(
    database, channel_id: int, user: User, storage: MagicMock, filename: str, media_type: str
) -> None:
    started_at = datetime.now(UTC)
    async with database() as session:
        response = await PreparedDownloadService(session).prepare(
            channel_id, user, SystemUserAuthContext(), b"prepared snapshot", filename, media_type
        )
    finished_at = datetime.now(UTC)

    async with database() as session:
        record = await PreparedDownloadService(session).get_download(response.id, channel_id, user)
        assert isinstance(record.id, UUID)
        assert record.id.version == 4
        assert record.id == response.id
        assert record.owner_user_id == user.id
        assert record.channel_id == channel_id
        assert record.filename == filename
        assert record.media_type == media_type
        extension = filename.rsplit(".", 1)[1]
        assert (
            record.file_path
            == f"files/integration-tests/prepared-downloads/{response.id}.{extension}"
        )
        assert started_at <= record.created_at <= finished_at
        assert record.updated_at == record.created_at
        assert record.expires_at == response.expires_at == record.created_at + timedelta(hours=1)
        for timestamp in (record.created_at, record.updated_at, record.expires_at):
            assert timestamp.utcoffset() == timedelta(0)
        expires_at = record.expires_at

    async with database() as session:
        refreshed_user = User(id=user.id, username="refreshed", name="Refreshed Identity")
        record = await PreparedDownloadService(session).get_download(
            response.id, channel_id, refreshed_user
        )
        assert record.expires_at == expires_at
    storage.put_file.assert_awaited_once_with(
        f"prepared-downloads/{response.id}.{extension}", media_type, b"prepared snapshot"
    )
    storage.delete_file.assert_not_awaited()


async def test_model_uuid_default_and_non_utc_timestamp_round_trip(
    database, channel_id: int, user: User
) -> None:
    deadline = datetime.now(timezone(timedelta(hours=5, minutes=30))) + timedelta(hours=1)
    async with database() as session:
        record = models.PreparedDownload(
            owner_user_id=user.id,
            channel_id=channel_id,
            file_path="files/integration-tests/prepared-downloads/default.csv",
            filename="discovery_datasets.csv",
            media_type=MediaTypes.CSV,
            expires_at=deadline,
        )
        session.add(record)
        await session.commit()
        identifier = record.id
    async with database() as session:
        record = await session.get(models.PreparedDownload, identifier)
        assert record is not None
        assert isinstance(record.id, UUID)
        assert record.id.version == 4
        assert record.expires_at == deadline.astimezone(UTC)
        assert record.expires_at.utcoffset() == timedelta(0)
        assert record.created_at.utcoffset() == record.updated_at.utcoffset() == timedelta(0)


async def test_channel_deletion_retains_file_reference_until_expired_cleanup(
    database, channel_id: int, user: User, storage: MagicMock
) -> None:
    async with database() as session:
        response = await PreparedDownloadService(session).prepare(
            channel_id,
            user,
            SystemUserAuthContext(),
            b"prepared snapshot",
            "discovery_datasets.csv",
            MediaTypes.CSV,
        )
    async with database() as session:
        channel = await session.get(models.Channel, channel_id)
        await session.delete(channel)
        await session.commit()
    async with database() as session:
        record = await session.get(models.PreparedDownload, response.id)
        assert record is not None
        assert record.channel_id is None
        assert record.expires_at == response.expires_at
        file_path = record.file_path
        with pytest.raises(PreparedDownloadNotFoundError):
            await PreparedDownloadService(session).get_download(response.id, channel_id, user)
    async with database() as session:
        await PreparedDownloadService(session).cleanup(SystemUserAuthContext(), batch_size=1)
    storage.delete_file.assert_not_awaited()
    async with database() as session:
        assert await session.get(models.PreparedDownload, response.id) is not None
        await session.execute(
            update(models.PreparedDownload)
            .where(models.PreparedDownload.id == response.id)
            .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        await session.commit()

    async def delete_file(path: str) -> None:
        assert path == file_path
        async with database() as session:
            retained = await session.get(models.PreparedDownload, response.id)
            assert retained is not None
            assert retained.channel_id is None
            assert retained.file_path == path

    storage.delete_file.side_effect = delete_file
    async with database() as session:
        await PreparedDownloadService(session).cleanup(SystemUserAuthContext(), batch_size=1)
    async with database() as session:
        assert await session.get(models.PreparedDownload, response.id) is None
    storage.delete_file.assert_awaited_once_with(file_path)
    async with database() as session:
        await PreparedDownloadService(session).cleanup(SystemUserAuthContext(), batch_size=1)
    storage.delete_file.assert_awaited_once_with(file_path)


@pytest.mark.parametrize("deletion_stage", ["after_upload", "after_precheck"])
@pytest.mark.parametrize("compensation_fails", [False, True])
async def test_concurrent_channel_deletion_maps_to_404_and_compensates_upload(
    database,
    channel_id: int,
    user: User,
    storage: MagicMock,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    deletion_stage: str,
    compensation_fails: bool,
) -> None:
    upload_finished = asyncio.Event()
    ready_for_deletion = asyncio.Event()
    deletion_finished = asyncio.Event()
    prechecks: list[bool] = []
    integrity_errors: list[IntegrityError] = []
    original_upload = storage.put_file.side_effect
    original_precheck = PreparedDownloadService.ensure_channel_exists

    async def upload(name: str, media_type: str, content: bytes) -> FileItem:
        file = await original_upload(name, media_type, content)
        upload_finished.set()
        if deletion_stage == "after_upload":
            ready_for_deletion.set()
            await deletion_finished.wait()
        return file

    async def precheck(service: PreparedDownloadService, identifier: int) -> None:
        await original_precheck(service, identifier)
        prechecks.append(upload_finished.is_set())
        if deletion_stage == "after_precheck" and upload_finished.is_set():
            ready_for_deletion.set()
            await deletion_finished.wait()

    def capture_error(context) -> None:
        if isinstance(context.sqlalchemy_exception, IntegrityError):
            integrity_errors.append(context.sqlalchemy_exception)

    storage.put_file.side_effect = upload
    if compensation_fails:
        storage.delete_file.side_effect = RuntimeError("Compensation unavailable")
    monkeypatch.setattr(PreparedDownloadService, "ensure_channel_exists", precheck)
    engine = database.kw["bind"]
    event.listen(engine.sync_engine, "handle_error", capture_error)
    request = asyncio.create_task(client.post(f"/channels/{channel_id}/discovery-datasets/exports"))
    try:
        await asyncio.wait_for(ready_for_deletion.wait(), timeout=10)
        assert not request.done()
        async with database() as session:
            await session.execute(text("SET LOCAL lock_timeout = '2s'"))
            await session.execute(delete(models.Channel).where(models.Channel.id == channel_id))
            await session.commit()
        deletion_finished.set()
        response = await asyncio.wait_for(request, timeout=10)
    finally:
        deletion_finished.set()
        if not request.done():
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        event.remove(engine.sync_engine, "handle_error", capture_error)

    assert response.status_code == 404
    assert response.json() == {"detail": f"Channel {channel_id} not found."}
    assert response.headers["cache-control"] == "no-store"
    assert prechecks == ([False, True] if deletion_stage == "after_precheck" else [False])
    if deletion_stage == "after_precheck":
        assert len(integrity_errors) == 1
        assert integrity_errors[0].orig.sqlstate == "23503"
        assert "prepared_downloads_channel_id_fkey" in str(integrity_errors[0].orig)
    else:
        assert integrity_errors == []
    upload_name = storage.put_file.await_args.args[0]
    storage.delete_file.assert_awaited_once_with(f"files/integration-tests/{upload_name}")
    async with database() as session:
        assert await session.get(models.Channel, channel_id) is None
        assert (
            await session.scalar(
                select(models.PreparedDownload.id).where(
                    models.PreparedDownload.owner_user_id == user.id
                )
            )
            is None
        )


@pytest.mark.parametrize("failure", ["duplicate_uuid", "unrelated_foreign_key"])
async def test_unrelated_integrity_errors_are_not_remapped(
    database,
    channel_id: int,
    user: User,
    storage: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    identifier = uuid4()
    async with database() as session:
        if failure == "duplicate_uuid":
            session.add(
                models.PreparedDownload(
                    id=identifier,
                    owner_user_id=user.id,
                    channel_id=channel_id,
                    file_path="files/integration-tests/existing.csv",
                    filename="discovery_datasets.csv",
                    media_type=MediaTypes.CSV,
                    expires_at=datetime.now(UTC) + timedelta(hours=1),
                )
            )
            await session.commit()
            monkeypatch.setattr(prepared_downloads, "uuid4", lambda: identifier)
        else:
            assert await session.get(models.DataSet, -1) is None
            session.add(models.ChannelDataset(channel_id=channel_id, dataset_id=-1))
        with pytest.raises(IntegrityError) as error:
            await PreparedDownloadService(session).prepare(
                channel_id,
                user,
                SystemUserAuthContext(),
                b"prepared snapshot",
                "discovery_datasets.csv",
                MediaTypes.CSV,
            )
        assert error.value.orig.sqlstate == ("23505" if failure == "duplicate_uuid" else "23503")
        constraint = (
            "prepared_downloads_pkey"
            if failure == "duplicate_uuid"
            else "channel_datasets_dataset_id_fkey"
        )
        assert constraint in str(error.value.orig)
        assert not session.in_transaction()
        assert await session.get(models.Channel, channel_id) is not None
    upload_name = storage.put_file.await_args.args[0]
    storage.delete_file.assert_awaited_once_with(f"files/integration-tests/{upload_name}")
    async with database() as session:
        records = (
            await session.scalars(
                select(models.PreparedDownload).where(
                    models.PreparedDownload.owner_user_id == user.id
                )
            )
        ).all()
        assert len(records) == (1 if failure == "duplicate_uuid" else 0)
        if records:
            assert records[0].file_path == "files/integration-tests/existing.csv"


async def test_prepared_download_migration_schema_matches_target(database) -> None:
    async with database() as session:
        assert (
            await session.scalar(text("SELECT version_num FROM alembic_version"))
            == Versions.ALEMBIC_TARGET_VERSION
        )
        connection = await session.connection()

        def verify_schema(connection) -> None:
            schema = inspect(connection)
            columns = {
                column["name"]: column for column in schema.get_columns("prepared_downloads")
            }
            assert isinstance(
                columns["id"]["type"], models.PreparedDownload.__table__.c.id.type.__class__
            )
            assert schema.get_pk_constraint("prepared_downloads")["constrained_columns"] == ["id"]
            assert columns["channel_id"]["nullable"]
            for name in (
                "owner_user_id",
                "file_path",
                "filename",
                "media_type",
                "expires_at",
                "created_at",
                "updated_at",
            ):
                assert not columns[name]["nullable"]
            for name in ("expires_at", "created_at", "updated_at"):
                assert columns[name]["type"].timezone
            foreign_keys = schema.get_foreign_keys("prepared_downloads")
            assert len(foreign_keys) == 1
            assert foreign_keys[0]["constrained_columns"] == ["channel_id"]
            assert foreign_keys[0]["referred_table"] == "channels"
            assert foreign_keys[0]["referred_columns"] == ["id"]
            assert foreign_keys[0]["options"]["ondelete"] == "SET NULL"
            indexes = schema.get_indexes("prepared_downloads")
            assert any(
                index["name"] == "ix_prepared_downloads_expires_at"
                and index["column_names"] == ["expires_at"]
                and not index["unique"]
                for index in indexes
            )

        await connection.run_sync(verify_schema)
