import logging
import os
import tempfile
import zipfile
from mimetypes import guess_type

import yaml
from fastapi import BackgroundTasks, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import func

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.audit.decorators import audit_action
from statgpt.admin.settings.exim import JobsConfig
from statgpt.common import utils
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import AuditActionType, AuditEntityType
from statgpt.common.schemas import PreprocessingStatusEnum as StatusEnum
from statgpt.common.services import ChannelSerializer, ChannelService
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils import (
    attachments_storage_factory,
    dial_client_factory,
    download_file_by_path,
)
from statgpt.common.utils.elastic import ElasticSearchFactory
from statgpt.common.vectorstore import DedupCounts, VectorStoreFactory

from .background_tasks import background_task
from .exceptions import raise_for_integrity_error
from .status_recovery import set_failed_status

_log = logging.getLogger(__name__)


class AdminPortalChannelService(ChannelService):
    def __init__(self, session: AsyncSession | None = None) -> None:
        super().__init__(session, None)

    @staticmethod
    async def _export_dial_file_to_folder(
        dial_file_path: str, folder_path: str, auth_context: AuthContext
    ) -> None:
        async with dial_client_factory(dial_settings.url, auth_context.api_key) as dial:
            content, _ = await download_file_by_path(dial, dial_file_path)

        dial_files_root = os.path.join(folder_path, JobsConfig.DIAL_FILES_FOLDER)
        target_file = utils.safe_join(dial_files_root, dial_file_path)
        utils.write_bytes(content, target_file)

    @staticmethod
    async def _import_dial_file_from_folder(
        file_path: str, dial_file_path: str, auth_context: AuthContext
    ) -> None:
        content = utils.read_bytes(file_path)
        mime_type = guess_type(file_path)[0]
        if not mime_type:
            mime_type = "application/octet-stream"
        async with attachments_storage_factory(api_key=auth_context.api_key) as storage:
            await storage.put_file(dial_file_path, mime_type, content)

    async def _create_channel_model(self, data: schemas.ChannelBase) -> models.Channel:
        item = models.Channel(
            title=data.title,
            description=data.description,
            deployment_id=data.deployment_id,
            details=data.details.model_dump(mode="json", by_alias=True),
            llm_model=data.llm_model,
        )

        try:
            self._session.add(item)
            await self._session.flush()
        except IntegrityError as e:
            raise_for_integrity_error(
                e, f"Key deployment_id='{data.deployment_id}' already exists."
            )
        return item

    @audit_action(entity_type=AuditEntityType.CHANNEL, action_type=AuditActionType.CREATE)
    async def create_channel(self, data: schemas.ChannelBase) -> schemas.Channel:
        item = await self._create_channel_model(data)
        return ChannelSerializer.db_to_schema(item)

    @audit_action(entity_type=AuditEntityType.CHANNEL, action_type=AuditActionType.UPDATE)
    async def update(self, item_id: int, data: schemas.ChannelUpdate) -> schemas.Channel:
        item = await self._get_item_or_raise(item_id)

        if data.details is not None:
            data.details = data.details.model_dump(mode="json", by_alias=True)  # type: ignore

        query = (
            update(models.Channel)
            .where(models.Channel.id == item.id)
            .values(**data.model_dump(mode="json", exclude_unset=True), updated_at=func.now())
            .returning(models.Channel)
        )
        try:
            item = (await self._session.execute(query)).scalar_one()
            await self._session.flush()
        except IntegrityError as e:
            raise_for_integrity_error(
                e, f"Key deployment_id='{data.deployment_id}' already exists."
            )

        return ChannelSerializer.db_to_schema(item)

    @audit_action(entity_type=AuditEntityType.CHANNEL, action_type=AuditActionType.DELETE)
    async def delete(self, item_id: int) -> schemas.Channel:
        item = await self._get_item_or_raise(item_id)
        deleted_item = ChannelSerializer.db_to_schema(item)
        _log.info(f"Deleting {item}")

        await self._clear_vector_store(item)

        if self.is_channel_hybrid(deleted_item):
            await self._clear_elastic_indexes(item)

        await self._session.delete(item)
        await self._session.flush()
        return deleted_item

    async def _clear_vector_store(self, channel: models.Channel) -> None:
        vector_store_factory = VectorStoreFactory()

        collections = [
            channel.indicator_table_name,
            channel.non_indicator_dimensions_table_name,
            channel.special_dimensions_table_name,
        ]
        for col in collections:
            vector_store = await vector_store_factory.get_embeddingless_vector_store(col)
            await vector_store.clear()

    async def _clear_elastic_indexes(self, channel: models.Channel) -> None:
        _log.info(f"Clearing Elastic indexes for channel {channel!r}")
        for index in [channel.matching_index_name, channel.indicators_index_name]:
            await self._clear_elastic_index(index)

    @staticmethod
    async def _clear_elastic_index(index_name: str) -> None:
        """Drops index if it is possible, otherwise deletes all documents from the index."""

        try:
            index = await ElasticSearchFactory.get_index(index_name)
        except Exception:
            _log.exception(f"Failed to get Elastic index {index_name!r}")
            return

        try:
            await index.delete()
            _log.info(f"Dropped Elastic index {index_name!r}")
            return
        except Exception as e:
            _log.info(f"Exception: {e}")
            _log.warning(f"Failed to drop Elastic index {index_name!r}, trying to clear it...")

        try:
            res = await index.delete_by_query(query={"match_all": {}})
            _log.debug(res)
            _log.info(f"Cleared {res['deleted']} documents from Elastic index {index_name!r}")
        except Exception:
            _log.exception(f"Failed to clear Elastic index {index_name!r}")

    async def export_channel_to_folder(
        self,
        channel_id: int,
        folder_path: str,
        scope: schemas.ExportScope,
        auth_context: AuthContext,
    ) -> models.Channel:
        channel_db = await self.get_model_by_id(channel_id)
        channel_schema = ChannelSerializer.db_to_schema(channel_db)

        if scope.includes_configs():
            channel_file = os.path.join(folder_path, JobsConfig.CHANNEL_FILE)
            channel_data = channel_schema.model_dump(mode="json", include=JobsConfig.CHANNEL_FIELDS)
            utils.write_yaml(channel_data, channel_file)

        if (
            scope.includes_dial_files()
            and channel_schema.details.plain_content
            and channel_schema.details.plain_content.details.file_path
        ):
            await self._export_dial_file_to_folder(
                channel_schema.details.plain_content.details.file_path,
                folder_path,
                auth_context,
            )

        return channel_db

    async def import_channel_config_from_zip(
        self,
        zip_file: zipfile.ZipFile,
        clean_up: bool,
        scope: schemas.ExportScope,
        deployment_id: str,
        auth_context: AuthContext,
    ) -> tuple[bool, models.Channel]:
        if scope.includes_dial_files():
            await self._import_dial_files_from_zip(zip_file, auth_context)

        if scope.includes_configs():
            channel_data = await self._load_channel_data_from_zip(zip_file)
            if clean_up:
                await self._cleanup_existing_channel(channel_data.deployment_id)
                existing_channel = None
            else:
                existing_channel = await self.find_channel_by_deployment_id(deployment_id)

            if existing_channel is not None:
                _log.info(
                    f"Merging import into existing channel id={existing_channel.id} "
                    f"(deployment_id={channel_data.deployment_id!r})"
                )
                is_merge = True
                channel = await self.update(
                    existing_channel.id,
                    schemas.ChannelUpdate(
                        title=channel_data.title,
                        description=channel_data.description,
                        deployment_id=channel_data.deployment_id,
                        llm_model=channel_data.llm_model,
                        details=channel_data.details,
                    ),
                )
            else:
                is_merge = False
                channel = await self.create_channel(channel_data)
            await self._session.commit()
            return is_merge, await self.get_model_by_id(channel.id)

        return True, await self.get_channel_by_deployment_id(deployment_id)

    async def _import_dial_files_from_zip(
        self, zip_file: zipfile.ZipFile, auth_context: AuthContext
    ) -> None:
        _log.info("Uploading DIAL files for channel import")
        with tempfile.TemporaryDirectory() as temp_dir:
            members = []
            for name in zip_file.namelist():
                if not name.startswith(JobsConfig.DIAL_FILES_FOLDER):
                    continue
                # Guard against zip-slip: skip any entry that would resolve
                # outside the extraction directory (CWE-22).
                try:
                    utils.safe_join(temp_dir, name)
                except ValueError:
                    _log.warning("Skipping zip entry escaping extraction dir: %r", name)
                    continue
                members.append(name)
            zip_file.extractall(temp_dir, members=members)

            dial_files_folder = os.path.join(temp_dir, JobsConfig.DIAL_FILES_FOLDER)
            for root, _, files in os.walk(dial_files_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    dial_file_path = os.path.relpath(file_path, dial_files_folder)
                    await self._import_dial_file_from_folder(
                        file_path, dial_file_path, auth_context
                    )

    async def _load_channel_data_from_zip(self, zip_file: zipfile.ZipFile) -> schemas.ChannelBase:
        with zip_file.open(JobsConfig.CHANNEL_FILE) as channel_file:
            channel_data_json = yaml.safe_load(channel_file.read())

        channel_data = schemas.ChannelBase.model_validate(channel_data_json)
        _log.info(f"Importing channel: {channel_data!r}")
        return channel_data

    async def find_channel_by_deployment_id(self, deployment_id: str) -> models.Channel | None:
        try:
            return await self.get_channel_by_deployment_id(deployment_id)
        except NoResultFound:
            return None

    async def _cleanup_existing_channel(self, deployment_id: str) -> None:
        try:
            existing_channel = await self.get_channel_by_deployment_id(deployment_id)
        except NoResultFound:
            pass  # No channel found, nothing to delete
        else:
            await self.delete(existing_channel.id)

    @staticmethod
    async def _deduplicate_collection(collection_name: str) -> DedupCounts:
        """Deduplicates a single vector store collection by document content.

        Documents with identical content are merged into a single keeper and
        the metadata references are remapped, so per-version associations are
        preserved.
        """
        _log.info(f"Deduplicating collection {collection_name!r}")
        vector_store = await VectorStoreFactory().get_embeddingless_vector_store(collection_name)
        return await vector_store.deduplicate_by_document_content()

    async def deduplicate_channel_dimensions(
        self, channel_id: int
    ) -> tuple[DedupCounts, DedupCounts]:
        """Deduplicates the non-indicator and special dimensions vector stores for the channel.

        Returns ``(non_indicator_counts, special_counts)``.
        """
        async with self._scoped_session():
            channel = await self.get_model_by_id(channel_id)
            # Extract scalar values while still inside the session scope:
            non_indicator_dims_table = channel.non_indicator_dimensions_table_name
            special_dims_table = channel.special_dimensions_table_name

        non_indicator_counts = await self._deduplicate_collection(non_indicator_dims_table)
        special_counts = await self._deduplicate_collection(special_dims_table)

        _log.info(f"Dimension deduplication completed for channel {channel_id}")
        return non_indicator_counts, special_counts

    async def trigger_deduplication(
        self, background_tasks: BackgroundTasks, channel_id: int
    ) -> schemas.DeduplicationJob:
        """Creates a DeduplicationJob, schedules the background dedup task, and returns the job."""
        channel = await self.get_model_by_id(channel_id)

        job = models.DeduplicationJob(channel_id=channel.id, status=StatusEnum.QUEUED)
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)

        background_tasks.add_task(
            deduplicate_dimensions_in_background_task,
            deduplication_job_id=job.id,
        )

        return schemas.DeduplicationJob.model_validate(job, from_attributes=True)

    async def create_deduplication_jobs(
        self, channel_ids: list[int]
    ) -> list[schemas.DeduplicationJob]:
        """Create DeduplicationJob records (status=QUEUED) for the given channels.

        Used by the batch auto-update script to persist a job per channel before
        running deduplication in parallel.

        Raises ValueError if any of the supplied channel IDs does not exist.
        """
        if not channel_ids:
            return []

        existing_result = await self._session.scalars(
            select(models.Channel.id).where(models.Channel.id.in_(channel_ids))
        )
        existing_ids = set(existing_result.all())
        missing_ids = sorted(set(channel_ids) - existing_ids)
        if missing_ids:
            raise ValueError(
                f"Cannot create deduplication jobs: channel(s) not found: {missing_ids}"
            )

        jobs: list[models.DeduplicationJob] = []
        for channel_id in channel_ids:
            job = models.DeduplicationJob(channel_id=channel_id, status=StatusEnum.QUEUED)
            self._session.add(job)
            jobs.append(job)
        await self._session.commit()
        for job in jobs:
            await self._session.refresh(job)
        return [schemas.DeduplicationJob.model_validate(job, from_attributes=True) for job in jobs]

    async def get_deduplication_job_by_id(self, job_id: int) -> schemas.DeduplicationJob:
        job: models.DeduplicationJob | None = await self._session.get(
            models.DeduplicationJob, job_id
        )
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Deduplication job not found"
            )
        return schemas.DeduplicationJob.model_validate(job, from_attributes=True)

    async def get_deduplication_jobs(
        self, channel_id: int, limit: int, offset: int
    ) -> list[schemas.DeduplicationJob]:
        query = (
            select(models.DeduplicationJob)
            .where(models.DeduplicationJob.channel_id == channel_id)
            .order_by(models.DeduplicationJob.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.scalars(query)
        return [
            schemas.DeduplicationJob.model_validate(job, from_attributes=True)
            for job in result.all()
        ]

    async def get_deduplication_jobs_count(self, channel_id: int) -> int:
        query = (
            select(func.count())
            .select_from(models.DeduplicationJob)
            .where(models.DeduplicationJob.channel_id == channel_id)
        )
        return await self._session.scalar(query) or 0

    async def _get_deduplication_job_or_raise(self, job_id: int) -> models.DeduplicationJob:
        job: models.DeduplicationJob | None = await self._session.get(
            models.DeduplicationJob, job_id
        )
        if job is None:
            raise RuntimeError(f"Deduplication job {job_id} not found")
        return job

    async def _set_deduplication_job_status(
        self,
        job: models.DeduplicationJob,
        status_value: StatusEnum,
        *,
        reason_for_failure: str | None = None,
        non_indicator_counts: DedupCounts | None = None,
        special_counts: DedupCounts | None = None,
    ) -> None:
        job.status = status_value
        if reason_for_failure is not None:
            job.reason_for_failure = reason_for_failure
        if non_indicator_counts is not None:
            job.non_indicator_remapped = non_indicator_counts.remapped
            job.non_indicator_deleted = non_indicator_counts.deleted
        if special_counts is not None:
            job.special_remapped = special_counts.remapped
            job.special_deleted = special_counts.deleted
        job.updated_at = func.now()
        await self._session.commit()
        await self._session.refresh(job)

    async def process_deduplication_job(self, deduplication_job_id: int) -> None:
        """Process a deduplication job: mark IN_PROGRESS, run dedup, write counts/status."""
        _log.info(f"Processing deduplication job {deduplication_job_id}")
        try:
            async with self._scoped_session():
                job = await self._get_deduplication_job_or_raise(deduplication_job_id)
                await self._set_deduplication_job_status(job, StatusEnum.IN_PROGRESS)
                channel_id = job.channel_id

            non_indicator_counts, special_counts = await self.deduplicate_channel_dimensions(
                channel_id
            )

            async with self._scoped_session():
                job = await self._get_deduplication_job_or_raise(deduplication_job_id)
                await self._set_deduplication_job_status(
                    job,
                    StatusEnum.COMPLETED,
                    non_indicator_counts=non_indicator_counts,
                    special_counts=special_counts,
                )
        except Exception as exc:
            _log.exception(
                f"Failed to deduplicate dimensions for deduplication job {deduplication_job_id}"
            )
            try:
                async with self._scoped_session():
                    job = await self._get_deduplication_job_or_raise(deduplication_job_id)
                    await self._set_deduplication_job_status(
                        job,
                        StatusEnum.FAILED,
                        reason_for_failure=str(exc),
                    )
            except Exception:
                _log.exception(
                    f"Failed to record FAILED status for deduplication job {deduplication_job_id}"
                )

    async def set_failed_status_for_stuck_deduplication_jobs(self) -> None:
        """Sets the status of all stuck DeduplicationJob records to FAILED.

        Reuses the same recovery helper used for Job and AutoUpdateJob, so all
        background-job types follow identical fix_statuses semantics (including
        the 12-hour staleness guard).
        """
        await set_failed_status(
            self._session,
            models.DeduplicationJob,
            models.DeduplicationJob.status,
            "status",
        )
        await self._session.commit()


@background_task
async def deduplicate_dimensions_in_background_task(deduplication_job_id: int) -> None:
    try:
        service = AdminPortalChannelService()
        await service.process_deduplication_job(deduplication_job_id=deduplication_job_id)
    except Exception:
        _log.exception(f"Failed to deduplicate dimensions (job_id={deduplication_job_id})")
