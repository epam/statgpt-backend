import asyncio
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from typing import BinaryIO

from aidial_client.types.metadata import FileItem
from fastapi import BackgroundTasks, HTTPException, UploadFile, status
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import func

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.audit.context import AuditContext, get_audit_context, update_audit_context
from statgpt.admin.audit.decorators import audit_action
from statgpt.admin.settings.exim import JobsConfig
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import AuditActionType, AuditEntityType
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils import (
    AttachmentsStorage,
    attachments_storage_factory,
    dial_client_factory,
    format_exception_reason,
    write_file_to,
)

from .channel import AdminPortalChannelService as ChannelService
from .dataset import AdminPortalDataSetService as DataSetService
from .glossary_of_terms import AdminPortalGlossaryOfTermsService as GlossaryOfTermsService

_log = logging.getLogger(__name__)


class ExportMetadata(BaseModel):
    export_version: int | None = None
    export_start_datetime: str
    export_finish_datetime: str
    scope: schemas.ExportScope = schemas.ExportScope.FULL
    deployment_id: str


class JobsService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    async def _delete_dial_files(
        to_date: datetime,
        deleted_files: list[FileItem],
        dry_run: bool,
        auth_context: AuthContext,
    ) -> None:
        attachments_storage: AttachmentsStorage
        async with attachments_storage_factory(api_key=auth_context.api_key) as attachments_storage:
            to_date_timestamp = int(to_date.timestamp() * 1000)

            for folder in [JobsConfig.DIAL_EXPORT_FOLDER, JobsConfig.DIAL_IMPORT_FOLDER]:
                files = await attachments_storage.get_files_in_folder(folder)
                for file in files:
                    if file.updated_at is not None and file.updated_at < to_date_timestamp:
                        if not dry_run:
                            await attachments_storage.delete_file(file.url)
                        deleted_files.append(file)

    async def _get_jobs_models(self) -> list[models.Job]:
        query = select(models.Job)
        q_result = await self._session.execute(query)
        return [item for item in q_result.scalars().all()]

    async def _delete_jobs(self, to_date: datetime, dry_run: bool) -> list[models.Job]:
        jobs = await self._get_jobs_models()

        jobs = [j for j in jobs if j.updated_at.timestamp() < to_date.timestamp()]

        if not dry_run and jobs:
            for item in jobs:
                await self._session.delete(item)
            await self._session.commit()

        return jobs

    async def clear_jobs(
        self, dry_run: bool, to_date: datetime, auth_context: AuthContext
    ) -> schemas.ClearJobsResult:
        _log.info(f"Clearing jobs before {to_date}. Dry run: {dry_run}")

        deleted_files: list[FileItem] = []
        deleted_jobs: list[schemas.Job] = []
        try:
            await self._delete_dial_files(to_date, deleted_files, dry_run, auth_context)
            _log.info(f"Deleted {len(deleted_files)} files from DIAL")

            deleted_jobs = [
                schemas.Job.model_validate(j, from_attributes=True)
                for j in await self._delete_jobs(to_date, dry_run)
            ]
            _log.info(f"Deleted {len(deleted_jobs)} jobs from the database")

            return schemas.ClearJobsResult(deleted_files=deleted_files, deleted_jobs=deleted_jobs)
        except Exception as e:
            _log.exception(e)
            return schemas.ClearJobsResult(
                reason_for_failure=str(e), deleted_files=deleted_files, deleted_jobs=deleted_jobs
            )

    async def get_job_model_by_id(self, job_id: int) -> models.Job:
        job: models.Job | None = await self._session.get(models.Job, job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job with id={job_id} not found"
            )
        return job

    async def get_jobs_count(self, channel_id: int) -> int:
        query = (
            select(func.count("*"))
            .select_from(models.Job)
            .where(models.Job.channel_id == channel_id)
        )
        return (await self._session.execute(query)).scalar_one()

    async def get_jobs_schemas(
        self, channel_id: int, limit: int | None, offset: int
    ) -> list[schemas.Job]:
        query = (
            select(models.Job)
            .where(models.Job.channel_id == channel_id)
            .limit(limit)
            .offset(offset)
            .order_by(models.Job.updated_at.desc())
        )
        q_result = await self._session.execute(query)
        jobs = [item for item in q_result.scalars().all()]
        return [schemas.Job.model_validate(item, from_attributes=True) for item in jobs]

    async def get_job_schema_by_id(self, job_id: int) -> schemas.Job:
        job = await self.get_job_model_by_id(job_id)
        return schemas.Job.model_validate(job, from_attributes=True)

    async def _update_job_status(
        self, job: models.Job, new_status: schemas.PreprocessingStatusEnum
    ) -> None:
        job.status = new_status
        job.updated_at = func.now()
        await self._session.commit()
        await self._session.refresh(job)

    async def create_export_job(
        self,
        background_tasks: BackgroundTasks,
        channel_id: int,
        scope: schemas.ExportScope,
        auth_context: AuthContext,
    ) -> schemas.Job:
        channel_service = ChannelService(self._session)
        channel_db = await channel_service.get_model_by_id(channel_id)

        job = models.Job(
            type=schemas.JobType.EXPORT,
            status=schemas.PreprocessingStatusEnum.NOT_STARTED,
            channel_id=channel_db.id,
        )
        self._session.add(job)
        await self._session.commit()

        background_tasks.add_task(export_channel_in_background_task, job.id, scope, auth_context)
        await self._update_job_status(job, schemas.PreprocessingStatusEnum.QUEUED)

        return schemas.Job.model_validate(job, from_attributes=True)

    @audit_action(entity_type=AuditEntityType.IMPORT_JOB, action_type=AuditActionType.CREATE)
    async def create_import_job(
        self,
        background_tasks: BackgroundTasks,
        file: UploadFile,
        clean_up: bool,
        update_datasets: bool,
        update_data_sources: bool,
        auth_context: AuthContext,
    ) -> schemas.Job:
        job = models.Job(
            type=schemas.JobType.IMPORT,
            status=schemas.PreprocessingStatusEnum.NOT_STARTED,
        )
        self._session.add(job)
        await self._session.flush()

        try:
            if not file.filename or not file.content_type:
                raise ValueError("File must have a filename and content type")
            file_type = file.filename.split(".")[-1]
            file_name = f"job-{job.id}.{file_type}"

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = os.path.join(tmp_dir, file_name)
                await file.seek(0)
                with open(tmp_path, "wb") as out:
                    while chunk := await file.read(1024 * 1024):
                        out.write(chunk)

                async with attachments_storage_factory(
                    api_key=auth_context.api_key
                ) as attachments_storage:
                    resp = await attachments_storage.put_local_file(
                        f"{JobsConfig.DIAL_IMPORT_FOLDER}/{file_name}",
                        tmp_path,
                    )
                    job.file = resp.url

            _log.info(
                f"Creating import job with args: {clean_up=}, {update_datasets=}, {update_data_sources=}"
            )
            background_tasks.add_task(
                import_channel_in_background_task,
                job.id,
                clean_up,
                update_datasets,
                update_data_sources,
                auth_context,
                get_audit_context(),
            )
            job.status = schemas.PreprocessingStatusEnum.QUEUED
        except Exception as e:
            _log.exception(e)
            job.reason_for_failure = format_exception_reason(e)
            job.status = schemas.PreprocessingStatusEnum.FAILED

        job.updated_at = func.now()
        await self._session.flush()
        await self._session.refresh(job)
        return schemas.Job.model_validate(job, from_attributes=True)

    @staticmethod
    async def _export_data_to_folder(
        channel_id: int, data_dir: str, scope: schemas.ExportScope, auth_context: AuthContext
    ) -> str:
        """Export channel data including datasets and embeddings to the folder."""

        async with models.get_readonly_session_context_manager() as session:
            channel_service = ChannelService(session)
            channel_db = await channel_service.export_channel_to_folder(
                channel_id, data_dir, scope=scope, auth_context=auth_context
            )

            if scope is schemas.ExportScope.CONFIGS or scope is schemas.ExportScope.FULL:
                glossary_service = GlossaryOfTermsService(session)
                await glossary_service.export_glossary_to_folder(channel_db, data_dir)

            dataset_service = DataSetService(session)
            await dataset_service.export_datasets(
                channel_db, data_dir, scope=scope, auth_context=auth_context
            )

            return channel_db.deployment_id

    async def export_channel_in_background(
        self, job_id: int, scope: schemas.ExportScope, auth_context: AuthContext
    ) -> schemas.Job:
        _log.info(f"Exporting channel data to zip file. Job id={job_id}")
        job: models.Job = await self.get_job_model_by_id(job_id)
        await self._update_job_status(job, schemas.PreprocessingStatusEnum.IN_PROGRESS)

        try:
            export_start_datetime = datetime.now().isoformat()
            with tempfile.TemporaryDirectory() as tmp_dir:
                # folder for channel data before zipping:
                data_dir = os.path.join(tmp_dir, "data")
                os.makedirs(data_dir)

                if not job.channel_id:
                    raise ValueError("Job must have a channel_id to export data")
                deployment_id = await self._export_data_to_folder(
                    job.channel_id, data_dir, scope=scope, auth_context=auth_context
                )

                archive_name = f"{deployment_id}-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f')}"

                export_finish_datetime = datetime.now().isoformat()
                metadata = ExportMetadata(
                    export_version=JobsConfig.CURRENT_EXPORT_VERSION,
                    export_start_datetime=export_start_datetime,
                    export_finish_datetime=export_finish_datetime,
                    scope=scope,
                    deployment_id=deployment_id,
                )
                metadata_path = os.path.join(data_dir, "metadata.json")
                with open(metadata_path, "w", encoding="utf-8") as meta_file:
                    meta_file.write(metadata.model_dump_json(indent=2))

                res_file_path = os.path.abspath(os.path.join(tmp_dir, archive_name))
                _log.info(f"Compressing {data_dir} to {res_file_path}.zip (non-blocking)")
                path = await asyncio.to_thread(shutil.make_archive, res_file_path, 'zip', data_dir)
                _log.info(f"Compression completed: {path}")

                attachments_storage: AttachmentsStorage
                async with attachments_storage_factory(
                    api_key=auth_context.api_key
                ) as attachments_storage:
                    resp = await attachments_storage.put_local_file(
                        f"{JobsConfig.DIAL_EXPORT_FOLDER}/{os.path.basename(path)}",
                        path,
                        show_progress=True,
                    )
                    file_url = resp.url
        except Exception as e:
            _log.exception(e)
            job.reason_for_failure = format_exception_reason(e)
            await self._update_job_status(job, schemas.PreprocessingStatusEnum.FAILED)
            return schemas.Job.model_validate(job, from_attributes=True)

        job.file = file_url
        await self._update_job_status(job, schemas.PreprocessingStatusEnum.COMPLETED)
        return schemas.Job.model_validate(job, from_attributes=True)

    @staticmethod
    async def download_zip_file(
        file_url: str, zip_file: BinaryIO, auth_context: AuthContext
    ) -> None:
        async with dial_client_factory(
            base_url=dial_settings.url, api_key=auth_context.api_key
        ) as dial:
            await write_file_to(dial, file_url, zip_file)

    @staticmethod
    def _validate_export_version(metadata: ExportMetadata) -> None:
        """Validate that the archive export version is supported.

        Raises ValueError with a user-friendly message when the version is
        missing (legacy archive) or not in the supported set.
        """
        version = metadata.export_version
        supported = JobsConfig.SUPPORTED_EXPORT_VERSIONS

        if version is None:
            raise ValueError(
                "The archive does not contain export version information. "
                "It was likely created by an older version of the application "
                "and may be incompatible. "
                "Please re-export the channel with the current version of the application."
            )

        if version not in supported:
            supported_str = ", ".join(str(v) for v in sorted(supported))
            raise ValueError(
                f"Unsupported archive version: {version}. "
                f"This application supports archive versions: {supported_str}. "
                f"Please re-export the channel with a compatible version of the application."
            )

    async def _import_data_from_zip(
        self,
        job: models.Job,
        zip_file: zipfile.ZipFile,
        clean_up: bool,
        update_datasets: bool,
        update_data_sources: bool,
        auth_context: AuthContext,
    ) -> int:
        """Import channel data including datasets and embeddings from the zip file."""

        async with models.get_session_context_manager() as session:
            deployment_id = None
            scope = schemas.ExportScope.FULL
            try:
                with zip_file.open("metadata.json") as meta_file:
                    metadata = ExportMetadata.model_validate_json(meta_file.read())
                    self._validate_export_version(metadata)
                    deployment_id = metadata.deployment_id
                    scope = metadata.scope
            except KeyError:
                raise ValueError(
                    "The archive does not contain metadata.json. "
                    "It may be corrupted or created by an incompatible version of the application. "
                    "Please re-export the channel with the current version of the application."
                )
            except ValidationError as e:
                raise ValueError(
                    "The archive contains invalid metadata. "
                    "It may be corrupted or created by an incompatible version of the application. "
                    "Please re-export the channel with the current version of the application."
                ) from e

            channel_service = ChannelService(session)
            channel_db = await channel_service.import_channel_from_zip(
                zip_file,
                clean_up,
                scope=scope,
                deployment_id=deployment_id,
                auth_context=auth_context,
            )

            job.channel_id = channel_db.id
            await self._update_job_status(job, schemas.PreprocessingStatusEnum.IN_PROGRESS)
            if scope.includes_configs():
                glossary_service = GlossaryOfTermsService(session)
                await glossary_service.import_glossary_from_zip(zip_file, channel_db.id)

            dataset_service = DataSetService(session)
            await dataset_service.import_datasets_and_data_sources_from_zip(
                channel_db,
                zip_file,
                update_datasets,
                update_data_sources,
                scope=scope,
                auth_context=auth_context,
            )

            return channel_db.id

    async def import_channel_in_background(
        self,
        job_id: int,
        clean_up: bool,
        update_datasets: bool,
        update_data_sources: bool,
        auth_context: AuthContext,
    ) -> schemas.Job:
        _log.info(f"Importing channel from zip file. Job id={job_id}")
        job: models.Job = await self.get_job_model_by_id(job_id)
        await self._update_job_status(job, schemas.PreprocessingStatusEnum.IN_PROGRESS)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_file_path = os.path.join(tmp_dir, "import.zip")

                if not job.file:
                    raise ValueError("Job must have a file url to import data")

                with open(zip_file_path, "wb") as zip_file:
                    await self.download_zip_file(
                        file_url=job.file,
                        zip_file=zip_file,
                        auth_context=auth_context,
                    )

                with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
                    channel_id = await self._import_data_from_zip(
                        job, zip_file, clean_up, update_datasets, update_data_sources, auth_context
                    )
        except Exception as e:
            _log.exception(e)
            job.reason_for_failure = format_exception_reason(e)
            await self._update_job_status(job, schemas.PreprocessingStatusEnum.FAILED)
            return schemas.Job.model_validate(job, from_attributes=True)

        await self._update_job_status(job, schemas.PreprocessingStatusEnum.COMPLETED)
        _log.info(f"Channel(id={channel_id}) imported successfully. Job id={job_id}")
        return schemas.Job.model_validate(job, from_attributes=True)


async def export_channel_in_background_task(
    job_id: int, scope: schemas.ExportScope, auth_context: AuthContext
) -> None:
    try:
        async with models.get_session_context_manager() as session:
            service = JobsService(session)
            await service.export_channel_in_background(
                job_id=job_id, scope=scope, auth_context=auth_context
            )
    except Exception as e:
        _log.exception(e)


async def import_channel_in_background_task(
    job_id: int,
    clean_up: bool,
    update_datasets: bool,
    update_data_sources: bool,
    auth_context: AuthContext,
    audit_context: AuditContext,
) -> None:
    try:
        update_audit_context(audit_context)
        async with models.get_session_context_manager() as session:
            service = JobsService(session)
            await service.import_channel_in_background(
                job_id=job_id,
                clean_up=clean_up,
                update_datasets=update_datasets,
                update_data_sources=update_data_sources,
                auth_context=auth_context,
            )
    except Exception as e:
        _log.exception(e)
