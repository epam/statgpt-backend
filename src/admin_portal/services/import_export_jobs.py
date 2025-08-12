import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from typing import BinaryIO

import httpx
from fastapi import BackgroundTasks, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import func

import common.models as models
import common.schemas as schemas
from admin_portal.config import JobsConfig
from common.auth.auth_context import AuthContext
from common.config import DialConfig
from common.config import multiline_logger as logger
from common.utils import AttachmentResponse, AttachmentsStorage, attachments_storage_factory

from .channel import AdminPortalChannelService as ChannelService
from .dataset import AdminPortalDataSetService as DataSetService
from .glossary_of_terms import AdminPortalGlossaryOfTermsService as GlossaryOfTermsService


class JobsService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    async def _delete_dial_files(
        to_date: datetime,
        deleted_files: list[AttachmentResponse],
        dry_run: bool,
        auth_context: AuthContext,
    ) -> None:
        attachments_storage: AttachmentsStorage
        async with attachments_storage_factory(api_key=auth_context.api_key) as attachments_storage:
            to_date_timestamp = int(to_date.timestamp() * 1000)

            for folder in [JobsConfig.DIAL_EXPORT_FOLDER, JobsConfig.DIAL_IMPORT_FOLDER]:
                files = await attachments_storage.get_files_in_folder(f"{folder}/")
                for file in files:
                    if file.updated_at < to_date_timestamp:
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
        self, dry_run: bool, to_date: datetime, auth_contex: AuthContext
    ) -> schemas.ClearJobsResult:
        logger.info(f"Clearing jobs before {to_date}. Dry run: {dry_run}")

        deleted_files, deleted_jobs = [], []
        try:
            await self._delete_dial_files(to_date, deleted_files, dry_run, auth_contex)
            logger.info(f"Deleted {len(deleted_files)} files from DIAL")

            deleted_jobs = [
                schemas.Job.model_validate(j, from_attributes=True)
                for j in await self._delete_jobs(to_date, dry_run)
            ]
            logger.info(f"Deleted {len(deleted_jobs)} jobs from the database")

            return schemas.ClearJobsResult(deleted_files=deleted_files, deleted_jobs=deleted_jobs)
        except Exception as e:
            logger.exception(e)
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
        self, background_tasks: BackgroundTasks, channel_id: int, auth_context: AuthContext
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

        background_tasks.add_task(export_channel_in_background_task, job.id, auth_context)
        await self._update_job_status(job, schemas.PreprocessingStatusEnum.QUEUED)

        return schemas.Job.model_validate(job, from_attributes=True)

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
        await self._session.commit()

        try:
            file_type = file.filename.split(".")[-1]
            file_name = f"job-{job.id}.{file_type}"

            async with attachments_storage_factory(
                api_key=auth_context.api_key
            ) as attachments_storage:
                resp = await attachments_storage.put_file(
                    f"{JobsConfig.DIAL_IMPORT_FOLDER}/{file_name}",
                    mime_type=file.content_type,
                    content=file.file,
                )
                job.file = resp.url
        except Exception as e:
            logger.exception(e)
            job.reason_for_failure = str(e)
            await self._update_job_status(job, schemas.PreprocessingStatusEnum.FAILED)
            return schemas.Job.model_validate(job, from_attributes=True)

        logger.info(
            f"Creating import job with args: {clean_up=}, {update_datasets=}, {update_data_sources=}"
        )
        background_tasks.add_task(
            import_channel_in_background_task,
            job.id,
            clean_up,
            update_datasets,
            update_data_sources,
            auth_context,
        )
        await self._update_job_status(job, schemas.PreprocessingStatusEnum.QUEUED)

        return schemas.Job.model_validate(job, from_attributes=True)

    @staticmethod
    async def _export_data_to_folder(
        channel_id: int, data_dir: str, auth_context: AuthContext
    ) -> None:
        """Export channel data including datasets and embeddings to the folder."""

        async with models.get_session_contex_manager() as session:
            channel_service = ChannelService(session)
            channel_db = await channel_service.export_channel_to_folder(
                channel_id, data_dir, auth_context
            )

            glossary_service = GlossaryOfTermsService(session)
            await glossary_service.export_glossary_to_folder(channel_db, data_dir)

            dataset_service = DataSetService(session)
            await dataset_service.export_datasets(channel_db, data_dir, auth_context=auth_context)

    async def export_channel_in_background(self, job_id: int, auth_context: AuthContext) -> None:
        logger.info(f"Exporting channel data to zip file. Job id={job_id}")
        job: models.Job = await self.get_job_model_by_id(job_id)
        await self._update_job_status(job, schemas.PreprocessingStatusEnum.IN_PROGRESS)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # folder for channel data before zipping:
                data_dir = os.path.join(tmp_dir, "data")
                os.makedirs(data_dir)

                await self._export_data_to_folder(
                    job.channel_id, data_dir, auth_context=auth_context
                )

                res_file_path = os.path.abspath(os.path.join(tmp_dir, f"job-{job.id}"))
                path = shutil.make_archive(res_file_path, 'zip', data_dir)

                attachments_storage: AttachmentsStorage
                async with attachments_storage_factory(
                    api_key=auth_context.api_key
                ) as attachments_storage:
                    resp = await attachments_storage.put_local_file(
                        f"{JobsConfig.DIAL_EXPORT_FOLDER}/{os.path.basename(path)}", path
                    )
                    file_url = resp.url
        except Exception as e:
            logger.exception(e)
            job.reason_for_failure = str(e)
            await self._update_job_status(job, schemas.PreprocessingStatusEnum.FAILED)
            return

        job.file = file_url
        await self._update_job_status(job, schemas.PreprocessingStatusEnum.COMPLETED)

    @staticmethod
    async def download_zip_file(
        file_url: str, zip_file: BinaryIO, auth_context: AuthContext
    ) -> None:
        """TODO: Perhaps this method should be moved in Dial core or attachments module."""
        client = httpx.AsyncClient(
            base_url=DialConfig.get_url(),
            headers={'Api-Key': auth_context.api_key},
        )
        async with client.stream('GET', f"/v1/{file_url}") as response:
            async for chunk in response.aiter_bytes():
                zip_file.write(chunk)

    @staticmethod
    async def _import_data_from_zip(
        zip_file: zipfile.ZipFile,
        clean_up: bool,
        update_datasets: bool,
        update_data_sources: bool,
        auth_context: AuthContext,
    ) -> int:
        """Import channel data including datasets and embeddings from the zip file."""

        async with models.get_session_contex_manager() as session:
            channel_service = ChannelService(session)
            channel_db = await channel_service.import_channel_from_zip(
                zip_file, clean_up, auth_context
            )

            glossary_service = GlossaryOfTermsService(session)
            await glossary_service.import_glossary_from_zip(zip_file, channel_db.id)

            dataset_service = DataSetService(session)
            await dataset_service.import_datasets_and_data_sources_from_zip(
                channel_db,
                zip_file,
                update_datasets,
                update_data_sources,
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
    ) -> None:
        logger.info(f"Importing channel from zip file. Job id={job_id}")
        job: models.Job = await self.get_job_model_by_id(job_id)
        await self._update_job_status(job, schemas.PreprocessingStatusEnum.IN_PROGRESS)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_file_path = os.path.join(tmp_dir, "import.zip")

                with open(zip_file_path, "wb") as zip_file:
                    await self.download_zip_file(
                        file_url=job.file, zip_file=zip_file, auth_context=auth_context
                    )

                with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
                    channel_id = await self._import_data_from_zip(
                        zip_file, clean_up, update_datasets, update_data_sources, auth_context
                    )
        except Exception as e:
            logger.exception(e)
            job.reason_for_failure = str(e)
            await self._update_job_status(job, schemas.PreprocessingStatusEnum.FAILED)
            return

        job.channel_id = channel_id
        await self._update_job_status(job, schemas.PreprocessingStatusEnum.COMPLETED)
        logger.info(f"Channel(id={channel_id}) imported successfully. Job id={job_id}")


async def export_channel_in_background_task(job_id: int, auth_context: AuthContext) -> None:
    try:
        async with models.get_session_contex_manager() as session:
            service = JobsService(session)
            await service.export_channel_in_background(job_id=job_id, auth_context=auth_context)
    except Exception as e:
        logger.exception(e)


async def import_channel_in_background_task(
    job_id: int,
    clean_up: bool,
    update_datasets: bool,
    update_data_sources: bool,
    auth_context: AuthContext,
) -> None:
    try:
        async with models.get_session_contex_manager() as session:
            service = JobsService(session)
            await service.import_channel_in_background(
                job_id=job_id,
                clean_up=clean_up,
                update_datasets=update_datasets,
                update_data_sources=update_data_sources,
                auth_context=auth_context,
            )
    except Exception as e:
        logger.exception(e)
