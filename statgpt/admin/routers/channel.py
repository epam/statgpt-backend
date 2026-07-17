import logging
from datetime import datetime, timedelta
from typing import Annotated

from aidial_client import ResourceNotFoundError
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.auth.user import require_jwt_auth
from statgpt.admin.services import AdminPortalChannelService as ChannelService
from statgpt.admin.services import AdminPortalDataSetService as DataSetService
from statgpt.admin.services import JobsService
from statgpt.admin.settings.exim import JobsConfig
from statgpt.common.data.sdmx.v21.dataset import InvalidConfigurationError
from statgpt.common.models.database import get_session_context_manager
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils.cancel_dependency import cancel_on_disconnect
from statgpt.common.utils.dial import open_file_stream

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/channels",
    tags=["channels"],
    dependencies=[Depends(require_jwt_auth, use_cache=False)],
)


@router.get("")
async def get_channels(
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ListResponse[schemas.Channel]:
    """Returns a list of channels"""

    service = ChannelService(session)
    channels = await service.get_channels_schemas(limit=limit, offset=offset)
    channels_count = await service.get_channels_count()

    return schemas.ListResponse[schemas.Channel](
        data=channels,
        limit=limit,
        offset=offset,
        count=len(channels),
        total=channels_count,
    )


@router.post("")
async def create_channel(
    data: schemas.ChannelBase,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.Channel:
    """Create a new channel"""

    return await ChannelService(session).create_channel(data)


@router.get("/{item_id}")
async def get_channel_by_id(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.Channel:

    return await ChannelService(session).get_schema_by_id(item_id)


@router.post("/{channel_id}/export")
async def export_channel(
    background_tasks: BackgroundTasks,
    channel_id: int,
    scope: schemas.ExportScope = Query(default=schemas.ExportScope.FULL),
) -> schemas.Job:
    """Create a background job to export channel data to a zip file.
    Use the job id to check the status of the job.
    """

    async with get_session_context_manager() as session:
        return await JobsService(session).create_export_job(
            background_tasks, channel_id, scope=scope, auth_context=SystemUserAuthContext()
        )


IMPORT_CHANNEL_CLEAN_UP_DESCRIPTION = (
    "If enabled and a channel with the same `deployment_id` exists, it will be deleted"
    " and rebuilt from scratch."
    " If disabled and a channel with the same `deployment_id` exists, the archive is merged"
    " into it: configs and datasets are updated in place, new datasets and indexes are added,"
    " and dimension indexes are deduplicated afterwards."
)


@router.post("/import")
async def import_channel(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    clean_up: Annotated[bool, Query(description=IMPORT_CHANNEL_CLEAN_UP_DESCRIPTION)] = False,
    update_datasets: Annotated[
        bool, Query(description='Whether to update the datasets if it already exists')
    ] = False,
    update_data_sources: Annotated[
        bool, Query(description='Whether to update the data sources if it already exists')
    ] = False,
) -> schemas.Job:
    """Create a background job to import a channel from a zip file.
    Use the job id to check the status of the job.
    """

    async with get_session_context_manager() as session:
        return await JobsService(session).create_import_job(
            background_tasks,
            file,
            clean_up,
            update_datasets,
            update_data_sources,
            auth_context=SystemUserAuthContext(),
        )


@router.get('/{channel_id}/jobs')
async def get_jobs(
    channel_id: int,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ListResponse[schemas.Job]:
    """Get a list of import/export jobs for the specified channel"""

    service = JobsService(session)
    jobs = await service.get_jobs_schemas(channel_id=channel_id, limit=limit, offset=offset)
    jobs_count = await service.get_jobs_count(channel_id=channel_id)

    return schemas.ListResponse[schemas.Job](
        data=jobs,
        limit=limit,
        offset=offset,
        count=len(jobs),
        total=jobs_count,
    )


@router.get("/jobs/{job_id}")
async def get_job_by_id(
    job_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.Job:
    """Get information (e.g. status) about the import/export job"""

    return await JobsService(session).get_job_schema_by_id(job_id)


@router.get("/jobs/{job_id}/download")
async def download_job_result_by_id(
    job_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> StreamingResponse:
    """Download the zip file with the exported channel data by job id.

    The job must be of type `EXPORT` and have status `COMPLETED`.
    """

    job = await JobsService(session).get_job_model_by_id(job_id)

    if job.type != schemas.JobType.EXPORT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job with id={job_id} is not an export job",
        )

    if job.status != schemas.PreprocessingStatusEnum.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job with id={job_id} is not completed",
        )

    if not job.file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id={job_id} has no associated file",
        )

    try:
        stream, media_type, aclose = await open_file_stream(
            dial_settings.url, SystemUserAuthContext().api_key, job.file
        )
    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File of the job with id={job_id} was not found in the file storage",
        )
    return StreamingResponse(stream, media_type=media_type, background=BackgroundTask(aclose))


DRY_RUN_DESCRIPTION = (
    "If true, the jobs and files will not be deleted. But the result will be returned."
    " Using this flag, you can check what will be deleted without actually deleting anything."
)

OLDER_THAN_DESCRIPTION = (
    "Only jobs and files older than the number of hours specified here will be deleted."
)


@router.delete("/jobs")
async def clear_jobs(
    dry_run: Annotated[bool, Query(description=DRY_RUN_DESCRIPTION)] = False,
    older_than: Annotated[
        int, Query(description=OLDER_THAN_DESCRIPTION)
    ] = JobsConfig.JOBS_RETENTION_HOURS,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.ClearJobsResult:
    """Clear all jobs and files updated before the specified datetime."""

    to_date = datetime.now() - timedelta(hours=older_than)
    return await JobsService(session).clear_jobs(
        dry_run, to_date, auth_context=SystemUserAuthContext()
    )


@router.post("/{item_id}")
async def update_channel(
    item_id: int,
    data: schemas.ChannelUpdate,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.Channel:
    """Update channel name, description or deployment_id"""

    return await ChannelService(session).update(item_id, data)


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_channel(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
) -> None:
    """Delete channel by id"""

    await ChannelService(session).delete(item_id)


@router.get("/{channel_id}/datasets")
async def get_list_of_channel_datasets(
    channel_id: int,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ListResponse[schemas.ChannelDatasetExpanded]:
    """Returns a list of datasets for the specified channel"""

    service = DataSetService(session)
    channel_datasets = await service.get_channel_dataset_schemas(
        limit=limit,
        offset=offset,
        channel_id=channel_id,
        auth_context=SystemUserAuthContext(),
    )
    total_count = await service.get_channel_datasets_count(channel_id=channel_id)

    return schemas.ListResponse[schemas.ChannelDatasetExpanded](
        data=channel_datasets,
        limit=limit,
        offset=offset,
        count=len(channel_datasets),
        total=total_count,
    )


@router.post(
    "/{channel_id}/datasets/reload-indicators",
    status_code=status.HTTP_202_ACCEPTED,
)
async def reload_indicators_for_all_channel_datasets(
    background_tasks: BackgroundTasks,
    channel_id: int,
    max_n_embeddings: Annotated[
        int | None,
        Query(
            description="Debugging flag that allows you to set the maximum number of documents for building embeddings.",
            ge=1,
        ),
    ] = None,
) -> schemas.ListResponse[schemas.ChannelDatasetExpanded]:
    """Clears existing indicators for all datasets in the channel and loads them from the data source.
    If any channel dataset is in the status `QUEUED` or `IN_PROGRESS`, it will be skipped.
    This endpoint only starts background jobs.
    """

    async with get_session_context_manager() as session:
        try:
            channel_datasets = await DataSetService(session).reload_all_indicators(
                background_tasks=background_tasks,
                channel_id=channel_id,
                max_n_embeddings=max_n_embeddings,
                auth_context=SystemUserAuthContext(),
            )
        except ExceptionGroup as eg:
            invalid_config_errors = [
                e for e in eg.exceptions if isinstance(e, InvalidConfigurationError)
            ]
            if invalid_config_errors:
                for err in eg.exceptions:
                    logger.error("Error during channel reindex", exc_info=err)
                detail = [e.to_dict() for e in invalid_config_errors]
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
            raise

    return schemas.ListResponse[schemas.ChannelDatasetExpanded](
        data=channel_datasets,
        limit=len(channel_datasets),
        offset=0,
        count=len(channel_datasets),
        total=len(channel_datasets),
    )


@router.post(
    path="/{channel_id}/datasets/deduplicate",
    status_code=status.HTTP_202_ACCEPTED,
)
async def deduplicate_channel(
    background_tasks: BackgroundTasks,
    channel_id: int,
) -> schemas.DeduplicationJob:
    """Deduplicates the non-indicator and special dimensions vector stores for the channel.

    Creates a deduplication job, schedules the actual work in the background, and
    returns the job immediately with status 202 Accepted. Poll the job status via
    ``GET /channels/deduplication-jobs/{job_id}`` to observe progress and retrieve
    per-dimension counts of remapped rows and deleted orphan documents.
    """
    async with get_session_context_manager() as session:
        service = ChannelService(session)
        return await service.trigger_deduplication(
            background_tasks=background_tasks, channel_id=channel_id
        )


@router.get("/{channel_id}/datasets/deduplication-jobs")
async def get_deduplication_jobs(
    channel_id: int,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ListResponse[schemas.DeduplicationJob]:
    """Get a paginated list of deduplication jobs for a channel."""
    service = ChannelService(session)
    jobs = await service.get_deduplication_jobs(channel_id=channel_id, limit=limit, offset=offset)
    total = await service.get_deduplication_jobs_count(channel_id)
    return schemas.ListResponse[schemas.DeduplicationJob](
        data=jobs,
        limit=limit,
        offset=offset,
        count=len(jobs),
        total=total,
    )


@router.get("/deduplication-jobs/{job_id}")
async def get_deduplication_job_by_id(
    job_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.DeduplicationJob:
    """Get a deduplication job by ID for polling status."""
    return await ChannelService(session).get_deduplication_job_by_id(job_id)


@router.get(path="/{channel_id}/index-status")
async def get_index_status(
    channel_id: int,
    scope: schemas.ChannelIndexStatusScope,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ChannelIndexStatus:
    """Get the index status for the specified channel."""

    service = DataSetService(session)
    return await service.check_index_status(channel_id=channel_id, scope=scope)


@router.get("/{channel_id}/datasets/{dataset_id}")
async def get_channel_dataset(
    channel_id: int,
    dataset_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ChannelDatasetExpanded:
    return await DataSetService(session).get_channel_dataset_schema(
        channel_id=channel_id, dataset_id=dataset_id, auth_context=SystemUserAuthContext()
    )


@router.post("/{channel_id}/datasets/{dataset_id}")
async def add_dataset_to_channel(
    channel_id: int,
    dataset_id: int,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.ChannelDatasetBase:
    return await DataSetService(session).add_dataset_to_channel(
        channel_id=channel_id, dataset_id=dataset_id
    )


@router.post(
    "/{channel_id}/datasets/{dataset_id}/reload-indicators",
    status_code=status.HTTP_202_ACCEPTED,
)
async def reload_indicators_for_channel_dataset(
    background_tasks: BackgroundTasks,
    channel_id: int,
    dataset_id: int,
    max_n_embeddings: Annotated[
        int | None,
        Query(
            description="Debugging flag that allows you to set the maximum number of documents for building embeddings.",
            ge=1,
        ),
    ] = None,
) -> schemas.ChannelDatasetExpanded:
    """Clears existing indicators for the dataset and loads them from the data source.
    This endpoint only starts a background job.
    """

    async with get_session_context_manager() as session:
        try:
            return await DataSetService(session).reload_indicators(
                background_tasks=background_tasks,
                channel_id=channel_id,
                dataset_id=dataset_id,
                max_n_embeddings=max_n_embeddings,
                auth_context=SystemUserAuthContext(),
            )
        except InvalidConfigurationError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.to_dict())


@router.delete("/{channel_id}/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_channel_dataset(channel_id: int, dataset_id: int):
    await DataSetService().remove_channel_dataset(channel_id=channel_id, dataset_id=dataset_id)


@router.get("/{channel_id}/datasets/{dataset_id}/versions")
async def get_channel_dataset_versions(
    channel_id: int,
    dataset_id: int,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ListResponse[schemas.ChannelDatasetVersion]:
    """Returns a list of dataset versions for the specified channel and dataset"""

    service = DataSetService(session)
    versions = await service.get_channel_dataset_versions_schemas(
        limit=limit,
        offset=offset,
        channel_id=channel_id,
        dataset_id=dataset_id,
    )
    total_count = await service.get_channel_dataset_versions_count(
        channel_id=channel_id, dataset_id=dataset_id
    )

    return schemas.ListResponse[schemas.ChannelDatasetVersion](
        data=versions,
        limit=limit,
        offset=offset,
        count=len(versions),
        total=total_count,
    )


@router.get(path="/{channel_id}/datasets/{dataset_id}/versions/check-latest-up-to-date")
async def is_channel_dataset_latest_version_up_to_date(
    channel_id: int,
    dataset_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ChangesBetweenVersionAndActualData:
    """Check if the latest completed version of the specified channel dataset is up to date."""
    return await DataSetService(session).is_channel_dataset_latest_version_up_to_date(
        channel_id=channel_id, dataset_id=dataset_id, auth_context=SystemUserAuthContext()
    )


@router.post(path="/{channel_id}/datasets/{dataset_id}/versions/rollback")
async def rollback_channel_dataset_to_previous_version(
    channel_id: int,
    dataset_id: int,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.ChannelDatasetVersion:
    """Rolls back the specified dataset in the channel to a previous 'COMPLETED' version."""
    return await DataSetService(session).rollback_channel_dataset_to_previous_version(
        channel_id=channel_id, dataset_id=dataset_id
    )


@router.delete(
    path="/{channel_id}/datasets/{dataset_id}/versions/clear-data",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def clear_channel_dataset_versions_data(
    channel_id: int,
    dataset_id: int,
    session: AsyncSession = Depends(models.get_session),
):
    """Clears the data for all versions except the latest completed one for the specified dataset in the channel."""
    await DataSetService(session).clear_channel_dataset_versions_data(
        channel_id=channel_id, dataset_id=dataset_id
    )


@router.post(
    path="/{channel_id}/datasets/{dataset_id}/versions/auto-update-jobs",
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_auto_update(
    background_tasks: BackgroundTasks,
    channel_id: int,
    dataset_id: int,
) -> schemas.AutoUpdateJob:
    """Trigger an auto-update check for a channel dataset.

    Creates an auto-update job that checks for changes and reindexes if needed.
    The job runs in the background. Poll the job status to track progress.
    """
    async with get_session_context_manager() as session:
        return await DataSetService(session).trigger_auto_update(
            background_tasks=background_tasks,
            channel_id=channel_id,
            dataset_id=dataset_id,
            auth_context=SystemUserAuthContext(),
        )


@router.get(path="/{channel_id}/datasets/{dataset_id}/versions/auto-update-jobs")
async def get_auto_update_jobs(
    channel_id: int,
    dataset_id: int,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ListResponse[schemas.AutoUpdateJob]:
    """Get a paginated list of auto-update jobs for a channel dataset."""
    service = DataSetService(session)
    channel_dataset = await service.get_channel_dataset_model_or_raise(
        channel_id=channel_id, dataset_id=dataset_id
    )
    jobs = await service.get_auto_update_jobs(
        channel_dataset_id=channel_dataset.id,
        limit=limit,
        offset=offset,
    )
    total = await service.get_auto_update_jobs_count(channel_dataset.id)
    return schemas.ListResponse[schemas.AutoUpdateJob](
        data=jobs,
        limit=limit,
        offset=offset,
        count=len(jobs),
        total=total,
    )


@router.get(path="/auto-update-jobs/{job_id}")
async def get_auto_update_job_by_id(
    job_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.AutoUpdateJob:
    """Get an auto-update job by ID for polling status."""
    return await DataSetService(session).get_auto_update_job_by_id(job_id)
