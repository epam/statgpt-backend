from datetime import datetime, timedelta
from typing import Annotated

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

import common.models as models
import common.schemas as schemas
from admin_portal.auth.auth_context import SystemUserAuthContext
from admin_portal.auth.user import User, require_jwt_auth
from admin_portal.services import AdminPortalChannelService as ChannelService
from admin_portal.services import AdminPortalDataSetService as DataSetService
from admin_portal.services import JobsService
from admin_portal.settings.exim import JobsConfig
from common.settings.dial import dial_settings

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
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.Channel:

    return await ChannelService(session).get_schema_by_id(item_id)


@router.post("/{channel_id}/export")
async def export_channel(
    background_tasks: BackgroundTasks,
    channel_id: int,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.Job:
    """Create a background job to export channel data to a zip file.
    Use the job id to check the status of the job.
    """

    return await JobsService(session).create_export_job(
        background_tasks, channel_id, auth_context=SystemUserAuthContext()
    )


IMPORT_CHANNEL_CLEAN_UP_DESCRIPTION = (
    "If enabled and a channel with the same `deployment_id` exists, it will be deleted."
    " If disabled and a channel with the same `deployment_id` exists, the import job will fail."
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
    session: AsyncSession = Depends(models.get_session),
) -> schemas.Job:
    """Create a background job to import a channel from a zip file.
    Use the job id to check the status of the job.
    """

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
) -> schemas.Job:
    """Get information (e.g. status) about the import/export job"""

    return await JobsService(session).get_job_schema_by_id(job_id)


@router.get("/jobs/{job_id}/download")
async def download_job_result_by_id(
    job_id: int,
    session: AsyncSession = Depends(models.get_session),
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

    # Code below was copied from the httpx documentation
    # https://www.python-httpx.org/async/#streaming-responses
    client = httpx.AsyncClient(
        base_url=dial_settings.url,
        headers={'Api-Key': SystemUserAuthContext().api_key},
    )
    req = client.build_request("GET", f"/v1/{job.file}")
    r = await client.send(req, stream=True)
    media_type = r.headers.get('content-type')
    return StreamingResponse(
        r.aiter_bytes(), background=BackgroundTask(r.aclose), media_type=media_type
    )


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

    await ChannelService(session).delete(item_id, auth_context=SystemUserAuthContext())


@router.get("/{channel_id}/datasets")
async def get_list_of_channel_datasets(
    channel_id: int,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
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
    reindex_indicators: bool = True,
    harmonize_indicator: bool | None = None,
    reindex_dimensions: bool = True,
    max_n_embeddings: Annotated[
        int | None,
        Query(
            description="Debugging flag that allows you to set the maximum number of documents for building embeddings.",
            ge=1,
        ),
    ] = None,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.ListResponse[schemas.ChannelDatasetBase]:
    """Clears existing indicators for all datasets in the channel and loads them from the data source.
    If any channel dataset is in the status `QUEUED` or `IN_PROGRESS`, it will be skipped.
    This endpoint only starts background jobs.
    """

    channel_datasets = await DataSetService(session).reload_all_indicators(
        background_tasks=background_tasks,
        channel_id=channel_id,
        reindex_indicators=reindex_indicators,
        harmonize_indicator=harmonize_indicator,
        reindex_dimensions=reindex_dimensions,
        max_n_embeddings=max_n_embeddings,
        auth_context=SystemUserAuthContext(),
    )
    return schemas.ListResponse[schemas.ChannelDatasetBase](
        data=channel_datasets,
        limit=len(channel_datasets),
        offset=0,
        count=len(channel_datasets),
        total=len(channel_datasets),
    )


@router.get("/{channel_id}/datasets/{dataset_id}")
async def get_channel_dataset(
    channel_id: int,
    dataset_id: int,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.ChannelDatasetBase:
    return await DataSetService(session).get_channel_dataset_schema(
        channel_id=channel_id, dataset_id=dataset_id
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
    reindex_indicators: bool = True,
    harmonize_indicator: bool | None = None,
    reindex_dimensions: bool = True,
    max_n_embeddings: Annotated[
        int | None,
        Query(
            description="Debugging flag that allows you to set the maximum number of documents for building embeddings.",
            ge=1,
        ),
    ] = None,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.ChannelDatasetBase:
    """Clears existing indicators for the dataset and loads them from the data source.
    This endpoint only starts a background job.
    """

    return await DataSetService(session).reload_indicators(
        background_tasks=background_tasks,
        channel_id=channel_id,
        dataset_id=dataset_id,
        reindex_indicators=reindex_indicators,
        harmonize_indicator=harmonize_indicator,
        reindex_dimensions=reindex_dimensions,
        max_n_embeddings=max_n_embeddings,
        auth_context=SystemUserAuthContext(),
    )


@router.delete("/{channel_id}/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_channel_dataset(
    channel_id: int,
    dataset_id: int,
    session: AsyncSession = Depends(models.get_session),
):
    await DataSetService(session).remove_channel_dataset(
        channel_id=channel_id, dataset_id=dataset_id, auth_context=SystemUserAuthContext()
    )
