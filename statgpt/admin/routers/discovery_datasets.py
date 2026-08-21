"""Administrator REST surface for Grade C discovery datasets.

A discovery dataset makes a dataset discoverable without onboarding it: an agency
describes the datasets it publishes, StatGPT indexes those descriptions, and the agent
refers users to the official source. Records belong to a channel and are uploaded from a
filled discovery workbook (.xlsx) or its CSV equivalent.

Records are shaped like glossary terms: the channel-scoped routes below cover a channel's
collection, while a single record is addressed globally under `/discovery-datasets`.
"""

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.auth.user import require_jwt_auth
from statgpt.admin.services import AdminPortalDiscoveryDatasetService as DiscoveryDatasetService
from statgpt.admin.services import AdminPortalDiscoveryIndexingJobService as IndexingJobService
from statgpt.common.models import get_session_context_manager
from statgpt.common.utils.cancel_dependency import cancel_on_disconnect

channel_discovery_datasets_router = APIRouter(
    prefix="/{channel_id}/discovery-datasets",
    tags=["discovery_datasets"],
    dependencies=[Depends(require_jwt_auth)],
)
discovery_datasets_router = APIRouter(
    prefix="/discovery-datasets",
    tags=["discovery_datasets"],
    dependencies=[Depends(require_jwt_auth)],
)

_PAYLOAD_ERROR_RESPONSE: dict[int | str, dict[str, Any]] = {
    status.HTTP_400_BAD_REQUEST: {"model": schemas.DiscoveryPayloadErrorResponse}
}
"""Declared so the per-cell error report reaches the OpenAPI schema.

The point of refusing a write with one entry per offending record is that a caller can act
on it; a generated client cannot unless the shape is advertised.
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ channel-scoped ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@channel_discovery_datasets_router.get("")
async def get_channel_discovery_datasets(
    channel_id: int,
    validation_status: schemas.DiscoveryValidationStatus | None = None,
    indexing_status: schemas.DiscoveryIndexingStatus | None = None,
    agency: str | None = None,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ListResponse[schemas.DiscoveryDataset]:
    """Returns a list of discovery datasets for the channel.

    `agency` matches the way the natural key does: ignoring case and surrounding whitespace.
    """
    service = DiscoveryDatasetService(session)

    records = await service.get_record_schemas_by_channel(
        channel_id, limit, offset, validation_status, indexing_status, agency
    )
    total = await service.get_records_count(channel_id, validation_status, indexing_status, agency)

    return schemas.ListResponse[schemas.DiscoveryDataset](
        data=records,
        limit=limit,
        offset=offset,
        count=len(records),
        total=total,
    )


@channel_discovery_datasets_router.post("")
async def add_discovery_dataset_to_channel(
    channel_id: int,
    data: schemas.DiscoveryDatasetBase,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.DiscoveryDataset:
    """Add a single discovery dataset to the channel.

    Returns 422 when the agency or dataset id is empty - that is a schema violation, so it
    arrives in pydantic's shape - and 409 when the channel already holds a record with the
    same key.
    """
    return await DiscoveryDatasetService(session).add_record(channel_id, data)


@channel_discovery_datasets_router.post("/bulk", responses=_PAYLOAD_ERROR_RESPONSE)
async def add_discovery_datasets_bulk(
    channel_id: int,
    data: list[schemas.DiscoveryDatasetBase],
    session: AsyncSession = Depends(models.get_session),
) -> list[schemas.DiscoveryDataset]:
    """Add multiple discovery datasets to the channel.

    All or nothing: nothing is saved unless every record can be. Two records describing the
    same dataset are refused with 400, naming the offending positions; an empty key is a
    schema violation and is refused with 422, carrying the item's position in `loc`.
    """
    return await DiscoveryDatasetService(session).add_records_bulk(channel_id, data)


@channel_discovery_datasets_router.delete("/bulk")
async def clear_channel_discovery_datasets(
    channel_id: int,
    session: AsyncSession = Depends(models.get_session),
) -> list[schemas.DiscoveryDataset]:
    """Delete all discovery datasets of the channel and return the deleted records."""
    return await DiscoveryDatasetService(session).delete_records_bulk(channel_id=channel_id)


@channel_discovery_datasets_router.post("/upload", responses=_PAYLOAD_ERROR_RESPONSE)
async def upload_discovery_datasets(
    channel_id: int,
    file: UploadFile,
    mode: schemas.DiscoveryUploadMode = schemas.DiscoveryUploadMode.UPSERT,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.DiscoveryUploadSummary:
    """Load a filled discovery workbook (.xlsx) or CSV into the channel.

    Records are matched on (agency, dataset id), ignoring case and spacing. `upsert` keeps
    records that the file does not mention; `replace` deletes them. A record whose fields
    are unchanged is not rewritten, so its validation and indexing state is preserved.

    Synchronous, because the value of this endpoint is the per-cell error report: a
    structural problem returns 400 naming every offending cell and saves nothing. Uploading
    does not validate content - that happens on the next indexing job.
    """
    service = DiscoveryDatasetService(session)
    data = await service.read_upload(file)
    return await service.upload(channel_id, data, file.filename, mode)


@channel_discovery_datasets_router.post("/indexing-jobs", status_code=status.HTTP_202_ACCEPTED)
async def trigger_discovery_indexing(
    background_tasks: BackgroundTasks,
    channel_id: int,
) -> schemas.DiscoveryIndexingJob:
    """Re-validate and re-publish every discovery dataset of the channel.

    Creates a job, schedules the work in the background, and returns the job immediately
    with 202 Accepted. Poll it via `GET /discovery-datasets/indexing-jobs/{job_id}`.
    Returns 409 while a job for the channel is already queued or running, and 409 when the
    channel has no discovery RAG application configured.

    A run evaluates the check set over every record, then reconciles the channel's Generic
    RAG documents with the verdicts: a valid record is published (or republished, if it was
    edited since), an invalid one has its document withdrawn, and a document no record
    claims any more is removed. A record already indexed and unchanged is left alone.
    """
    async with get_session_context_manager() as session:
        return await IndexingJobService(session).trigger(
            background_tasks=background_tasks, channel_id=channel_id
        )


@channel_discovery_datasets_router.get("/indexing-jobs")
async def get_discovery_indexing_jobs(
    channel_id: int,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.ListResponse[schemas.DiscoveryIndexingJob]:
    """Get a paginated list of indexing jobs for the channel, newest first."""
    service = IndexingJobService(session)
    jobs = await service.get_jobs(channel_id=channel_id, limit=limit, offset=offset)
    total = await service.get_jobs_count(channel_id)
    return schemas.ListResponse[schemas.DiscoveryIndexingJob](
        data=jobs,
        limit=limit,
        offset=offset,
        count=len(jobs),
        total=total,
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ single record ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `/bulk` and `/indexing-jobs` are declared before `/{item_id}` so the int converter does
# not swallow them.


@discovery_datasets_router.post("/bulk")
async def update_discovery_datasets_bulk(
    data: list[schemas.DiscoveryDatasetUpdateBulk],
    session: AsyncSession = Depends(models.get_session),
) -> list[schemas.DiscoveryDataset]:
    """Update multiple discovery datasets by id."""
    return await DiscoveryDatasetService(session).update_records_bulk(data)


@discovery_datasets_router.delete("/bulk")
async def delete_discovery_datasets_bulk(
    item_ids: list[int],
    session: AsyncSession = Depends(models.get_session),
) -> list[schemas.DiscoveryDataset]:
    """Delete multiple discovery datasets by their ids and return the deleted records."""
    return await DiscoveryDatasetService(session).delete_records_bulk(item_ids=item_ids)


@discovery_datasets_router.get("/indexing-jobs/{job_id}")
async def get_discovery_indexing_job_by_id(
    job_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.DiscoveryIndexingJob:
    """Get an indexing job by id, for polling its status."""
    return await IndexingJobService(session).get_job_by_id(job_id)


@discovery_datasets_router.get("/{item_id}")
async def get_discovery_dataset_by_id(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    _=Depends(cancel_on_disconnect),
) -> schemas.DiscoveryDataset:
    """Returns a discovery dataset by id."""
    return await DiscoveryDatasetService(session).get_record_schema_by_id(item_id)


@discovery_datasets_router.post("/{item_id}")
async def update_discovery_dataset(
    item_id: int,
    data: schemas.DiscoveryDatasetUpdate,
    session: AsyncSession = Depends(models.get_session),
) -> schemas.DiscoveryDataset:
    """Update the received fields of a discovery dataset.

    Editing any descriptive field resets the record's validation verdict, and marks it
    outdated if it had been published: the indexed document no longer matches the record.
    """
    return await DiscoveryDatasetService(session).update(item_id, data)


@discovery_datasets_router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_discovery_dataset(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
) -> None:
    """Delete a discovery dataset by id."""
    await DiscoveryDatasetService(session).delete(item_id)
