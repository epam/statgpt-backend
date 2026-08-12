import logging

from fastapi import BackgroundTasks, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.common import models, schemas
from statgpt.common.schemas import (
    DiscoveryIndexingStatus,
    DiscoveryValidationStatus,
    PreprocessingStatusEnum,
)
from statgpt.common.services import ChannelService, DiscoveryDatasetService
from statgpt.common.services.base import DbServiceBase
from statgpt.common.utils import format_exception_reason, get_ts_utcnow

from .background_tasks import background_task
from .discovery_validation import DiscoveryValidator
from .exceptions import IndexingJobInProgressError

_log = logging.getLogger(__name__)

PUBLISH_NOT_IMPLEMENTED = "Publishing to the discovery RAG is not implemented yet."
"""Why a record that passed validation still ends a run unindexed.

The publish stage lands with the Generic RAG ingestion client. Until then a run reports
that nothing was published rather than claiming a record is indexed when it is not.
"""

_ACTIVE_STATUSES = (PreprocessingStatusEnum.QUEUED, PreprocessingStatusEnum.IN_PROGRESS)


class AdminPortalDiscoveryIndexingJobService(DbServiceBase):
    """Creates, reports on, and runs indexing jobs over a channel's discovery datasets.

    Owns the run's stages - validate, then publish - so adding a real publisher later
    changes neither the API nor the schemas.
    """

    def __init__(self, session: AsyncSession | None = None) -> None:
        super().__init__(session, None)  # No need for session lock in Admin Portal

    @staticmethod
    def _serialize(job: models.DiscoveryIndexingJob) -> schemas.DiscoveryIndexingJob:
        return schemas.DiscoveryIndexingJob.model_validate(job, from_attributes=True)

    async def _get_active_job(self, channel_id: int) -> models.DiscoveryIndexingJob | None:
        query = (
            select(models.DiscoveryIndexingJob)
            .where(
                models.DiscoveryIndexingJob.channel_id == channel_id,
                models.DiscoveryIndexingJob.status.in_(_ACTIVE_STATUSES),
            )
            .order_by(models.DiscoveryIndexingJob.id.desc())
            .limit(1)
        )
        async with self._lock_session() as session:
            return (await session.execute(query)).scalar_one_or_none()

    async def trigger(
        self, background_tasks: BackgroundTasks, channel_id: int
    ) -> schemas.DiscoveryIndexingJob:
        """Create a job, schedule the run in the background, and return the job."""
        channel = await ChannelService(self._session).get_model_by_id(channel_id)

        if active := await self._get_active_job(channel.id):
            raise IndexingJobInProgressError(channel_id=channel.id, job_id=active.id)

        job = models.DiscoveryIndexingJob(
            channel_id=channel.id, status=PreprocessingStatusEnum.QUEUED
        )
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)

        background_tasks.add_task(run_discovery_indexing_in_background_task, job_id=job.id)

        return self._serialize(job)

    async def get_job_by_id(self, job_id: int) -> schemas.DiscoveryIndexingJob:
        """A job id is globally addressable, like the deduplication and auto-update jobs."""
        return self._serialize(await self._get_job_model_or_raise(job_id))

    async def _get_job_model_or_raise(self, job_id: int) -> models.DiscoveryIndexingJob:
        async with self._lock_session() as session:
            job: models.DiscoveryIndexingJob | None = await session.get(
                models.DiscoveryIndexingJob, job_id
            )
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Discovery indexing job with id={job_id} not found",
            )
        return job

    async def get_jobs(
        self, channel_id: int, limit: int, offset: int
    ) -> list[schemas.DiscoveryIndexingJob]:
        query = (
            select(models.DiscoveryIndexingJob)
            .where(models.DiscoveryIndexingJob.channel_id == channel_id)
            .order_by(models.DiscoveryIndexingJob.id.desc())
            .limit(limit)
            .offset(offset)
        )
        async with self._lock_session() as session:
            result = await session.execute(query)
        return [self._serialize(job) for job in result.scalars().all()]

    async def get_jobs_count(self, channel_id: int) -> int:
        query = (
            select(func.count("*"))
            .select_from(models.DiscoveryIndexingJob)
            .where(models.DiscoveryIndexingJob.channel_id == channel_id)
        )
        async with self._lock_session() as session:
            return (await session.execute(query)).scalar_one()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the run ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def process_job(self, job_id: int) -> None:
        """Run a job to completion: validate every record, then publish the valid ones.

        A per-record failure never aborts the run; it is recorded on the record. Only a
        failure of the run itself marks the job FAILED.
        """
        async with self._scoped_session() as session:
            job = await self._get_job_model_or_raise(job_id)
            job.status = PreprocessingStatusEnum.IN_PROGRESS
            await session.commit()

            try:
                records = await DiscoveryDatasetService(session).get_record_models_by_channel(
                    job.channel_id, limit=None, offset=0
                )
                valid, invalid = self._validate_records(records)

                job.records_total = len(records)
                job.records_valid = valid
                job.records_invalid = invalid
                # The publish stage is not implemented, so nothing is claimed to be indexed.
                job.documents_upserted = 0
                job.documents_deleted = 0
                job.details = (
                    f"Validated {len(records)} record(s): {valid} valid, {invalid} invalid."
                    f" {PUBLISH_NOT_IMPLEMENTED}"
                )
                job.status = PreprocessingStatusEnum.COMPLETED
                await session.commit()
            except Exception as e:
                _log.exception(f"Discovery indexing job {job_id} failed")
                await session.rollback()
                job = await self._get_job_model_or_raise(job_id)
                job.reason_for_failure = format_exception_reason(e)
                job.status = PreprocessingStatusEnum.FAILED
                await session.commit()
                return

            _log.info(f"Discovery indexing job {job_id} completed: {job.details}")

    @staticmethod
    def _validate_records(records: list[models.DiscoveryDataset]) -> tuple[int, int]:
        """Evaluate the check set over every record, in place. Returns (valid, invalid).

        Every record is re-evaluated regardless of its current status, so a changed check
        set takes effect on the next run instead of leaving stored verdicts stale.
        """
        validator = DiscoveryValidator()
        evaluated_at = get_ts_utcnow()
        valid = invalid = 0

        for record in records:
            issues = validator.validate(record)
            record.validated_at = evaluated_at
            if issues:
                record.validation_status = DiscoveryValidationStatus.INVALID
                record.validation_issues = [issue.model_dump(mode="json") for issue in issues]
                # An invalid record is not published at all, so its indexing status is left
                # as it is rather than being reported as an indexing failure.
                invalid += 1
            else:
                record.validation_status = DiscoveryValidationStatus.VALID
                record.validation_issues = None
                record.indexing_status = DiscoveryIndexingStatus.FAILED
                record.index_error = PUBLISH_NOT_IMPLEMENTED
                valid += 1

        return valid, invalid


@background_task
async def run_discovery_indexing_in_background_task(job_id: int) -> None:
    try:
        await AdminPortalDiscoveryIndexingJobService().process_job(job_id=job_id)
    except Exception:
        _log.exception(f"Failed to run discovery indexing (job_id={job_id})")
