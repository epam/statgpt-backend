import logging

from fastapi import BackgroundTasks
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.common import models, schemas
from statgpt.common.services import ChannelSerializer, ChannelService, DiscoveryDatasetService
from statgpt.common.services.base import DbServiceBase
from statgpt.common.services.generic_rag import GenericRagIngestionClient
from statgpt.common.utils import format_exception_reason, get_ts_utcnow

from .background_tasks import background_task
from .discovery_publisher import DiscoveryPublisher, PublishCounts
from .discovery_validation import DiscoveryValidator
from .exceptions import (
    DiscoveryIndexingJobNotFoundError,
    DiscoveryRagNotConfiguredError,
    IndexingJobInProgressError,
    raise_for_conflict,
)
from .status_recovery import set_failed_status

_log = logging.getLogger(__name__)

_ACTIVE_STATUSES = (
    schemas.PreprocessingStatusEnum.QUEUED,
    schemas.PreprocessingStatusEnum.IN_PROGRESS,
)


class AdminPortalDiscoveryIndexingJobService(DbServiceBase):
    """Creates, reports on, and runs indexing jobs over a channel's discovery datasets.

    Owns the run's two stages: validate every record, then reconcile the channel's Generic
    RAG documents with them. Only the sequencing lives here - what a record becomes, and
    what happens to its document, belongs to `DiscoveryPublisher`.
    """

    def __init__(self, session: AsyncSession | None = None) -> None:
        super().__init__(session, None)  # No need for session lock in Admin Portal

    @staticmethod
    def _serialize(job: models.DiscoveryIndexingJob) -> schemas.DiscoveryIndexingJob:
        return schemas.DiscoveryIndexingJob.model_validate(job, from_attributes=True)

    @staticmethod
    def _rag_config(channel: models.Channel) -> schemas.DiscoveryRagConfig:
        """The channel's publish target, or a domain error naming what is missing."""
        config = ChannelSerializer.db_to_schema(channel).details.discovery_rag
        if config is None:
            raise DiscoveryRagNotConfiguredError(channel.id)
        return config

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
        self, background_tasks: BackgroundTasks, channel_id: int, force: bool = False
    ) -> schemas.DiscoveryIndexingJob:
        """Create a job, schedule the run in the background, and return the job.

        The up-front check names the job that is already running; the partial unique index
        `uq_discovery_indexing_jobs_active` is what actually enforces one active job per
        channel, so two simultaneous requests cannot both get past this.

        `force` travels with the scheduled task rather than on the job row: nothing re-reads
        a job to resume it, so there is nothing for a column to be read back by.
        """
        channel = await ChannelService(self._session).get_model_by_id(channel_id)

        # Fail here rather than inside the run: a channel with nowhere to publish to is a
        # configuration mistake the caller can fix, not a job worth recording.
        self._rag_config(channel)

        if active := await self._get_active_job(channel_id):
            raise IndexingJobInProgressError(channel_id=channel_id, job_id=active.id)

        job = models.DiscoveryIndexingJob(
            channel_id=channel.id, status=schemas.PreprocessingStatusEnum.QUEUED
        )
        self._session.add(job)
        try:
            await self._session.commit()
        except IntegrityError as e:
            await self._session.rollback()
            # Lost the race against a concurrent trigger. Re-read so the message can name the
            # job that won, and fall back to an unnamed one if it has since finished.
            active = await self._get_active_job(channel_id)
            raise_for_conflict(
                e,
                IndexingJobInProgressError(
                    channel_id=channel_id, job_id=active.id if active else None
                ),
            )
        await self._session.refresh(job)

        background_tasks.add_task(
            run_discovery_indexing_in_background_task, job_id=job.id, force=force
        )

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
            raise DiscoveryIndexingJobNotFoundError(job_id)
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

    async def set_failed_status_for_stuck_discovery_indexing_jobs(self) -> None:
        """Sets the status of all stuck DiscoveryIndexingJob records to FAILED.

        Reuses the recovery helper the other background-job types use, so discovery runs
        follow identical fix_statuses semantics (including the 12-hour staleness guard).

        This is the only thing that clears a job abandoned mid-run - a cancelled task never
        reaches the failure handler in `process_job` - and until it does, `trigger` keeps
        answering 409 for the channel.
        """
        await set_failed_status(
            self._session,
            models.DiscoveryIndexingJob,
            models.DiscoveryIndexingJob.status,
            "status",
        )
        await self._session.commit()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the run ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def process_job(self, job_id: int, force: bool = False) -> None:
        """Run a job to completion: validate every record, then publish the valid ones.

        A per-record failure never aborts the run; it is recorded on the record. Only a
        failure of the run itself marks the job FAILED.

        `force` republishes every valid record, including those already indexed and
        unchanged - see `DiscoveryPublisher`.
        """
        async with self._scoped_session() as session:
            job = await self._get_job_model_or_raise(job_id)
            job.status = schemas.PreprocessingStatusEnum.IN_PROGRESS
            await session.commit()

            try:
                channel = await ChannelService(session).get_model_by_id(job.channel_id)
                records = await DiscoveryDatasetService(session).get_record_models_by_channel(
                    job.channel_id, limit=None, offset=0
                )
                valid, invalid = self._validate_records(records)

                job.records_total = len(records)
                job.records_valid = valid
                job.records_invalid = invalid
                # Commit the verdicts before the network stage: they are established, and a
                # publish stage that dies should not take them down with it.
                await session.commit()

                counts = await self._publish_records(channel, records, force=force)

                job.documents_upserted = counts.upserted
                job.documents_deleted = counts.deleted
                job.details = (
                    f"{'Forced rebuild. ' if force else ''}"
                    f"Validated {len(records)} record(s): {valid} valid, {invalid} invalid."
                    f" Published {counts.upserted}, removed {counts.deleted} document(s),"
                    f" skipped {counts.skipped} already indexed, failed {counts.failed}."
                )
                job.status = schemas.PreprocessingStatusEnum.COMPLETED
                await session.commit()
            except Exception as e:
                _log.exception(f"Discovery indexing job {job_id} failed")
                await session.rollback()
                job = await self._get_job_model_or_raise(job_id)
                job.reason_for_failure = format_exception_reason(e)
                job.status = schemas.PreprocessingStatusEnum.FAILED
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
                record.validation_status = schemas.DiscoveryValidationStatus.INVALID
                record.validation_issues = [issue.model_dump(mode="json") for issue in issues]
                invalid += 1
            else:
                record.validation_status = schemas.DiscoveryValidationStatus.VALID
                record.validation_issues = None
                valid += 1

        return valid, invalid

    async def _publish_records(
        self, channel: models.Channel, records: list[models.DiscoveryDataset], force: bool
    ) -> PublishCounts:
        """Reconcile the channel's RAG documents with the records, in place.

        The indexing status of each record is written by the publisher; this only owns the
        client's lifetime, so the connection pool is closed when the run ends rather than
        held for as long as the process lives.
        """
        config = self._rag_config(channel)
        async with GenericRagIngestionClient.for_application(config.get_application_id()) as client:
            publisher = DiscoveryPublisher(client, channel=channel.deployment_id, force=force)
            await publisher.verify_metadata_schema()
            return await publisher.publish(records)


@background_task
async def run_discovery_indexing_in_background_task(job_id: int, force: bool = False) -> None:
    try:
        await AdminPortalDiscoveryIndexingJobService().process_job(job_id=job_id, force=force)
    except Exception:
        _log.exception(f"Failed to run discovery indexing (job_id={job_id})")
