"""Tests for the discovery indexing job's own orchestration.

What a record becomes, and what happens to its document, belongs to `DiscoveryPublisher` and
is tested in `test_discovery_publisher.py`. What is tested here is the sequencing the job
service owns: refusing a channel with nowhere to publish to, committing the validation
verdicts before the network stage, carrying `force` through to the publisher, and reporting
the run on the job row.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from statgpt.admin.services import discovery_indexing_job as job_module
from statgpt.admin.services.discovery_indexing_job import (
    AdminPortalDiscoveryIndexingJobService,
    run_discovery_indexing_in_background_task,
)
from statgpt.admin.services.discovery_publisher import PublishCounts
from statgpt.admin.services.exceptions import (
    DiscoveryRagNotConfiguredError,
    IndexingJobInProgressError,
)
from statgpt.common import models, schemas
from statgpt.common.utils import get_ts_utcnow

_APPLICATION = "statgpt-generic-rag-grade-b-and-c"
_AREA_APPLICATION = "statgpt-generic-rag-reference-areas"

_BASE_DETAILS = schemas.ChannelConfig(
    supreme_agent=schemas.SupremeAgentConfig(
        name="T", domain="D", terminology_domain="T", language_instructions=["i"]
    )
).model_dump(mode="json", by_alias=True, exclude_none=True)


def _channel(*, discovery_datasets: bool = True, reference_areas: bool = False) -> models.Channel:
    details = dict(_BASE_DETAILS)
    if discovery_datasets:
        tool_details: dict[str, Any] = {
            "applicationId": _APPLICATION,
            "templates": {"wrapper": "{items}", "item": "- {name}"},
        }
        if reference_areas:
            tool_details["referenceAreaApplicationId"] = _AREA_APPLICATION
        details["discoveryDatasets"] = {
            "type": "DISCOVERY_DATASETS",
            "name": "discovery_datasets",
            "description": "Discovery datasets.",
            "details": tool_details,
        }
    return models.Channel(
        id=7,
        title="Channel",
        description="",
        deployment_id="statgpt-gtdc",
        llm_model="gpt-4o",
        details=details,
        created_at=get_ts_utcnow(),
        updated_at=get_ts_utcnow(),
    )


def _job() -> models.DiscoveryIndexingJob:
    return models.DiscoveryIndexingJob(
        id=42, channel_id=7, status=schemas.PreprocessingStatusEnum.QUEUED
    )


def _record(**overrides: Any) -> models.DiscoveryDataset:
    """A stored row, carrying only what the validator and the publisher read."""
    values: dict[str, Any] = {
        "id": 1,
        "channel_id": 7,
        "reference_area": "Indonesia (IDN)",
        "regional_coverage": "",
        "excluded_regional_values": "",
        "agency": "Bank Indonesia (BI)",
        "dataset_id": "TABEL1_1",
        "name": "Broad Money",
        "description": "Money and Banking table.",
        "url": "https://www.bi.go.id/TABEL1_1.xls",
        "time_coverage": "From 1989-01 to 2026-06",
        "frequency_coverage": "Monthly",
        "indicators_coverage": "broad money (M2)",
        "missing_indicators": "",
        "validation_status": schemas.DiscoveryValidationStatus.NOT_VALIDATED,
        "validation_issues": None,
        "validated_at": None,
        "indexing_status": schemas.DiscoveryIndexingStatus.NEW,
        "indexed_at": None,
        "index_error": None,
    }
    values.update(overrides)
    return cast(models.DiscoveryDataset, SimpleNamespace(**values))


class _Spy:
    """The collaborators `process_job` reaches for, replaced wholesale.

    `commits` records the state of the job row and of the records at every commit, which is
    the only way to assert that the verdicts were durable *before* the network stage ran.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        records: list[models.DiscoveryDataset] | None = None,
        counts: PublishCounts | None = None,
        publish_error: Exception | None = None,
        area_counts: PublishCounts | None = None,
        area_error: Exception | None = None,
        verify_errors: dict[str, Exception] | None = None,
        channel: models.Channel | None = None,
    ) -> None:
        self.job = _job()
        self.records = records if records is not None else [_record()]
        self.counts = counts or PublishCounts(upserted=1)
        self.area_counts = area_counts or PublishCounts(upserted=2, deleted=1, skipped=3)
        self.commits: list[dict[str, Any]] = []
        self.published: list[dict[str, Any]] = []
        self.stages: list[str] = []
        self.applications: list[str] = []
        self.closed = False

        channel = channel if channel is not None else _channel()
        monkeypatch.setattr(
            job_module, "ChannelService", lambda _: SimpleNamespace(get_model_by_id=self._channel)
        )
        self._channel_model = channel
        monkeypatch.setattr(
            job_module,
            "DiscoveryDatasetService",
            lambda _: SimpleNamespace(get_record_models_by_channel=self._get_records),
        )
        monkeypatch.setattr(job_module, "DiscoveryPublisher", self._publisher)
        monkeypatch.setattr(job_module, "ReferenceAreaPublisher", self._area_publisher)
        monkeypatch.setattr(
            job_module.GenericRagIngestionClient, "for_application", self._for_application
        )
        self._publish_error = publish_error
        self._area_error = area_error
        self._verify_errors = verify_errors or {}

        self.session = AsyncMock()
        self.session.commit.side_effect = self._on_commit
        self.session.get.return_value = self.job

        self.service = AdminPortalDiscoveryIndexingJobService()
        self.service._DbServiceBase__session = self.session  # type: ignore[attr-defined]

    async def _channel(self, channel_id: int) -> models.Channel:
        return self._channel_model

    async def _get_records(self, *_: Any, **__: Any) -> list[models.DiscoveryDataset]:
        return self.records

    def _for_application(self, application_id: str) -> Any:
        self.application_id = application_id
        self.applications.append(application_id)
        spy = self

        class _Client:
            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *_: object) -> None:
                spy.closed = True

        return _Client()

    def _publisher(self, client: Any, *, channel: str, force: bool) -> Any:
        self.publisher_channel = channel
        self.publisher_force = force
        return SimpleNamespace(
            verify_metadata_schema=self._verifier("documents"), publish=self._publish
        )

    def _verifier(self, target: str) -> Any:
        async def verify() -> None:
            self.stages.append(f"verify:{target}")
            if error := self._verify_errors.get(target):
                raise error

        return verify

    async def _publish(self, records: list[models.DiscoveryDataset]) -> PublishCounts:
        if self._publish_error is not None:
            raise self._publish_error
        self.published.append({"count": len(records)})
        self.stages.append("documents")
        for record in records:
            record.indexing_status = schemas.DiscoveryIndexingStatus.INDEXED
        return self.counts

    def _area_publisher(self, client: Any, *, channel: str, force: bool) -> Any:
        self.area_publisher_channel = channel
        self.area_publisher_force = force
        return SimpleNamespace(
            verify_metadata_schema=self._verifier("reference-areas"), publish=self._publish_areas
        )

    async def _publish_areas(self, records: list[models.DiscoveryDataset]) -> PublishCounts:
        if self._area_error is not None:
            raise self._area_error
        self.stages.append("reference-areas")
        return self.area_counts

    async def _on_commit(self) -> None:
        self.commits.append(
            {
                "job_status": self.job.status,
                "job_details": self.job.details,
                "published_calls": len(self.published),
                "validation": [record.validation_status for record in self.records],
                "indexing": [record.indexing_status for record in self.records],
            }
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triggering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _trigger_service(
    monkeypatch: pytest.MonkeyPatch,
    channel: models.Channel,
    active_job: Any = None,
) -> tuple[AdminPortalDiscoveryIndexingJobService, AsyncMock]:
    """A service whose channel lookup and active-job check are answered without a database."""
    monkeypatch.setattr(
        job_module,
        "ChannelService",
        lambda _: SimpleNamespace(get_model_by_id=AsyncMock(return_value=channel)),
    )

    async def refresh(job: models.DiscoveryIndexingJob, *_: Any, **__: Any) -> None:
        job.id = 42
        job.created_at = job.updated_at = get_ts_utcnow()

    service = AdminPortalDiscoveryIndexingJobService()
    session = AsyncMock()
    session.add = MagicMock()
    session.refresh.side_effect = refresh
    service._DbServiceBase__session = session  # type: ignore[attr-defined]
    service._get_active_job = AsyncMock(return_value=active_job)  # type: ignore[method-assign]
    return service, session


async def test_a_channel_without_a_publish_target_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused at the request, so an administrator is not told by a job that failed later."""
    service, session = _trigger_service(monkeypatch, _channel(discovery_datasets=False))
    background_tasks = MagicMock()

    with pytest.raises(
        DiscoveryRagNotConfiguredError, match="discoveryDatasets.details.applicationId"
    ):
        await service.trigger(background_tasks=background_tasks, channel_id=7)

    session.add.assert_not_called()
    background_tasks.add_task.assert_not_called()


async def test_a_missing_publish_target_is_reported_before_a_running_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixable misconfiguration outranks the transient conflict."""
    service, _ = _trigger_service(
        monkeypatch, _channel(discovery_datasets=False), active_job=SimpleNamespace(id=9)
    )

    with pytest.raises(DiscoveryRagNotConfiguredError):
        await service.trigger(background_tasks=MagicMock(), channel_id=7)


async def test_an_already_running_job_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    service, session = _trigger_service(monkeypatch, _channel(), active_job=SimpleNamespace(id=9))

    with pytest.raises(IndexingJobInProgressError, match="9"):
        await service.trigger(background_tasks=MagicMock(), channel_id=7)

    session.add.assert_not_called()


@pytest.mark.parametrize("force", [False, True])
async def test_force_is_scheduled_with_the_background_task(
    monkeypatch: pytest.MonkeyPatch, force: bool
) -> None:
    """`force` rides on the task, since nothing re-reads a job row to resume it."""
    service, _ = _trigger_service(monkeypatch, _channel())
    background_tasks = MagicMock()

    await service.trigger(background_tasks=background_tasks, channel_id=7, force=force)

    (scheduled,), kwargs = background_tasks.add_task.call_args
    assert scheduled is run_discovery_indexing_in_background_task
    assert kwargs["force"] is force


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the run ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_run_validates_then_publishes_and_reports_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _Spy(monkeypatch, counts=PublishCounts(upserted=3, deleted=2, skipped=1, failed=1))

    await spy.service.process_job(job_id=42)

    assert spy.job.status is schemas.PreprocessingStatusEnum.COMPLETED
    assert spy.job.records_total == 1
    assert (spy.job.documents_upserted, spy.job.documents_deleted) == (3, 2)
    assert spy.job.details is not None
    assert "1 valid, 0 invalid" in spy.job.details
    assert "Published 3, removed 2 document(s), skipped 1 already indexed, failed 1" in (
        spy.job.details
    )
    assert not spy.job.details.startswith("Forced rebuild.")


async def test_the_verdicts_are_committed_before_the_network_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """They are established, and a publish stage that dies must not take them down."""
    spy = _Spy(monkeypatch)

    await spy.service.process_job(job_id=42)

    # First commit: IN_PROGRESS. Second: the verdicts, with nothing published yet.
    verdicts = spy.commits[1]
    assert verdicts["published_calls"] == 0
    assert verdicts["validation"] == [schemas.DiscoveryValidationStatus.VALID]
    assert spy.commits[-1]["published_calls"] == 1


async def test_a_failing_publish_stage_leaves_the_verdicts_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _Spy(monkeypatch, publish_error=RuntimeError("channel unreachable"))

    await spy.service.process_job(job_id=42)

    assert spy.job.status is schemas.PreprocessingStatusEnum.FAILED
    assert spy.job.reason_for_failure is not None
    assert "channel unreachable" in spy.job.reason_for_failure
    # The verdict commit happened, and it happened before the failure.
    assert spy.commits[1]["validation"] == [schemas.DiscoveryValidationStatus.VALID]
    spy.session.rollback.assert_awaited()


async def test_an_invalid_record_is_reported_as_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _Spy(monkeypatch, records=[_record(url="not-a-url", frequency_coverage="Hourly")])

    await spy.service.process_job(job_id=42)

    assert spy.job.records_valid == 0
    assert spy.job.records_invalid == 1
    assert spy.records[0].validation_status is schemas.DiscoveryValidationStatus.INVALID
    assert spy.records[0].validation_issues


async def test_a_run_publishes_into_the_configured_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The channel is identified to the publisher by deployment id, not by row id."""
    spy = _Spy(monkeypatch)

    await spy.service.process_job(job_id=42)

    assert spy.application_id == _APPLICATION
    assert spy.publisher_channel == "statgpt-gtdc"
    assert spy.closed, "the client's connection pool must not outlive the run"


@pytest.mark.parametrize("force", [False, True])
async def test_force_reaches_the_publisher(monkeypatch: pytest.MonkeyPatch, force: bool) -> None:
    spy = _Spy(monkeypatch)

    await spy.service.process_job(job_id=42, force=force)

    assert spy.publisher_force is force
    assert spy.job.details is not None
    assert spy.job.details.startswith("Forced rebuild.") is force


async def test_a_run_on_a_channel_that_lost_its_target_fails_the_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configuration can be removed between the trigger and the run."""
    spy = _Spy(monkeypatch, channel=_channel(discovery_datasets=False))

    await spy.service.process_job(job_id=42)

    assert spy.job.status is schemas.PreprocessingStatusEnum.FAILED
    assert spy.job.reason_for_failure is not None
    assert "discoveryDatasets" in spy.job.reason_for_failure


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the reference-area vocabulary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_the_vocabulary_is_published_after_the_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It describes what the channel holds, so publishing it first would describe nothing."""
    spy = _Spy(monkeypatch, channel=_channel(reference_areas=True))

    await spy.service.process_job(job_id=42, force=True)

    assert spy.stages == [
        "verify:documents",
        "verify:reference-areas",
        "documents",
        "reference-areas",
    ]
    assert spy.applications == [_APPLICATION, _AREA_APPLICATION]
    assert spy.area_publisher_channel == "statgpt-gtdc"
    assert spy.area_publisher_force is True
    assert spy.job.status is schemas.PreprocessingStatusEnum.COMPLETED
    assert spy.job.details is not None
    assert "Reference-area vocabulary: published 2, removed 1, unchanged 3." in spy.job.details


async def test_a_channel_without_a_vocabulary_skips_the_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The axis is simply unavailable at chat time; the records are published either way."""
    spy = _Spy(monkeypatch)

    await spy.service.process_job(job_id=42)

    assert spy.stages == ["verify:documents", "documents"]
    assert spy.applications == [_APPLICATION]
    assert spy.job.status is schemas.PreprocessingStatusEnum.COMPLETED
    assert spy.job.details is not None
    assert "Reference-area vocabulary" not in spy.job.details


async def test_a_failed_vocabulary_fails_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vocabulary that does not match the records narrows queries away from real answers."""
    spy = _Spy(
        monkeypatch,
        channel=_channel(reference_areas=True),
        area_error=RuntimeError("vocabulary channel unreachable"),
    )

    await spy.service.process_job(job_id=42)

    assert spy.job.status is schemas.PreprocessingStatusEnum.FAILED
    assert spy.job.reason_for_failure is not None
    assert "vocabulary channel unreachable" in spy.job.reason_for_failure
    # The records were published all the same, and their statuses were committed.
    assert spy.stages == ["verify:documents", "verify:reference-areas", "documents"]
    assert spy.records[0].indexing_status is schemas.DiscoveryIndexingStatus.INDEXED


async def test_a_misconfigured_vocabulary_channel_stops_the_run_before_it_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema is one cheap read, and the job fails on it either way.

    Verified after the documents, it would fail having already done every bit of its work -
    once per run, until someone fixed the configuration.
    """
    spy = _Spy(
        monkeypatch,
        channel=_channel(reference_areas=True),
        verify_errors={"reference-areas": RuntimeError("metadata_schema does not declare roles")},
    )

    await spy.service.process_job(job_id=42)

    assert spy.job.status is schemas.PreprocessingStatusEnum.FAILED
    assert spy.stages == ["verify:documents", "verify:reference-areas"]
    assert spy.published == []
