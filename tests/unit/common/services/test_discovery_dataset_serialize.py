"""Tests for how a stored discovery record is serialized for the API."""

import datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from statgpt.common import models, schemas
from statgpt.common.services.discovery_dataset import DiscoveryDatasetService

_USAGE = schemas.DiscoveryLLMCallUsage(
    deployment="gpt-4o",
    model="gpt-4o",
    prompt_tokens=100,
    completion_tokens=20,
    cost=0.001,
    duration_seconds=1.5,
)
_RUN = schemas.DiscoverySummaryRun(
    calls=[
        schemas.DiscoverySummaryCall(
            step=schemas.DiscoverySummaryStep.CHUNK,
            part=1,
            answer={"groups": ["Money"]},
            usage=_USAGE,
        )
    ]
)
_CALLS = _RUN.model_dump(mode="json")
_NO_CALLS = schemas.DiscoverySummaryRun(calls=[]).model_dump(mode="json")
_CALL_FIELDS = {"indicatorsSummaryCalls", "indicatorsSummaryFailedCalls"}


def _item() -> models.DiscoveryDataset:
    now = datetime.datetime(2026, 10, 29, tzinfo=datetime.UTC)
    values: dict[str, Any] = {
        "id": 1,
        "created_at": now,
        "updated_at": now,
        "channel_id": 7,
        "reference_area": "Indonesia (IDN)",
        "regional_coverage": "",
        "agency": "Bank Indonesia (BI)",
        "dataset_id": "TABEL1_1",
        "name": "Broad Money",
        "description": "",
        "url": "",
        "time_coverage": "",
        "frequency_coverage": "Monthly",
        "indicators_coverage": [{"name": "M2", "units": "Rp billions"}],
        "indicators_summary": "Unit: Rp billions; M2",
        "indicators_summarized_at": now,
        "indicators_summary_calls": _CALLS,
        "indicators_summary_failed_calls": _NO_CALLS,
        "missing_indicators": "",
        "validation_status": schemas.DiscoveryValidationStatus.VALID,
        "validation_issues": None,
        "validated_at": None,
        "indexing_status": schemas.DiscoveryIndexingStatus.INDEXED,
        "indexed_at": None,
        "index_error": None,
    }
    return cast(models.DiscoveryDataset, SimpleNamespace(**values))


def test_the_summary_calls_are_left_out_of_a_record_unless_asked_for() -> None:
    dumped = DiscoveryDatasetService._serialize(_item()).model_dump(mode="json", by_alias=True)

    assert _CALL_FIELDS.isdisjoint(dumped)


@pytest.mark.parametrize("failed_calls", [_NO_CALLS, None], ids=["empty", "none"])
def test_the_summary_calls_asked_for_are_serialized_as_stored(
    failed_calls: dict[str, Any] | None,
) -> None:
    item = _item()
    item.indicators_summary_failed_calls = failed_calls

    record = DiscoveryDatasetService._serialize(item, include_summary_calls=True)
    dumped = record.model_dump(mode="json", by_alias=True)

    assert record.indicators_summary_calls == _RUN
    assert dumped["indicatorsSummaryCalls"]["usage"] == [
        {
            "deployment": "gpt-4o",
            "model": "gpt-4o",
            "calls": 1,
            "promptTokens": 100,
            "completionTokens": 20,
            "cost": 0.001,
            "durationSeconds": 1.5,
        }
    ]
    # A run with no calls is a failure with no answers; only `None` is left out.
    assert ("indicatorsSummaryFailedCalls" in dumped) is (failed_calls is not None)
