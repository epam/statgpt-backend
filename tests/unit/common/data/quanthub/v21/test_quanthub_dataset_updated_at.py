"""Tests for resolving `QuanthubSdmx21DataSet.updated_at` with the default source priority."""

from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace

import pytest

from statgpt.common.data.base import DatasetCitation, PropertySource
from statgpt.common.data.base.sdmx_schemas import Sdmx30AnnotationModel
from statgpt.common.data.quanthub.config import QuanthubDataSetConfig
from statgpt.common.data.quanthub.v21.dataset import QuanthubSdmx21DataSet


def _default_updated_at() -> list[PropertySource]:
    return QuanthubDataSetConfig.model_fields["updated_at"].get_default(call_default_factory=True)


@dataclass
class _Config:
    """The only part of the dataset config `updated_at` reads."""

    updated_at: list[PropertySource] = field(default_factory=_default_updated_at)
    citation: DatasetCitation | None = None


def _dataset(
    annotations: dict[str, str] | None = None, citation_last_updated: str | None = None
) -> QuanthubSdmx21DataSet:
    """A dataset stub carrying only what `updated_at` reads."""
    dataset = QuanthubSdmx21DataSet.__new__(QuanthubSdmx21DataSet)
    dataset._config = _Config(  # type: ignore[assignment]
        citation=DatasetCitation(last_updated=citation_last_updated)
    )
    dataset._annotations = [
        Sdmx30AnnotationModel(id=annotation_id, value=value)
        for annotation_id, value in (annotations or {}).items()
    ]
    dataset._attribute_values = {}
    return dataset


async def _updated_at(dataset: QuanthubSdmx21DataSet) -> datetime | None:
    return await dataset.updated_at(SimpleNamespace())  # type: ignore[arg-type]


async def test_the_data_updated_at_annotation_is_used():
    dataset = _dataset({"dataUpdatedAt": "2025-07-15T10:20:30"})

    assert await _updated_at(dataset) == datetime(2025, 7, 15, 10, 20, 30)


async def test_the_data_updated_at_annotation_wins_over_the_other_sources():
    dataset = _dataset(
        {"lastUpdatedAt": "2025-09-01T00:00:00", "dataUpdatedAt": "2025-07-15T00:00:00"},
        citation_last_updated="2025-01-01",
    )

    assert await _updated_at(dataset) == datetime(2025, 7, 15)


@pytest.mark.parametrize(
    "annotations",
    [
        pytest.param({}, id="missing"),
        pytest.param({"dataUpdatedAt": "  "}, id="blank"),
        pytest.param({"dataUpdatedAt": "not a date"}, id="unparsable"),
    ],
)
async def test_the_last_updated_at_annotation_is_the_fallback(annotations: dict[str, str]):
    dataset = _dataset({**annotations, "lastUpdatedAt": "2025-09-01T00:00:00"})

    assert await _updated_at(dataset) == datetime(2025, 9, 1)


async def test_the_citation_is_the_last_fallback():
    dataset = _dataset(citation_last_updated="2025-01-01")

    assert await _updated_at(dataset) == datetime(2025, 1, 1)


async def test_no_source_resolves_to_none():
    assert await _updated_at(_dataset()) is None
