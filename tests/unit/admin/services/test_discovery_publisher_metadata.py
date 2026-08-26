"""Tests for the country filter axis the publisher adds to a document's metadata."""

from dataclasses import dataclass

from statgpt.admin.services.discovery_publisher import build_metadata
from statgpt.common import schemas
from statgpt.common.services.discovery_reference_area import SENTINEL

_CHANNEL = "statgpt-channel"


@dataclass(frozen=True)
class _Record:
    """The descriptive half of a record, satisfying the `DiscoveryRecord` protocol."""

    reference_area: str = "Indonesia (IDN)"
    regional_coverage: str = "None"
    excluded_regional_values: str = ""
    agency: str = "Bank Indonesia (BI)"
    dataset_id: str = "TABEL1_1"
    name: str = "Broad Money"
    description: str = "Money and banking table."
    url: str = "https://example.com/seki"
    time_coverage: str = "From 1989-01 to 2026-06"
    frequency_coverage: str = "Monthly"
    indicators_coverage: str = "broad money (Rp billions)"
    missing_indicators: str = "consumer prices"


def _metadata(reference_area: str) -> schemas.DiscoveryDocumentMetadata:
    record = _Record(reference_area=reference_area)
    return build_metadata(record, _CHANNEL, schemas.DiscoveryGrade.C)


def test_the_verbatim_cell_is_still_published():
    """Nothing a submitter wrote is lost or rewritten; the parse is additive."""
    metadata = _metadata("Indonesia (IDN); Malaysia (MYS)")

    assert metadata.reference_area == "Indonesia (IDN); Malaysia (MYS)"


def test_a_single_country_becomes_a_one_value_filter_axis():
    assert _metadata("Indonesia (IDN)").reference_area_values == ["Indonesia (IDN)"]


def test_a_multi_country_cell_becomes_one_value_per_country():
    """The whole point of the axis: a filter on any one member reaches this dataset."""
    metadata = _metadata("Indonesia (IDN); Malaysia (MYS); Thailand (THA)")

    assert metadata.reference_area_values == [
        "Indonesia (IDN)",
        "Malaysia (MYS)",
        "Thailand (THA)",
    ]


def test_a_record_with_no_country_scope_carries_the_sentinel():
    """Otherwise a euro-area dataset would be dropped from a question about Germany."""
    assert SENTINEL in _metadata("Euro area").reference_area_values


def test_an_empty_cell_carries_only_the_sentinel():
    assert _metadata("").reference_area_values == [SENTINEL]


def test_partner_countries_do_not_become_filter_values():
    metadata = _metadata("Japan (JPN); partner countries: China; United States")

    assert metadata.reference_area_values == ["Japan (JPN)"]


def test_the_channel_and_the_grade_still_scope_the_document():
    """One RAG application serves several channels and both grades."""
    metadata = _metadata("Indonesia (IDN)")

    assert metadata.statgpt_channel == _CHANNEL
    assert metadata.grade == schemas.DiscoveryGrade.C


def test_the_filter_axis_is_declared_filterable():
    """A run refuses to publish into an application whose schema does not declare it, so a
    misconfigured application fails loudly instead of producing an unfilterable index."""
    assert "reference_area_values" in schemas.DiscoveryDocumentMetadata.filterable_fields()


def test_the_axis_reaches_the_generated_channel_schema_as_an_array_of_strings():
    """The service filters an array field by containment only when it is typed as one."""
    schema = schemas.DiscoveryDocumentMetadata.channel_json_schema()

    axis = schema["properties"]["reference_area_values"]
    assert axis["type"] == "array"
    assert axis["items"]["type"] == "string"
    assert axis["enable_filtering"] is True
