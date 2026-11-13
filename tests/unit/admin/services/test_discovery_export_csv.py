"""Tests for the discovery dataset CSV export.

The CSV an admin downloads has to be the one the channel export writes, because it is fed
straight back into the upload endpoint of another channel - with or without the validation
columns of the report.
"""

import csv
import datetime
import io
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from statgpt.admin.services.discovery_dataset import AdminPortalDiscoveryDatasetService
from statgpt.admin.services.discovery_export import VALIDATION_HEADERS, export_rows
from statgpt.admin.services.discovery_upload import (
    RECORD_FIELDS,
    ParsedFile,
    field_for_header,
    parse_discovery_file,
)
from statgpt.admin.services.exceptions import DiscoveryDatasetsNotInChannelError
from statgpt.common import models, utils
from statgpt.common.schemas import (
    DiscoveryDatasetBase,
    DiscoveryExportFilters,
    DiscoveryIndexingStatus,
    DiscoveryValidationStatus,
    read_indicators,
)


def _stored(
    agency: str = "IMF", dataset_id: str = "TABEL1_1", **overrides: object
) -> models.DiscoveryDataset:
    """A stand-in for a stored row, carrying the deployment-local state too.

    The point of several of these tests is that the state does not reach the file.
    """
    values: dict[str, object] = {name: "" for name in RECORD_FIELDS}
    values["indicators_coverage"] = []
    values.update(agency=agency, dataset_id=dataset_id)
    values.update(overrides)
    values.setdefault("id", 1)
    values.setdefault("validation_status", DiscoveryValidationStatus.VALID)
    values.setdefault("validation_issues", None)
    values.setdefault("validated_at", None)
    values.setdefault("indexing_status", DiscoveryIndexingStatus.INDEXED)
    return cast(models.DiscoveryDataset, SimpleNamespace(**values))


def _invalid(**overrides: Any) -> models.DiscoveryDataset:
    return _stored(
        validation_status=DiscoveryValidationStatus.INVALID,
        validated_at=datetime.datetime(
            2026, 3, 4, 12, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=2))
        ),
        validation_issues=[
            {"field": "frequency_coverage", "message": "At least one frequency is required."},
            {"field": "indicator_units", "message": "The check could not be evaluated."},
        ],
        **overrides,
    )


def _service(records: list[models.DiscoveryDataset]) -> AdminPortalDiscoveryDatasetService:
    service = AdminPortalDiscoveryDatasetService(session=MagicMock())
    service.get_record_models_by_channel = AsyncMock(return_value=records)  # type: ignore[method-assign]
    service.get_record_models_by_channel_and_ids = AsyncMock(return_value=records)  # type: ignore[method-assign]
    return service


async def _export(
    service: AdminPortalDiscoveryDatasetService,
    filters: DiscoveryExportFilters | None = None,
    include_validation: bool = False,
) -> str:
    content = await service.export_discovery_datasets(
        channel_id=1,
        filters=filters or DiscoveryExportFilters(),
        include_validation=include_validation,
    )
    return content.decode("utf-8")


@pytest.mark.asyncio
async def test_columns_are_the_descriptive_fields_in_order() -> None:
    """The upload path reads this file by header, so the order is part of the contract."""
    content = await _export(_service([_stored()]))

    header = content.splitlines()[0]
    assert header == ",".join(DiscoveryDatasetBase.model_fields)


@pytest.mark.asyncio
async def test_deployment_local_state_is_not_exported() -> None:
    """Validation and indexing state describes the channel that holds the record, not the
    dataset, and is re-derived by the next indexing job."""
    content = await _export(_service([_invalid()]))

    assert "validation_status" not in content
    assert "indexing_status" not in content
    assert "INDEXED" not in content
    assert "frequency is required" not in content


@pytest.mark.asyncio
async def test_a_slice_that_matches_nothing_is_a_header_on_its_own() -> None:
    """A filtered export has to stay a valid CSV, so an empty result is not an error."""
    content = await _export(_service([]))

    assert content == ",".join(DiscoveryDatasetBase.model_fields) + "\r\n"


@pytest.mark.asyncio
async def test_the_filters_reach_the_query() -> None:
    service = _service([])

    await _export(
        service,
        DiscoveryExportFilters(
            validation_status=DiscoveryValidationStatus.INVALID,
            indexing_status=DiscoveryIndexingStatus.FAILED,
            agency="IMF",
        ),
    )

    kwargs = service.get_record_models_by_channel.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["validation_status"] is DiscoveryValidationStatus.INVALID
    assert kwargs["indexing_status"] is DiscoveryIndexingStatus.FAILED
    assert kwargs["agency"] == "IMF"
    assert kwargs["limit"] is None


@pytest.mark.asyncio
async def test_the_download_matches_what_the_channel_export_writes(tmp_path) -> None:
    """The two have to agree byte for byte: an export of one channel is an upload to another."""
    records = [_stored(dataset_id="TABEL1_1"), _stored(agency="OECD", dataset_id="TABEL2_2")]
    service = _service(records)

    downloaded = await service.export_discovery_datasets(
        channel_id=1, filters=DiscoveryExportFilters()
    )

    written = tmp_path / "discovery_datasets.csv"
    utils.write_csv_from_dict_list(export_rows(records), str(written))
    assert downloaded == written.read_bytes()


@pytest.mark.asyncio
async def test_an_export_longer_than_pythons_field_limit_uploads_unchanged(
    default_csv_field_limit: int,
) -> None:
    indicators = [{"name": f"indicator {i}", "units": "units"} for i in range(10_000)]
    service = _service([_stored(indicators_coverage=indicators)])

    content = await _export(service)
    parsed = parse_discovery_file(
        content.encode("utf-8"), "export.csv", max_rows=10, max_indicators_per_dataset=50_000
    )

    assert parsed.rows[0].indicators == read_indicators(indicators)
    assert len(content) > default_csv_field_limit


@pytest.mark.asyncio
async def test_indicators_round_trip_through_the_export_unchanged() -> None:
    """Names ending in parentheses, units holding them, and ';' inside a value all survive."""
    indicators = [
        {"name": "broad money (M2)", "units": ""},
        {"name": "CPI", "units": "Index (2015=100)"},
        {"name": "exports; goods", "units": "USD"},
    ]
    service = _service([_stored(indicators_coverage=indicators)])

    content = await _export(service)
    [row] = csv.DictReader(io.StringIO(content))
    parsed = parse_discovery_file(
        content.encode("utf-8"), "export.csv", max_rows=10, max_indicators_per_dataset=10
    )

    assert json.loads(row["indicators_coverage"]) == indicators
    assert parsed.rows[0].indicators == read_indicators(indicators)
    assert not parsed.problems


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ validation columns ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.asyncio
async def test_the_validation_columns_follow_the_submitted_fields() -> None:
    content = await _export(_service([_stored()]), include_validation=True)

    header = content.splitlines()[0]
    assert header == ",".join([*DiscoveryDatasetBase.model_fields, *VALIDATION_HEADERS])


@pytest.mark.asyncio
async def test_the_validation_columns_carry_the_stored_verdict() -> None:
    """One issue per line, named by the column label; a crashed check keeps its raw name."""
    content = await _export(_service([_invalid()]), include_validation=True)

    [row] = csv.DictReader(io.StringIO(content))
    assert row["validation_status"] == "INVALID"
    # Stored in another zone, reported in UTC.
    assert row["validated_at"] == "2026-03-04T10:30:00+00:00"
    assert row["validation_issues"] == (
        "Frequency coverage: At least one frequency is required.\n"
        "indicator_units: The check could not be evaluated."
    )


@pytest.mark.asyncio
async def test_a_record_not_yet_validated_says_so() -> None:
    """An empty issues cell would read as clean, so the issues column says so too."""
    record = _stored(validation_status=DiscoveryValidationStatus.NOT_VALIDATED)

    content = await _export(_service([record]), include_validation=True)

    [row] = csv.DictReader(io.StringIO(content))
    assert row["validation_status"] == "NOT_VALIDATED"
    assert row["validated_at"] == ""
    assert row["validation_issues"].startswith("Not validated yet")


@pytest.mark.asyncio
async def test_a_report_uploads_back_as_the_same_records() -> None:
    """The verdict columns are ignored on upload; the records come back unchanged."""
    indicators = [{"name": "CPI", "units": ""}]
    record = _invalid(description="Prices", indicators_coverage=indicators)

    plain = await _export(_service([record]))
    report = await _export(_service([record]), include_validation=True)

    def parse(content: str) -> ParsedFile:
        return parse_discovery_file(
            content.encode("utf-8"), "export.csv", max_rows=10, max_indicators_per_dataset=10
        )

    parsed_plain, parsed_report = parse(plain), parse(report)
    assert not parsed_report.problems
    assert parsed_report.rows == parsed_plain.rows
    assert parsed_report.rows[0].values["description"] == "Prices"
    assert parsed_report.rows[0].indicators == read_indicators(indicators)


def test_no_validation_header_is_read_as_a_record_field() -> None:
    """Otherwise re-uploading a report would read a verdict back in as data."""
    for name, title in VALIDATION_HEADERS.items():
        assert field_for_header(name) is None
        assert field_for_header(title) is None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ selection by id ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.asyncio
async def test_ids_select_the_records() -> None:
    records = [_stored(id=3), _stored(id=5, dataset_id="OTHER")]
    service = _service(records)

    content = await _export(service, DiscoveryExportFilters(ids=[5, 3, 5]))

    service.get_record_models_by_channel.assert_not_awaited()  # type: ignore[attr-defined]
    args = service.get_record_models_by_channel_and_ids.await_args.args  # type: ignore[attr-defined]
    assert args == (1, {3, 5})
    assert len(list(csv.DictReader(io.StringIO(content)))) == 2


@pytest.mark.asyncio
async def test_an_id_the_channel_does_not_hold_is_refused_by_name() -> None:
    """Dropping it would produce a report that looks complete and is not."""
    service = _service([_stored(id=3)])

    with pytest.raises(DiscoveryDatasetsNotInChannelError) as error:
        await _export(service, DiscoveryExportFilters(ids=[3, 9, 4]))

    assert error.value.item_ids == [4, 9]
    assert "4, 9" in str(error.value)


@pytest.mark.parametrize(
    "other",
    [
        {"agency": "IMF"},
        {"validation_status": DiscoveryValidationStatus.INVALID},
        {"indexing_status": DiscoveryIndexingStatus.FAILED},
    ],
)
def test_ids_cannot_be_combined_with_the_filters(other: dict[str, object]) -> None:
    """The ids would win, so every caller - not only the router - has to be refused."""
    with pytest.raises(ValidationError, match="cannot be combined"):
        DiscoveryExportFilters(ids=[3], **other)  # type: ignore[arg-type]
