"""Tests for the discovery dataset workbook export.

The workbook is the upload template's shape, so the supplier can fix the cells and send it
back: every test that is not about the layout itself is about that round trip.
"""

import datetime
import io
import re
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import openpyxl
import pytest
from openpyxl.worksheet.worksheet import Worksheet

from statgpt.admin.services.discovery_dataset import AdminPortalDiscoveryDatasetService
from statgpt.admin.services.discovery_export import VALIDATION_HEADERS, build_workbook
from statgpt.admin.services.discovery_upload import (
    DATASETS_SHEET_FIELDS,
    FIELD_LABELS,
    INDICATOR_SHEET_HEADERS,
    RECORD_FIELDS,
    parse_discovery_file,
)
from statgpt.admin.services.exceptions import DiscoveryExportError
from statgpt.common import models
from statgpt.common.schemas import (
    DiscoveryExportFilters,
    DiscoveryExportFormat,
    DiscoveryIndexingStatus,
    DiscoveryValidationStatus,
    read_indicators,
)

_FLAGGED = "00FFC7CE"


def _stored(agency: str = "IMF", dataset_id: str = "TABEL1_1", **overrides: object):
    values: dict[str, object] = {name: "" for name in RECORD_FIELDS}
    values.update(
        reference_area="Indonesia",
        description="Money and banking",
        frequency_coverage="Monthly",
        indicators_coverage=[{"name": "Broad money (M2)", "units": "billions of Rp"}],
    )
    values.update(agency=agency, dataset_id=dataset_id)
    values.update(overrides)
    values.setdefault("id", 1)
    values.setdefault("validation_status", DiscoveryValidationStatus.VALID)
    values.setdefault("validation_issues", [])
    values.setdefault("validated_at", datetime.datetime(2026, 3, 4, 10, 30, tzinfo=datetime.UTC))
    values.setdefault("indexing_status", DiscoveryIndexingStatus.INDEXED)
    return cast(models.DiscoveryDataset, SimpleNamespace(**values))


def _load(data: bytes) -> openpyxl.Workbook:
    return openpyxl.load_workbook(io.BytesIO(data))


def _values(sheet: Worksheet) -> list[list[object]]:
    return [list(row) for row in sheet.iter_rows(values_only=True)]


def _parse(data: bytes):
    return parse_discovery_file(data, "report.xlsx", max_rows=100, max_indicators_per_dataset=100)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ layout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_the_workbook_has_the_templates_sheets() -> None:
    workbook = _load(build_workbook([_stored(), _stored(agency="BI", dataset_id="SEKI_2")]))

    assert workbook.sheetnames == ["Datasets", "TABEL1_1", "SEKI_2"]
    datasets = _values(workbook["Datasets"])
    assert datasets[0] == [FIELD_LABELS[name] for name in DATASETS_SHEET_FIELDS]
    assert datasets[1][:4] == ["Indonesia", None, "IMF", "TABEL1_1"]
    assert _values(workbook["TABEL1_1"]) == [
        list(INDICATOR_SHEET_HEADERS),
        ["Broad money (M2)", "billions of Rp"],
    ]


def test_the_header_row_is_bold() -> None:
    sheet = _load(build_workbook([_stored()]))["Datasets"]

    assert all(cell.font.bold for cell in sheet[1])


def test_a_dataset_without_indicators_still_gets_its_sheet() -> None:
    """The submitter needs somewhere to fill them in."""
    workbook = _load(build_workbook([_stored(indicators_coverage=[])]))

    assert _values(workbook["TABEL1_1"]) == [list(INDICATOR_SHEET_HEADERS)]


def test_the_validation_columns_follow_the_template_columns() -> None:
    record = _stored(
        validation_status=DiscoveryValidationStatus.INVALID,
        validation_issues=[{"field": "url", "message": "A URL is required."}],
    )

    sheet = _load(build_workbook([record], include_validation=True))["Datasets"]

    header, row = _values(sheet)
    assert header[9:] == list(VALIDATION_HEADERS.values())
    assert row[9:] == [
        "INVALID",
        datetime.datetime(2026, 3, 4, 10, 30),
        "Dataset URL: A URL is required.",
    ]
    assert sheet["L2"].alignment.wrap_text


def test_the_validation_time_is_written_in_utc() -> None:
    """Excel has no time zones, so the header promises UTC and the value has to be it."""
    plus_two = datetime.timezone(datetime.timedelta(hours=2))
    record = _stored(validated_at=datetime.datetime(2026, 3, 4, 12, 30, tzinfo=plus_two))

    sheet = _load(build_workbook([record], include_validation=True))["Datasets"]

    assert sheet["K2"].value == datetime.datetime(2026, 3, 4, 10, 30)
    assert sheet["K2"].is_date


def test_without_validation_there_are_no_validation_columns() -> None:
    sheet = _load(build_workbook([_stored()]))["Datasets"]

    assert sheet.max_column == len(DATASETS_SHEET_FIELDS)


def test_a_value_starting_with_equals_stays_text() -> None:
    """Submitted text is untrusted: a formula would run when the supplier opens the file."""
    record = _stored(
        description="=HYPERLINK(\"http://evil\")",
        indicators_coverage=[{"name": "=1+1", "units": "=SUM(A1:A2)"}],
    )

    workbook = _load(build_workbook([record]))

    cells = [workbook["Datasets"]["F2"], *workbook["TABEL1_1"][2]]
    assert [cell.data_type for cell in cells] == ["s", "s", "s"]
    assert workbook["Datasets"]["F2"].value == "=HYPERLINK(\"http://evil\")"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ highlighting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _fills(sheet: Worksheet) -> set[str]:
    return {
        cell.coordinate
        for row in sheet.iter_rows()
        for cell in row
        if cell.fill.fgColor.rgb == _FLAGGED
    }


def test_the_cell_an_issue_points_at_is_highlighted() -> None:
    record = _stored(
        frequency_coverage="",
        validation_status=DiscoveryValidationStatus.INVALID,
        validation_issues=[
            {"field": "frequency_coverage", "message": "At least one frequency is required."},
            {"field": "description", "message": "A description is required."},
        ],
    )

    workbook = _load(build_workbook([record], include_validation=True))

    assert _fills(workbook["Datasets"]) == {"F2", "I2"}


def test_blank_units_are_highlighted_on_the_indicators_sheet() -> None:
    record = _stored(
        indicators_coverage=[
            {"name": "CPI", "units": ""},
            {"name": "M2", "units": "billions of Rp"},
            {"name": "GDP", "units": ""},
        ],
        validation_status=DiscoveryValidationStatus.INVALID,
        validation_issues=[
            {"field": "indicators_coverage", "message": "Units of measure are required."}
        ],
    )

    workbook = _load(build_workbook([record], include_validation=True))

    assert _fills(workbook["TABEL1_1"]) == {"B2", "B4"}
    # The indicators have no column on the `Datasets` sheet, so nothing is flagged there.
    assert _fills(workbook["Datasets"]) == set()


def test_nothing_is_highlighted_without_validation() -> None:
    record = _stored(
        indicators_coverage=[{"name": "CPI", "units": ""}],
        validation_status=DiscoveryValidationStatus.INVALID,
        validation_issues=[
            {"field": "indicators_coverage", "message": "Units of measure are required."},
            {"field": "url", "message": "A URL is required."},
        ],
    )

    workbook = _load(build_workbook([record]))

    assert _fills(workbook["Datasets"]) == _fills(workbook["TABEL1_1"]) == set()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ round trip ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("include_validation", [False, True])
def test_the_workbook_uploads_back_as_the_same_records(include_validation: bool) -> None:
    records = [
        _stored(
            url="https://example.org/1960",
            time_coverage="1960",
            validation_status=DiscoveryValidationStatus.INVALID,
            validation_issues=[
                {"field": "indicators_coverage", "message": "Units of measure are required."}
            ],
            indicators_coverage=[
                {"name": "CPI", "units": ""},
                {"name": "Broad money (M2)", "units": "billions of Rp"},
            ],
        ),
        _stored(agency="BI", dataset_id="SEKI_2", indicators_coverage=[]),
    ]

    parsed = _parse(build_workbook(records, include_validation=include_validation))

    assert not parsed.problems
    assert [row.values for row in parsed.rows] == [
        {name: getattr(record, name) for name in DATASETS_SHEET_FIELDS} for record in records
    ]
    assert [row.indicators for row in parsed.rows] == [
        read_indicators(record.indicators_coverage) for record in records
    ]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ refusals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_one_dataset_id_under_two_agencies_is_refused() -> None:
    """The upload refuses a workbook naming one sheet twice, so the export must not write one."""
    records = [_stored(agency="IMF", dataset_id="DOT"), _stored(agency="OECD", dataset_id="dot")]

    with pytest.raises(DiscoveryExportError) as error:
        build_workbook(records)

    message = str(error.value)
    assert "agency='OECD', dataset id='dot'" in message
    assert "'IMF'" in message
    assert "CSV" in message


@pytest.mark.parametrize("dataset_id", ["Datasets", "instructions"])
def test_a_dataset_id_naming_a_template_sheet_is_refused(dataset_id: str) -> None:
    with pytest.raises(DiscoveryExportError, match="workbook's own sheets"):
        build_workbook([_stored(dataset_id=dataset_id)])


@pytest.mark.parametrize("dataset_id", ["A" * 32, "TABLE[1]", "a/b", "History"])
def test_a_dataset_id_no_sheet_can_be_named_after_is_refused(dataset_id: str) -> None:
    """Only possible for a record stored before the key type enforced it."""
    with pytest.raises(DiscoveryExportError, match=re.escape(f"dataset id={dataset_id!r}")):
        build_workbook([_stored(dataset_id=dataset_id)])


def test_a_refusal_names_every_offending_record() -> None:
    records = [_stored(dataset_id="Datasets"), _stored(dataset_id="A" * 40)]

    with pytest.raises(DiscoveryExportError, match="^2 selected record"):
        build_workbook(records)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ service ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.asyncio
async def test_the_service_returns_the_workbook() -> None:
    service = AdminPortalDiscoveryDatasetService(session=MagicMock())
    service.get_record_models_by_channel = AsyncMock(return_value=[_stored()])  # type: ignore[method-assign]

    content = await service.export_discovery_datasets(
        channel_id=1,
        filters=DiscoveryExportFilters(),
        export_format=DiscoveryExportFormat.XLSX,
        include_validation=True,
    )

    assert _load(content).sheetnames == ["Datasets", "TABEL1_1"]
