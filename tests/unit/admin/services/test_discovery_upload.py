import csv
import datetime
import io
from typing import Any

import openpyxl
import pytest

from statgpt.admin.services import discovery_upload
from statgpt.admin.services.discovery_upload import (
    DATASETS_SHEET_FIELDS,
    RECORD_FIELDS,
    ParsedFile,
    parse_discovery_file,
)
from statgpt.admin.services.exceptions import DiscoveryUploadFormatError

_DATASETS_HEADERS = [
    "Reference area / country",
    "Regional coverage",
    "Agency / organization",
    "Dataset ID",
    "Dataset name",
    "Description",
    "Dataset URL",
    "Time coverage",
    "Frequency coverage",
]

_ROW = [
    "Indonesia (IDN)",
    "None",
    "Bank Indonesia (BI)",
    "TABEL1_1",
    "Broad Money",
    "Money and banking table.",
    "https://www.bi.go.id/SEKI/tabel/TABEL1_1.xls",
    "From 1989-01 to 2026-06",
    "Monthly",
]

_INDICATOR_HEADERS = ["Indicator", "Units of measure"]

_INDICATORS = [
    ["broad money (M2)", "billions of Rp"],
    ["narrow money (M1)", "billions of Rp"],
]


def _workbook(
    rows: list[list[Any]],
    headers: list[Any] | None = None,
    sheet_name: str = "Datasets",
    extra_sheet_first: bool = True,
    indicator_sheets: dict[str, list[list[Any]]] | None = None,
    indicator_headers: list[Any] | None = None,
) -> bytes:
    """Build a workbook in memory, so no binary fixture has to be committed.

    `indicator_sheets` maps a sheet name to its data rows; the header row is added for it.
    Passing `None` gives the datasets rows one sheet each, named with their dataset ID.
    """
    wb = openpyxl.Workbook()
    first = wb.active
    assert first is not None
    if extra_sheet_first:
        # The template ships an `Instructions` sheet before `Datasets`.
        first.title = "Instructions"
        sheet = wb.create_sheet(sheet_name)
    else:
        first.title = sheet_name
        sheet = first

    sheet.append(_DATASETS_HEADERS if headers is None else headers)
    for row in rows:
        sheet.append(row)

    if indicator_sheets is None:
        dataset_id_index = _DATASETS_HEADERS.index("Dataset ID")
        indicator_sheets = {
            str(row[dataset_id_index]): list(_INDICATORS)
            for row in rows
            if len(row) > dataset_id_index and row[dataset_id_index]
        }

    for title, indicator_rows in indicator_sheets.items():
        indicators = wb.create_sheet(title)
        indicators.append(_INDICATOR_HEADERS if indicator_headers is None else indicator_headers)
        for row in indicator_rows:
            indicators.append(row)

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str], delimiter: str = ",") -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _parse(
    data: bytes,
    filename: str = "book.xlsx",
    *,
    max_rows: int = 100,
    max_indicators: int = 1000,
) -> ParsedFile:
    return parse_discovery_file(
        data, filename, max_rows=max_rows, max_indicators_per_dataset=max_indicators
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the Datasets sheet ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_parses_template_headers_and_reports_cell_references() -> None:
    parsed = _parse(_workbook([_ROW]))

    assert len(parsed.rows) == 1
    row = parsed.rows[0]
    assert row.row_number == 2
    assert row.values["agency"] == "Bank Indonesia (BI)"
    # Column C of the template, on the first data row.
    assert parsed.cell("agency", row.row_number) == "C2"
    assert not parsed.problems


def test_reads_second_sheet_when_datasets_sheet_is_absent() -> None:
    data = _workbook([_ROW], sheet_name="Sheet2")

    parsed = _parse(data)

    assert parsed.rows[0].values["dataset_id"] == "TABEL1_1"


def test_single_sheet_without_datasets_name_is_rejected() -> None:
    data = _workbook([_ROW], sheet_name="Sheet1", extra_sheet_first=False, indicator_sheets={})

    with pytest.raises(DiscoveryUploadFormatError, match="no second sheet"):
        _parse(data)


def test_falls_back_to_column_order_when_headers_are_unrecognized() -> None:
    """A renamed header row is still readable, and is never imported as a record."""
    headers = [f"col{i}" for i in range(len(DATASETS_SHEET_FIELDS))]

    parsed = _parse(_workbook([_ROW], headers=headers))

    assert len(parsed.rows) == 1
    assert parsed.rows[0].values["agency"] == "Bank Indonesia (BI)"


def test_partially_recognized_headers_missing_the_key_are_rejected() -> None:
    headers = ["Dataset name", "Description"] + [f"col{i}" for i in range(7)]

    with pytest.raises(DiscoveryUploadFormatError, match="missing required column"):
        _parse(_workbook([_ROW], headers=headers))


def test_two_columns_for_one_field_are_rejected() -> None:
    headers = _DATASETS_HEADERS + ["Agency / organization"]

    with pytest.raises(DiscoveryUploadFormatError, match="Two columns"):
        _parse(_workbook([_ROW], headers=headers))


def test_unknown_extra_columns_are_ignored() -> None:
    headers = _DATASETS_HEADERS + ["validation_status", "Some note"]

    parsed = _parse(_workbook([_ROW + ["INVALID", "note"]], headers=headers))

    assert parsed.rows[0].values["agency"] == "Bank Indonesia (BI)"
    assert "validation_status" not in parsed.rows[0].values


@pytest.mark.parametrize(
    "retired_header",
    [
        "Excluded regional values",
        "Indicators coverage (incl. units of measure)",
        "Relevant indicators not present in the dataset",
        "missing_indicators",
    ],
)
def test_a_workbook_following_the_old_template_is_refused(retired_header: str) -> None:
    """The values were authored, so dropping the column silently would look like data loss."""
    headers = _DATASETS_HEADERS + [retired_header]

    with pytest.raises(DiscoveryUploadFormatError, match="follows the old template"):
        _parse(_workbook([_ROW + ["something"]], headers=headers))


@pytest.mark.parametrize(
    "cell_value, expected",
    [
        (1960, "1960"),
        (1960.0, "1960"),
        (1.5, "1.5"),
        (True, "Yes"),
        (False, "No"),
        (None, ""),
        (datetime.datetime(2026, 5, 1), "2026-05-01"),
        (datetime.date(2026, 5, 1), "2026-05-01"),
    ],
)
def test_typed_cells_are_coerced_to_text(cell_value: object, expected: str) -> None:
    """openpyxl returns typed values, and Pydantic's lax mode will not coerce int -> str."""
    row: list = list(_ROW)
    row[7] = cell_value  # Time coverage

    parsed = _parse(_workbook([row]))

    assert parsed.rows[0].values["time_coverage"] == expected


def test_whitespace_is_normalized() -> None:
    """The agency cell below holds a literal U+00A0 after 'Bank', plus padding and runs."""
    row = list(_ROW)
    row[2] = "  Bank  Indonesia   (BI) "

    parsed = _parse(_workbook([row]))

    assert parsed.rows[0].values["agency"] == "Bank Indonesia (BI)"


def test_blank_rows_are_dropped_without_stopping_the_scan() -> None:
    """The template ships ~300 formatted-but-empty rows between and after the data.

    They are not data, so they are dropped silently rather than reported as skipped: a
    count inflated by template formatting only misleads whoever reads the upload summary.
    """
    blank: list = [None] * len(_DATASETS_HEADERS)
    whitespace_only = [" "] * len(_DATASETS_HEADERS)
    second = list(_ROW)
    second[3] = "TABEL1_2"

    parsed = _parse(_workbook([_ROW, blank, whitespace_only, second, blank]))

    assert [row.values["dataset_id"] for row in parsed.rows] == ["TABEL1_1", "TABEL1_2"]
    # Row numbers still point at the real spreadsheet rows.
    assert [row.row_number for row in parsed.rows] == [2, 5]


def test_row_cap_is_enforced() -> None:
    rows = []
    for index in range(3):
        row = list(_ROW)
        row[3] = f"TABEL{index}"
        rows.append(row)

    with pytest.raises(DiscoveryUploadFormatError, match="more than 2 data rows"):
        _parse(_workbook(rows), max_rows=2)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ dataset ids ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_a_dataset_id_used_twice_is_reported_even_across_agencies() -> None:
    """The id names a sheet, so one workbook cannot hold two datasets claiming it."""
    second = list(_ROW)
    second[2] = "Statistics Indonesia (BPS)"

    parsed = _parse(_workbook([_ROW, second]))

    assert [problem.row for problem in parsed.problems] == [3]
    problem = parsed.problems[0]
    assert problem.field == "dataset_id"
    assert problem.cell == "D3"
    assert "already used on row 2" in problem.message


def test_a_dataset_id_repeated_in_another_casing_is_reported() -> None:
    second = list(_ROW)
    second[3] = "tabel1_1 "
    sheets = {"TABEL1_1": list(_INDICATORS)}

    parsed = _parse(_workbook([_ROW, second], indicator_sheets=sheets))

    assert [problem.row for problem in parsed.problems] == [3]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ indicator sheets ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_indicator_rows_are_joined_with_their_units() -> None:
    parsed = _parse(_workbook([_ROW]))

    assert parsed.rows[0].values["indicators_coverage"] == (
        "broad money (M2) (billions of Rp); narrow money (M1) (billions of Rp)"
    )
    assert not parsed.problems


def test_an_indicator_without_units_is_carried_as_its_name_alone() -> None:
    sheets = {"TABEL1_1": [["broad money (M2)", None], ["policy rate", "percent per annum"]]}

    parsed = _parse(_workbook([_ROW], indicator_sheets=sheets))

    assert parsed.rows[0].values["indicators_coverage"] == (
        "broad money (M2); policy rate (percent per annum)"
    )


def test_the_sheet_is_matched_to_its_dataset_ignoring_case_and_spacing() -> None:
    sheets = {" tabel1_1 ": list(_INDICATORS)}

    parsed = _parse(_workbook([_ROW], indicator_sheets=sheets))

    assert parsed.rows[0].values["indicators_coverage"].startswith("broad money (M2)")
    assert not parsed.problems


def test_a_dataset_with_no_indicator_sheet_gets_an_empty_list() -> None:
    """Not a problem here: whether a record may list no indicators is validation's call."""
    parsed = _parse(_workbook([_ROW], indicator_sheets={}))

    assert parsed.rows[0].values["indicators_coverage"] == ""
    assert not parsed.problems


def test_a_sheet_holding_only_its_header_row_reads_the_same_as_no_sheet() -> None:
    parsed = _parse(_workbook([_ROW], indicator_sheets={"TABEL1_1": []}))

    assert parsed.rows[0].values["indicators_coverage"] == ""
    assert not parsed.problems


def test_blank_indicator_rows_are_dropped() -> None:
    sheets = {"TABEL1_1": [[None, None], ["broad money (M2)", "billions of Rp"], [" ", " "]]}

    parsed = _parse(_workbook([_ROW], indicator_sheets=sheets))

    assert parsed.rows[0].values["indicators_coverage"] == "broad money (M2) (billions of Rp)"
    assert not parsed.problems


def test_units_without_an_indicator_name_are_reported_at_their_cell() -> None:
    sheets = {"TABEL1_1": [["broad money (M2)", "billions of Rp"], [None, "percent"]]}

    parsed = _parse(_workbook([_ROW], indicator_sheets=sheets))

    assert len(parsed.problems) == 1
    problem = parsed.problems[0]
    assert problem.field == "indicators_coverage"
    # Located on the indicators sheet, but reported against the row that owns it.
    assert problem.row == 2
    assert problem.cell == "'TABEL1_1'!B3"
    # The indicators that could be read are still carried.
    assert parsed.rows[0].values["indicators_coverage"] == "broad money (M2) (billions of Rp)"


def test_a_sheet_no_dataset_row_names_is_ignored_unread() -> None:
    """Never parsed, so even a sheet over the indicator cap does not refuse the upload."""
    sheets = {
        "TABEL1_1": list(_INDICATORS),
        "Sheet1": [[f"indicator {index}", "percent"] for index in range(4)],
    }

    parsed = _parse(_workbook([_ROW], indicator_sheets=sheets), max_indicators=3)

    assert not parsed.problems
    assert parsed.rows[0].values["indicators_coverage"].startswith("broad money (M2)")


def test_two_sheets_folding_onto_one_dataset_id_are_reported_on_its_row() -> None:
    """Neither can be picked over the other, so the row says which sheets collide."""
    sheets = {"TABEL1_1": list(_INDICATORS), "tabel1_1 ": [["policy rate", "percent"]]}

    parsed = _parse(_workbook([_ROW], indicator_sheets=sheets))

    assert len(parsed.problems) == 1
    problem = parsed.problems[0]
    assert problem.field == "dataset_id"
    assert problem.cell == "D2"
    assert "'TABEL1_1', 'tabel1_1 '" in problem.message
    assert parsed.rows[0].values["indicators_coverage"] == ""


@pytest.mark.parametrize("dataset_id", ["Instructions", "datasets "])
def test_a_dataset_id_naming_a_template_sheet_is_reported(dataset_id: str) -> None:
    row = list(_ROW)
    row[3] = dataset_id

    parsed = _parse(_workbook([row], indicator_sheets={}))

    assert [(problem.field, problem.cell) for problem in parsed.problems] == [("dataset_id", "D2")]
    assert "workbook's own sheets" in parsed.problems[0].message
    assert parsed.rows[0].values["indicators_coverage"] == ""


def test_an_indicator_sheet_without_an_indicator_column_is_reported() -> None:
    sheets = {"TABEL1_1": [["broad money (M2)", "billions of Rp", "extra"]]}

    parsed = _parse(_workbook([_ROW], indicator_sheets=sheets, indicator_headers=["a", "b", "c"]))

    assert len(parsed.problems) == 1
    assert "no 'Indicator' column" in parsed.problems[0].message
    assert parsed.rows[0].values["indicators_coverage"] == ""


def test_a_two_column_indicator_sheet_falls_back_to_column_order() -> None:
    sheets = {"TABEL1_1": list(_INDICATORS)}

    parsed = _parse(
        _workbook(
            [_ROW], indicator_sheets=sheets, indicator_headers=["Name", "Unit of measurement"]
        )
    )

    assert parsed.rows[0].values["indicators_coverage"].startswith("broad money (M2)")
    assert not parsed.problems


def test_the_column_order_fallback_ignores_trailing_blank_header_cells() -> None:
    """A stray cell further right widens every row the sheet yields, the header included."""
    sheets = {"TABEL1_1": [["broad money (M2)", "billions of Rp", "note"]]}

    parsed = _parse(
        _workbook(
            [_ROW], indicator_sheets=sheets, indicator_headers=["Name", "Unit of measurement"]
        )
    )

    assert parsed.rows[0].values["indicators_coverage"] == "broad money (M2) (billions of Rp)"
    assert not parsed.problems


def test_the_instructions_sheet_is_not_read_as_indicators() -> None:
    parsed = _parse(_workbook([_ROW]))

    assert not parsed.problems


def test_the_per_sheet_indicator_cap_is_enforced() -> None:
    sheets = {"TABEL1_1": [[f"indicator {index}", "percent"] for index in range(4)]}

    with pytest.raises(DiscoveryUploadFormatError, match="more than 3 indicators"):
        _parse(_workbook([_ROW], indicator_sheets=sheets), max_indicators=3)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ other formats ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_legacy_xls_is_rejected_with_actionable_message() -> None:
    with pytest.raises(DiscoveryUploadFormatError, match="re-save it as .xlsx"):
        _parse(b"\xd0\xcf\x11\xe0some legacy content", "book.xls")


def test_corrupt_workbook_is_a_format_error_not_a_crash() -> None:
    with pytest.raises(DiscoveryUploadFormatError):
        _parse(b"PK\x03\x04not really a zip")


def test_unknown_file_type_is_rejected() -> None:
    with pytest.raises(DiscoveryUploadFormatError, match="Unsupported file type"):
        _parse(b"%PDF-1.4", "book.pdf")


def test_csv_with_field_names_round_trips_an_export() -> None:
    """A CSV exported by this application (snake_case headers) re-imports unchanged."""
    fieldnames = list(RECORD_FIELDS)
    row = dict(zip(fieldnames, [*_ROW, "broad money (M2) (billions of Rp)"]))

    parsed = _parse(_csv_bytes([row], fieldnames), "discovery_datasets.csv")

    assert parsed.rows[0].values == row
    assert parsed.rows[0].values["indicators_coverage"] == "broad money (M2) (billions of Rp)"
    # A CSV has no spreadsheet coordinates, so problems are reported by row only.
    assert parsed.cell("agency", 2) is None


def test_csv_accepts_the_workbook_spelling_of_the_indicators_column() -> None:
    headers = [*_DATASETS_HEADERS, "Indicators coverage"]
    row = dict(zip(headers, [*_ROW, "broad money (M2)"]))

    parsed = _parse(_csv_bytes([row], headers), "book.csv")

    assert parsed.rows[0].values["indicators_coverage"] == "broad money (M2)"


@pytest.mark.parametrize("retired_header", ["Excluded regional values", "missing_indicators"])
def test_csv_following_the_old_template_is_refused(retired_header: str) -> None:
    headers = [*_DATASETS_HEADERS, retired_header]
    row = dict(zip(headers, [*_ROW, "something"]))

    with pytest.raises(DiscoveryUploadFormatError, match="follows the old template"):
        _parse(_csv_bytes([row], headers), "book.csv")


def test_csv_with_workbook_headers_and_semicolon_delimiter() -> None:
    row = dict(zip(_DATASETS_HEADERS, _ROW))

    parsed = _parse(_csv_bytes([row], _DATASETS_HEADERS, delimiter=";"), "book.csv")

    assert parsed.rows[0].values["agency"] == "Bank Indonesia (BI)"
    assert parsed.rows[0].values["dataset_id"] == "TABEL1_1"


def test_csv_without_the_indicators_column_is_accepted() -> None:
    """Emptiness is caught once, at validation time, not on the write path."""
    row = dict(zip(_DATASETS_HEADERS, _ROW))

    parsed = _parse(_csv_bytes([row], _DATASETS_HEADERS), "book.csv")

    assert "indicators_coverage" not in parsed.rows[0].values


def test_csv_with_bom_and_cp1252_fallback() -> None:
    fieldnames = list(RECORD_FIELDS)
    row = dict(zip(fieldnames, [*_ROW, "broad money (M2)"]))
    row["agency"] = "Banco de España (BdE)"

    with_bom = "﻿".encode() + _csv_bytes([row], fieldnames)
    assert _parse(with_bom, "x.csv").rows[0].values["agency"] == row["agency"]

    as_cp1252 = _csv_bytes([row], fieldnames).decode().encode("cp1252")
    assert _parse(as_cp1252, "x.csv").rows[0].values["agency"] == row["agency"]


def test_empty_csv_is_rejected() -> None:
    with pytest.raises(DiscoveryUploadFormatError, match="empty"):
        _parse(b"", "x.csv")


def test_csv_reads_an_indicators_column_longer_than_pythons_field_limit(
    default_csv_field_limit: int,
) -> None:
    """A workbook's indicators sheet has no cell limit, so its export can exceed this."""
    coverage = "; ".join(f"indicator {i} (units)" for i in range(10_000))
    assert len(coverage) > default_csv_field_limit
    row = dict(zip(DATASETS_SHEET_FIELDS, _ROW), indicators_coverage=coverage)

    parsed = _parse(_csv_bytes([row], list(RECORD_FIELDS)), "x.csv", max_indicators=10_000)

    assert parsed.rows[0].values["indicators_coverage"] == coverage
    assert not parsed.problems


def test_csv_indicator_cap_is_reported_against_the_row() -> None:
    """The cap a workbook's indicators sheet is held to, so the two formats agree."""
    within = dict(zip(DATASETS_SHEET_FIELDS, _ROW), indicators_coverage="a (x); b (x); c (x)")
    over = {**within, "dataset_id": "TABEL1_2", "indicators_coverage": "a; b;; c; d;"}

    parsed = _parse(_csv_bytes([within, over], list(RECORD_FIELDS)), "x.csv", max_indicators=3)

    [problem] = parsed.problems
    assert problem.field == "indicators_coverage"
    assert problem.row == 3
    assert "4 indicators, more than 3" in problem.message
    assert len(parsed.rows) == 2


def test_unreadable_csv_is_a_format_error(
    default_csv_field_limit: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(discovery_upload, "allow_csv_fields_up_to", lambda limit: None)
    row = dict(zip(DATASETS_SHEET_FIELDS, _ROW), description="x" * (default_csv_field_limit + 1))

    with pytest.raises(DiscoveryUploadFormatError, match="could not be read"):
        _parse(_csv_bytes([row], list(DATASETS_SHEET_FIELDS)), "x.csv")
