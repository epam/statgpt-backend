"""The Excel report of a channel's discovery indicator summaries, for `discovery summary-report`.

One `Summaries` sheet lists every record, and each record with summarization calls gets a sheet
of its own holding them, linked both ways.
"""

import datetime
from collections.abc import Iterable, Sequence
from typing import Any

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.hyperlink import Hyperlink
from openpyxl.worksheet.worksheet import Worksheet
from pydantic import BaseModel, ConfigDict, ValidationError

from statgpt.common.schemas import DiscoveryDataset, DiscoverySummaryCall
from statgpt.common.schemas.discovery_dataset import (
    FORBIDDEN_DATASET_ID_CHARS,
    MAX_DATASET_ID_LENGTH,
)

SUMMARIES_SHEET = "Summaries"

MAX_CELL_CHARS = 32_767
"""What Excel holds in one cell. openpyxl writes longer text, and Excel then repairs the file."""

_TRUNCATED = "… [truncated]"

_RESERVED_SHEET_NAMES = (SUMMARIES_SHEET, "History")
"""Taken before any dataset sheet is named: the report's own sheet, and the one Excel keeps."""

_FALLBACK_SHEET_NAME = "dataset"

_INDICATOR_SEPARATOR = ";"

_CALLS_HEADER_ROW = 3

_BOLD = Font(bold=True)
_LINK = Font(color="0563C1", underline="single")
_TOP = Alignment(vertical="top")
_WRAPPED = Alignment(vertical="top", wrap_text=True)

_SUMMARY_COLUMNS = [
    ("Agency", 24, False),
    ("Dataset ID", 20, False),
    ("Name", 40, True),
    ("Validation", 14, False),
    ("Indexing", 12, False),
    ("Summarized at (UTC)", 20, False),
    ("Indicators", 11, False),
    ("Coverage chars", 15, False),
    ("Summary chars", 15, False),
    ("Ratio", 8, False),
    ("Parts", 7, False),
    ("Answers", 9, False),
    ("Rejected", 9, False),
    ("Last run failed", 15, False),
    ("Indicators coverage", 60, False),
    ("Indicators summary", 80, True),
]
"""Header, width and whether the text wraps. The raw coverage does not: it can run for pages."""

_RATIO_COLUMN = 10

_CALL_COLUMNS = [
    ("Run", 10, False),
    ("#", 5, False),
    ("Step", 8, False),
    ("Part", 6, False),
    ("Accepted", 10, False),
    ("Error", 50, True),
    ("Reasoning", 60, True),
    ("Units", 30, True),
    ("Groups", 80, True),
]


class _Answer(BaseModel):
    """The fields of a stored answer the report shows.

    Lenient on purpose: answers are stored as the model returned them, under whichever output
    model the run used, and a report must not fail on an older shape.
    """

    model_config = ConfigDict(extra="ignore")

    reasoning: str | None = None
    units: str | None = None
    groups: list[str] | None = None


def _parse_answer(call: DiscoverySummaryCall) -> _Answer:
    if call.answer is None:
        return _Answer()
    try:
        return _Answer.model_validate(call.answer)
    except ValidationError:
        return _Answer()


def _text(value: str) -> str:
    """Text as a cell can hold it: no control characters, within Excel's cell limit."""
    value = ILLEGAL_CHARACTERS_RE.sub("", value)
    if len(value) > MAX_CELL_CHARS:
        return value[: MAX_CELL_CHARS - len(_TRUNCATED)] + _TRUNCATED
    return value


def _sheet_name(dataset_id: str, taken: set[str]) -> str:
    """A sheet name for a record, unique among `taken` ignoring case, which it is added to.

    The dataset id is not unique on its own: two agencies can publish the same id, and records
    stored before ids had to be sheet-safe can hold characters no sheet name may.
    """
    base = "".join("_" if char in FORBIDDEN_DATASET_ID_CHARS else char for char in dataset_id)
    base = base.strip("'")[:MAX_DATASET_ID_LENGTH].strip("'") or _FALLBACK_SHEET_NAME

    name = base
    number = 1
    while name.casefold() in taken:
        number += 1
        suffix = f" ({number})"
        name = base[: MAX_DATASET_ID_LENGTH - len(suffix)] + suffix

    taken.add(name.casefold())
    return name


def _link(cell: Any, sheet: str, target: str) -> None:
    """Make a cell link to a cell of a sheet in the same workbook."""
    quoted = sheet.replace("'", "''")
    cell.hyperlink = Hyperlink(ref=cell.coordinate, location=f"'{quoted}'!{target}")
    cell.font = _LINK


def _write_row(sheet: Worksheet, row: int, values: Iterable[Any], wrapped: Sequence[bool]) -> None:
    for column, (value, wrap) in enumerate(zip(values, wrapped), start=1):
        if isinstance(value, str):
            value = _text(value)
        cell = sheet.cell(row=row, column=column, value=value)
        # openpyxl takes text starting with '=' for a formula; an answer is never one.
        if isinstance(value, str) and cell.data_type == "f":
            cell.data_type = "s"
        cell.alignment = _WRAPPED if wrap else _TOP


def _write_header(
    sheet: Worksheet, row: int, columns: Sequence[tuple[str, int, bool]], last_row: int
) -> None:
    for column, (title, width, _) in enumerate(columns, start=1):
        sheet.cell(row=row, column=column, value=title).font = _BOLD
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.freeze_panes = f"A{row + 1}"
    sheet.auto_filter.ref = f"A{row}:{get_column_letter(len(columns))}{max(last_row, row)}"


def _excel_datetime(value: datetime.datetime | None) -> datetime.datetime | None:
    """A datetime Excel can store: it has no time zones, so UTC, and naive."""
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def _count_indicators(coverage: str) -> int:
    return sum(1 for item in coverage.split(_INDICATOR_SEPARATOR) if item.strip())


def _summary_values(record: DiscoveryDataset) -> list[Any]:
    calls = record.indicators_summary_calls or []
    coverage_chars = len(record.indicators_coverage)
    summary_chars = len(record.indicators_summary)
    return [
        record.agency,
        record.dataset_id,
        record.name,
        record.validation_status.value,
        record.indexing_status.value,
        _excel_datetime(record.indicators_summarized_at),
        _count_indicators(record.indicators_coverage),
        coverage_chars,
        summary_chars,
        summary_chars / coverage_chars if coverage_chars else None,
        len({call.part for call in calls if call.part is not None}),
        len(calls),
        sum(1 for call in calls if call.error),
        bool(record.indicators_summary_failed_calls),
        record.indicators_coverage,
        record.indicators_summary,
    ]


def _call_rows(record: DiscoveryDataset) -> Iterable[list[Any]]:
    runs = [
        ("succeeded", record.indicators_summary_calls or []),
        ("failed", record.indicators_summary_failed_calls or []),
    ]
    for run, calls in runs:
        for number, call in enumerate(calls, start=1):
            answer = _parse_answer(call)
            yield [
                run,
                number,
                call.step.value,
                call.part,
                call.error is None,
                call.error,
                answer.reasoning,
                answer.units,
                "\n".join(answer.groups) if answer.groups else None,
            ]


def _write_calls_sheet(
    sheet: Worksheet, record: DiscoveryDataset, summaries_row: int, title: str
) -> None:
    back = sheet.cell(row=1, column=1, value=f"← {SUMMARIES_SHEET}")
    _link(back, SUMMARIES_SHEET, f"B{summaries_row}")
    sheet.cell(row=1, column=2, value=_text(title)).font = _BOLD

    wrapped = [wrap for _, _, wrap in _CALL_COLUMNS]
    row = _CALLS_HEADER_ROW
    for row, values in enumerate(_call_rows(record), start=_CALLS_HEADER_ROW + 1):
        _write_row(sheet, row, values, wrapped)
    _write_header(sheet, _CALLS_HEADER_ROW, _CALL_COLUMNS, row)


def _has_calls(record: DiscoveryDataset) -> bool:
    return bool(record.indicators_summary_calls or record.indicators_summary_failed_calls)


def build_summary_report(records: Sequence[DiscoveryDataset]) -> Workbook:
    """Lay out the records' indicator summaries and their LLM calls as a workbook."""
    workbook = Workbook()
    summaries = workbook.worksheets[0]
    summaries.title = SUMMARIES_SHEET

    taken = {name.casefold() for name in _RESERVED_SHEET_NAMES}
    wrapped = [wrap for _, _, wrap in _SUMMARY_COLUMNS]
    ordered = sorted(records, key=lambda r: (r.agency.casefold(), r.dataset_id.casefold()))

    row = 1
    for row, record in enumerate(ordered, start=2):
        _write_row(summaries, row, _summary_values(record), wrapped)
        summaries.cell(row=row, column=_RATIO_COLUMN).number_format = "0%"
        if _has_calls(record):
            name = _sheet_name(record.dataset_id, taken)
            _write_calls_sheet(
                workbook.create_sheet(name), record, row, f"{record.agency} · {record.dataset_id}"
            )
            _link(summaries.cell(row=row, column=2), name, "A1")
    _write_header(summaries, 1, _SUMMARY_COLUMNS, row)

    return workbook
