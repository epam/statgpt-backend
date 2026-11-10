"""Rendering of a channel's discovery datasets as a downloadable file.

Pure and synchronous, like `discovery_upload`, so it can be unit-tested without a session;
callers run the workbook through ``asyncio.to_thread``. Both formats are shaped to go
straight back into the upload: the header and sheet names are the parser's own constants,
so the writer and the parser cannot drift apart.

The validation columns are optional. They carry the verdict of the last indexing job, for
whoever supplied the data to correct it, and the upload ignores them: a verdict belongs to
the channel holding the record and is re-derived by its next indexing job.
"""

import datetime
import io
from collections.abc import Iterable, Sequence
from typing import Any

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE, Cell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet._write_only import WriteOnlyWorksheet
from pydantic import TypeAdapter, ValidationError

from statgpt.common import models, schemas, utils
from statgpt.common.schemas.discovery_dataset import DatasetIdKey
from statgpt.common.services import normalize_key_part

from .discovery_upload import (
    DATASETS_SHEET_FIELDS,
    DATASETS_SHEET_TITLE,
    FIELD_LABELS,
    INDICATOR_SHEET_HEADERS,
    INDICATORS_FIELD,
    PYDANTIC_VALUE_ERROR_PREFIX,
    RECORD_FIELDS,
    RESERVED_SHEET_KEYS,
)
from .exceptions import DiscoveryExportError

VALIDATION_STATUS_FIELD = "validation_status"
VALIDATED_AT_FIELD = "validated_at"
VALIDATION_ISSUES_FIELD = "validation_issues"

VALIDATION_HEADERS: dict[str, str] = {
    VALIDATION_STATUS_FIELD: "Validation status",
    VALIDATED_AT_FIELD: "Validated at (UTC)",
    VALIDATION_ISSUES_FIELD: "Validation issues",
}
"""Validation column -> its workbook header. A CSV names the columns by field name.

None of these may resolve to a record field on upload, or re-uploading a report would read
a verdict back in as data.
"""

_ISSUE_SEPARATOR = "\n"

_NAMED_RECORDS = 10
"""How many offending records a refusal names before falling back to a count."""

_BOLD = Font(bold=True)
_TOP = Alignment(vertical="top")
_WRAPPED = Alignment(vertical="top", wrap_text=True)
_FLAGGED = PatternFill(fill_type="solid", start_color="FFC7CE", end_color="FFC7CE")
"""Excel's own 'Bad' fill, on the cell an issue points at."""

_DATETIME_FORMAT = "yyyy-mm-dd hh:mm:ss"

_COLUMN_WIDTHS: dict[str, int] = {
    "reference_area": 24,
    "regional_coverage": 24,
    "agency": 20,
    "dataset_id": 20,
    "name": 40,
    "description": 60,
    "url": 40,
    "time_coverage": 16,
    "frequency_coverage": 20,
    VALIDATION_STATUS_FIELD: 18,
    VALIDATED_AT_FIELD: 20,
    VALIDATION_ISSUES_FIELD: 80,
}
_INDICATOR_COLUMN_WIDTHS = (60, 30)

_DATASET_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(DatasetIdKey)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ validation columns ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _issues(item: models.DiscoveryDataset) -> list[schemas.DiscoveryValidationIssue]:
    return [
        schemas.DiscoveryValidationIssue.model_validate(issue)
        for issue in item.validation_issues or ()
    ]


def format_issues(issues: Iterable[schemas.DiscoveryValidationIssue]) -> str:
    """One issue per line, `<column label>: <message>`.

    An issue whose field is no column - a check that crashed reports its own name - keeps
    the raw name, so it still says which check to look at.
    """
    return _ISSUE_SEPARATOR.join(
        f"{FIELD_LABELS.get(issue.field, issue.field)}: {issue.message}" for issue in issues
    )


def _utc(value: datetime.datetime | None) -> datetime.datetime | None:
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(datetime.timezone.utc)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CSV ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def csv_fieldnames(include_validation: bool) -> list[str]:
    return [*RECORD_FIELDS, *(VALIDATION_HEADERS if include_validation else ())]


def export_rows(
    records: Iterable[models.DiscoveryDataset], include_validation: bool = False
) -> list[dict[str, str]]:
    """Shape records for the export CSV.

    Without `include_validation` only the submitted fields travel: validation and indexing
    state describe this deployment's index rather than the dataset. With it, the verdict of
    the last indexing job is appended, which the upload ignores.

    Read straight off the row rather than through the write schema, so a record stored
    before the current rules - a dataset id no sheet could be named after - is still
    exportable. The export says what is stored; refusing a write is the write's business.
    The indicators are the exception, written as the JSON list a CSV carries them in.
    """
    rows: list[dict[str, str]] = []
    for item in records:
        row = {
            **{name: getattr(item, name) for name in RECORD_FIELDS},
            INDICATORS_FIELD: schemas.dump_indicators_json(
                schemas.read_indicators(item.indicators_coverage)
            ),
        }
        if include_validation:
            validated_at = _utc(item.validated_at)
            row[VALIDATION_STATUS_FIELD] = item.validation_status.value
            row[VALIDATED_AT_FIELD] = validated_at.isoformat() if validated_at else ""
            row[VALIDATION_ISSUES_FIELD] = format_issues(_issues(item))
        rows.append(row)
    return rows


def build_csv(records: Sequence[models.DiscoveryDataset], include_validation: bool = False) -> str:
    """The records as the CSV the channel export writes, plus the validation columns on request.

    The columns are named explicitly rather than taken from the first row, so a filter that
    matches nothing returns a header on its own instead of failing.
    """
    return utils.csv_from_dict_list(
        export_rows(records, include_validation), fieldnames=csv_fieldnames(include_validation)
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ XLSX ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _describe(item: models.DiscoveryDataset) -> str:
    return f"agency={item.agency!r}, dataset id={item.dataset_id!r}"


def _sheet_name_problem(dataset_id: str) -> str | None:
    """Why no indicators sheet can be named after a dataset id, or None if one can."""
    if normalize_key_part(dataset_id) in RESERVED_SHEET_KEYS:
        return "is the name of one of the workbook's own sheets"
    try:
        _DATASET_ID_ADAPTER.validate_python(dataset_id)
    except ValidationError as e:
        return str(e.errors()[0]["msg"]).removeprefix(PYDANTIC_VALUE_ERROR_PREFIX)
    return None


def _workbook_problems(records: Sequence[models.DiscoveryDataset]) -> list[str]:
    """What keeps the records from one workbook that the upload would read back.

    The dataset id names the dataset's indicators sheet, so it has to be a valid sheet name,
    not one of the template's own, and unique across agencies - the upload refuses a
    workbook naming one sheet twice.
    """
    problems: list[str] = []
    first_seen: dict[str, models.DiscoveryDataset] = {}
    for item in records:
        if reason := _sheet_name_problem(item.dataset_id):
            problems.append(f"{_describe(item)}: the dataset id {reason}")
            continue
        first = first_seen.setdefault(normalize_key_part(item.dataset_id), item)
        if first is not item:
            problems.append(
                f"{_describe(item)}: the dataset id is also used by agency {first.agency!r}"
            )
    return problems


def check_workbook_exportable(records: Sequence[models.DiscoveryDataset]) -> None:
    """Refuse records a workbook cannot hold, rather than rename sheets.

    A workbook whose sheets were renamed to fit would download fine and then fail to upload
    back, which defeats the point of the file. A CSV has none of these limits.
    """
    problems = _workbook_problems(records)
    if not problems:
        return
    shown = "; ".join(problems[:_NAMED_RECORDS])
    remaining = len(problems) - _NAMED_RECORDS
    more = f"; and {remaining} more" if remaining > 0 else ""
    raise DiscoveryExportError(
        f"{len(problems)} selected record(s) cannot be exported as a workbook, since a"
        f" dataset id names its indicators sheet: {shown}{more}. Export them as CSV, or"
        f" narrow the selection, for example to one agency."
    )


def _text_cell(sheet: WriteOnlyWorksheet, value: str, flagged: bool = False) -> Cell:
    """A text cell, never a formula: the text was submitted, so it is not trusted."""
    cell = WriteOnlyCell(sheet, value=ILLEGAL_CHARACTERS_RE.sub("", value) or None)
    if cell.value is not None:
        # openpyxl takes text starting with '=' for a formula.
        cell.data_type = "s"
    cell.alignment = _TOP
    if flagged:
        cell.fill = _FLAGGED
    return cell


def _header_row(sheet: WriteOnlyWorksheet, titles: Iterable[str]) -> list[Cell]:
    cells = []
    for title in titles:
        cell = WriteOnlyCell(sheet, value=title)
        cell.font = _BOLD
        cells.append(cell)
    return cells


def _set_widths(sheet: WriteOnlyWorksheet, widths: Iterable[int]) -> None:
    # Set before the first row is written, which a write-only sheet supports; the stubs
    # leave it out.
    for column, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(column)].width = width  # type: ignore[attr-defined]


def _validation_cells(
    sheet: WriteOnlyWorksheet,
    item: models.DiscoveryDataset,
    issues: Sequence[schemas.DiscoveryValidationIssue],
) -> list[Cell]:
    status = _text_cell(sheet, item.validation_status.value)

    validated_at = _utc(item.validated_at)
    # Excel has no time zones, so the value is written naive, in UTC, as the header says.
    when = WriteOnlyCell(sheet, value=validated_at.replace(tzinfo=None) if validated_at else None)
    when.number_format = _DATETIME_FORMAT
    when.alignment = _TOP

    described = _text_cell(sheet, format_issues(issues))
    described.alignment = _WRAPPED
    return [status, when, described]


def _write_datasets_sheet(
    workbook: Workbook, records: Sequence[models.DiscoveryDataset], include_validation: bool
) -> None:
    sheet = workbook.create_sheet(DATASETS_SHEET_TITLE)
    columns = [*DATASETS_SHEET_FIELDS, *(VALIDATION_HEADERS if include_validation else ())]
    _set_widths(sheet, (_COLUMN_WIDTHS[name] for name in columns))
    sheet.freeze_panes = "A2"

    titles = [FIELD_LABELS[name] for name in DATASETS_SHEET_FIELDS]
    if include_validation:
        titles.extend(VALIDATION_HEADERS.values())
    sheet.append(_header_row(sheet, titles))

    for item in records:
        issues = _issues(item) if include_validation else []
        flagged = {issue.field for issue in issues}
        row: list[Any] = [
            _text_cell(sheet, getattr(item, name), flagged=name in flagged)
            for name in DATASETS_SHEET_FIELDS
        ]
        if include_validation:
            row.extend(_validation_cells(sheet, item, issues))
        sheet.append(row)


def _write_indicators_sheet(
    workbook: Workbook, item: models.DiscoveryDataset, include_validation: bool
) -> None:
    """The dataset's indicators, on a sheet named with its id - even when it lists none,
    so the submitter has the sheet to fill in.

    A record whose indicators are at fault has each blank units cell flagged. An issue about
    a missing list flags nothing here, there being no row to flag; the issues column says it.
    """
    sheet = workbook.create_sheet(item.dataset_id)
    _set_widths(sheet, _INDICATOR_COLUMN_WIDTHS)
    sheet.freeze_panes = "A2"
    sheet.append(_header_row(sheet, INDICATOR_SHEET_HEADERS))

    flag_units = include_validation and any(
        issue.field == INDICATORS_FIELD for issue in _issues(item)
    )
    for indicator in schemas.read_indicators(item.indicators_coverage):
        sheet.append(
            [
                _text_cell(sheet, indicator.name),
                _text_cell(sheet, indicator.units, flagged=flag_units and not indicator.units),
            ]
        )


def build_workbook(
    records: Sequence[models.DiscoveryDataset], include_validation: bool = False
) -> bytes:
    """The records as the upload template's workbook, plus the validation columns on request.

    Raises `DiscoveryExportError` for records no workbook can hold - see
    `check_workbook_exportable`. Write-only, so rows are streamed out as they are appended
    instead of held as cell objects until the save.
    """
    check_workbook_exportable(records)

    workbook = Workbook(write_only=True)
    _write_datasets_sheet(workbook, records, include_validation)
    for item in records:
        _write_indicators_sheet(workbook, item, include_validation)

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
