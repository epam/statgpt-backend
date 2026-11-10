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

# openpyxl exposes the write-only sheet class under no public path.
from openpyxl.worksheet._write_only import WriteOnlyWorksheet
from pydantic import TypeAdapter, ValidationError

from statgpt.common import models, schemas, utils
from statgpt.common.schemas import DiscoveryValidationStatus
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

_VALIDATION_STATUS_FIELD = "validation_status"
_VALIDATED_AT_FIELD = "validated_at"
_VALIDATION_ISSUES_FIELD = "validation_issues"

VALIDATION_HEADERS: dict[str, str] = {
    _VALIDATION_STATUS_FIELD: "Validation status",
    _VALIDATED_AT_FIELD: "Validated at (UTC)",
    _VALIDATION_ISSUES_FIELD: "Validation issues",
}
"""Validation column -> its workbook header. A CSV names the columns by field name.

None of these may resolve to a record field on upload, or re-uploading a report would read
a verdict back in as data.
"""

_ISSUE_SEPARATOR = "\n"

_NOT_VALIDATED_NOTE = (
    "Not validated yet: no indexing job has checked this record since it was last written,"
    " so it may still have issues."
)
"""Said in the issues column instead of nothing, which would read as a clean record."""

_MAX_CELL_CHARS = 32_767
"""The most characters an Excel cell holds; a longer value makes Excel repair the file."""

_NAMED_RECORDS = 10
"""How many offending records a refusal names before falling back to a count."""

_BOLD = Font(bold=True)
_TOP = Alignment(vertical="top")
_WRAPPED = Alignment(vertical="top", wrap_text=True)
_FLAGGED = PatternFill(fill_type="solid", start_color="FFC7CE", end_color="FFC7CE")
"""Excel's own 'Bad' fill, on the cell an issue points at."""
_UNCHECKED = PatternFill(fill_type="solid", start_color="FFEB9C", end_color="FFEB9C")
"""Excel's own 'Neutral' fill, on the verdict of a record no indexing job has checked."""

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
    _VALIDATION_STATUS_FIELD: 18,
    _VALIDATED_AT_FIELD: 20,
    _VALIDATION_ISSUES_FIELD: 80,
}
_INDICATOR_COLUMN_WIDTHS = (60, 30)

_DATASET_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(DatasetIdKey)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ validation columns ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _issues(item: models.DiscoveryDataset) -> list[schemas.DiscoveryValidationIssue]:
    return [
        schemas.DiscoveryValidationIssue.model_validate(issue)
        for issue in item.validation_issues or ()
    ]


def _format_issues(issues: Iterable[schemas.DiscoveryValidationIssue]) -> str:
    """One issue per line, `<column label>: <message>`.

    An issue whose field is no column - a check that crashed reports its own name - keeps
    the raw name, so it still says which check to look at.
    """
    return _ISSUE_SEPARATOR.join(
        f"{FIELD_LABELS.get(issue.field, issue.field)}: {issue.message}" for issue in issues
    )


def _describe_issues(item: models.DiscoveryDataset) -> str:
    """The issues column: the issues, or a note that nothing has checked the record yet."""
    if item.validation_status is DiscoveryValidationStatus.NOT_VALIDATED:
        return _NOT_VALIDATED_NOTE
    return _format_issues(_issues(item))


def _utc(value: datetime.datetime | None) -> datetime.datetime | None:
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(datetime.timezone.utc)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CSV ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _csv_fieldnames(include_validation: bool) -> list[str]:
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
            row[_VALIDATION_STATUS_FIELD] = item.validation_status.value
            row[_VALIDATED_AT_FIELD] = validated_at.isoformat() if validated_at else ""
            row[_VALIDATION_ISSUES_FIELD] = _describe_issues(item)
        rows.append(row)
    return rows


def build_csv(records: Sequence[models.DiscoveryDataset], include_validation: bool = False) -> str:
    """The records as the CSV the channel export writes, plus the validation columns on request.

    The columns are named explicitly rather than taken from the first row, so a filter that
    matches nothing returns a header on its own instead of failing.
    """
    return utils.csv_from_dict_list(
        export_rows(records, include_validation), fieldnames=_csv_fieldnames(include_validation)
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ XLSX ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _describe(item: models.DiscoveryDataset) -> str:
    return f"agency={item.agency!r}, dataset id={item.dataset_id!r}"


def _sheet_title(item: models.DiscoveryDataset) -> str:
    """The indicators sheet's name: the dataset id as the write path normalizes it.

    The id is checked in this form, so the sheet has to be named in it too: a legacy id
    padded past Excel's 31 characters would otherwise pass the check and break the file.
    The upload pairs sheets with rows through the same normalization.
    """
    return utils.normalize_whitespace(item.dataset_id)


def _too_long_cells(item: models.DiscoveryDataset) -> list[str]:
    """The cells of the record that hold more than an Excel cell can, by label."""
    too_long = [
        FIELD_LABELS[name]
        for name in DATASETS_SHEET_FIELDS
        if len(getattr(item, name)) > _MAX_CELL_CHARS
    ]
    for indicator in schemas.read_indicators(item.indicators_coverage):
        for title, value in zip(INDICATOR_SHEET_HEADERS, (indicator.name, indicator.units)):
            if len(value) > _MAX_CELL_CHARS:
                too_long.append(f"{title} {_shorten(indicator.name)!r}")
    return too_long


def _shorten(text: str, limit: int = 40) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


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
    workbook naming one sheet twice. And no value may exceed what an Excel cell holds.
    """
    problems: list[str] = []
    first_seen: dict[str, models.DiscoveryDataset] = {}
    for item in records:
        if too_long := _too_long_cells(item):
            problems.append(
                f"{_describe(item)}: longer than the {_MAX_CELL_CHARS:,} characters a workbook"
                f" cell holds: {', '.join(too_long)}"
            )
        if reason := _sheet_name_problem(_sheet_title(item)):
            problems.append(f"{_describe(item)}: the dataset id {reason}")
            continue
        first = first_seen.setdefault(normalize_key_part(item.dataset_id), item)
        if first is not item:
            problems.append(
                f"{_describe(item)}: the dataset id is also used by agency {first.agency!r}"
            )
    return problems


def _check_workbook_exportable(records: Sequence[models.DiscoveryDataset]) -> None:
    """Refuse records a workbook cannot hold, rather than rename sheets.

    A workbook whose sheets were renamed or cells cut to fit would download fine and then
    upload back as something else, which defeats the point of the file. A CSV has none of
    these limits.
    """
    problems = _workbook_problems(records)
    if not problems:
        return
    shown = "; ".join(problems[:_NAMED_RECORDS])
    remaining = len(problems) - _NAMED_RECORDS
    more = f"; and {remaining} more" if remaining > 0 else ""
    raise DiscoveryExportError(
        f"{len(problems)} problem(s) keep the selected records from a workbook: {shown}{more}."
        f" Export them as CSV, or narrow the selection, for example to one agency."
    )


def _text_cell(sheet: WriteOnlyWorksheet, value: str, fill: PatternFill | None = None) -> Cell:
    """A text cell, never a formula: the text was submitted, so it is not trusted."""
    cell = WriteOnlyCell(sheet, value=ILLEGAL_CHARACTERS_RE.sub("", value) or None)
    if cell.value is not None:
        # openpyxl takes text starting with '=' for a formula.
        cell.data_type = "s"
    cell.alignment = _TOP
    if fill is not None:
        cell.fill = fill
    return cell


def _header_row(sheet: WriteOnlyWorksheet, titles: Iterable[str]) -> list[Cell]:
    cells: list[Cell] = []
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
) -> list[Cell]:
    unchecked = item.validation_status is DiscoveryValidationStatus.NOT_VALIDATED
    fill = _UNCHECKED if unchecked else None
    status = _text_cell(sheet, item.validation_status.value, fill)

    validated_at = _utc(item.validated_at)
    # Excel has no time zones, so the value is written naive, in UTC, as the header says.
    when = WriteOnlyCell(sheet, value=validated_at.replace(tzinfo=None) if validated_at else None)
    when.number_format = _DATETIME_FORMAT
    when.alignment = _TOP

    described = _text_cell(sheet, _describe_issues(item), fill)
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
            _text_cell(sheet, getattr(item, name), _FLAGGED if name in flagged else None)
            for name in DATASETS_SHEET_FIELDS
        ]
        if include_validation:
            row.extend(_validation_cells(sheet, item))
        sheet.append(row)


def _write_indicators_sheet(
    workbook: Workbook, item: models.DiscoveryDataset, include_validation: bool
) -> None:
    """The dataset's indicators, on a sheet named with its id - even when it lists none,
    so the submitter has the sheet to fill in.

    In a report, every blank units cell is flagged: units are always required, whether or
    not an indexing job has said so yet. A missing list flags nothing here, there being no
    row to flag; the issues column says it.
    """
    sheet = workbook.create_sheet(_sheet_title(item))
    _set_widths(sheet, _INDICATOR_COLUMN_WIDTHS)
    sheet.freeze_panes = "A2"
    sheet.append(_header_row(sheet, INDICATOR_SHEET_HEADERS))

    for indicator in schemas.read_indicators(item.indicators_coverage):
        flag_units = include_validation and not indicator.units
        sheet.append(
            [
                _text_cell(sheet, indicator.name),
                _text_cell(sheet, indicator.units, _FLAGGED if flag_units else None),
            ]
        )


def build_workbook(
    records: Sequence[models.DiscoveryDataset], include_validation: bool = False
) -> bytes:
    """The records as the upload template's workbook, plus the validation columns on request.

    Raises `DiscoveryExportError` for records no workbook can hold - see
    `_check_workbook_exportable`. Write-only, so rows are streamed out as they are appended
    instead of held as cell objects until the save.
    """
    _check_workbook_exportable(records)

    workbook = Workbook(write_only=True)
    _write_datasets_sheet(workbook, records, include_validation)
    for item in records:
        _write_indicators_sheet(workbook, item, include_validation)

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
