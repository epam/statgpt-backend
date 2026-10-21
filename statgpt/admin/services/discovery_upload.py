"""Parsing of an uploaded discovery dataset file.

Pure and synchronous on purpose: this is the part with real branching, so it has to sit
where it can be unit-tested without a session or an HTTP app. Callers run it through
``asyncio.to_thread`` so a large workbook does not stall the event loop.

openpyxl is used directly rather than ``pandas.read_excel``: pandas coerces column dtypes
(``1960`` arrives as ``1960.0``), turns blanks into ``NaN``, mangles repeated headers, and
discards the cell coordinates the error report needs.

A workbook is two shapes at once. The `Datasets` sheet holds one row per dataset, and each
dataset's indicators live on a sheet of their own, named with the dataset's ID; a sheet no
dataset names is ignored, and never read. A CSV stays
flat: the indicators arrive as one already-joined column, which is also how the export
writes them.
"""

import csv
import datetime
import io
import logging
import zipfile
from collections.abc import Iterable, Sequence, Set
from dataclasses import dataclass, field, replace
from enum import StrEnum, auto
from typing import Any

import openpyxl
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.utils.exceptions import InvalidFileException
from openpyxl.worksheet._read_only import ReadOnlyWorksheet

from statgpt.common import schemas
from statgpt.common.services import normalize_key_part
from statgpt.common.utils import normalize_whitespace

from .exceptions import DiscoveryUploadFormatError

_log = logging.getLogger(__name__)

DATASETS_SHEET_FIELDS: tuple[str, ...] = (
    "reference_area",  # A
    "regional_coverage",  # B
    "agency",  # C
    "dataset_id",  # D
    "name",  # E
    "description",  # F
    "url",  # G
    "time_coverage",  # H
    "frequency_coverage",  # I
)
"""The columns of the `Datasets` sheet, in order, used for the positional fallback."""

RECORD_FIELDS: tuple[str, ...] = (*DATASETS_SHEET_FIELDS, "indicators_coverage")
"""Every field a submitter writes. The indicators are a sheet in a workbook, a column in a CSV."""

REQUIRED_FIELDS: tuple[str, ...] = ("agency", "dataset_id")
"""Halves of the natural key. A file without these columns cannot be reconciled at all."""

FIELD_LABELS: dict[str, str] = {
    "reference_area": "Reference area / country",
    "regional_coverage": "Regional coverage",
    "agency": "Agency / organization",
    "dataset_id": "Dataset ID",
    "name": "Dataset name",
    "description": "Description",
    "url": "Dataset URL",
    "time_coverage": "Time coverage",
    "frequency_coverage": "Frequency coverage",
    "indicators_coverage": "Indicators coverage",
}
"""Human labels, so an error message names the column the way the file does."""

_WORKBOOK_HEADERS: dict[str, str] = {
    "reference area / country": "reference_area",
    "regional coverage": "regional_coverage",
    "agency / organization": "agency",
    "dataset id": "dataset_id",
    "dataset name": "name",
    "description": "description",
    "dataset url": "url",
    "time coverage": "time_coverage",
    "frequency coverage": "frequency_coverage",
    "indicators coverage (incl. units of measure)": "indicators_coverage",
    "indicators coverage": "indicators_coverage",
}
"""How the template spells each header. `indicators_coverage` is a CSV column only."""

_RETIRED_FIELDS: tuple[str, ...] = ("excluded_regional_values", "missing_indicators")
"""Columns the current template dropped, on every format.

They are resolved rather than ignored, so a file that still carries one is refused by name:
the values are authored, and accepting the file while dropping the column would be
indistinguishable from data loss.
"""

_RETIRED_WORKBOOK_HEADERS: dict[str, str] = {
    "excluded regional values": "excluded_regional_values",
    # The old template spelled this column two ways: the short one on `Instructions`, the
    # long one on `Datasets`. Both have to be recognized to be refused.
    "relevant indicators not present": "missing_indicators",
    "relevant indicators not present in the dataset": "missing_indicators",
}

_RETIRED_LABELS: dict[str, str] = {
    "excluded_regional_values": "Excluded regional values",
    "missing_indicators": "Relevant indicators not present in the dataset",
}
"""Labels for the retired columns, kept only so a rejection can name what it found."""


def _field_name_aliases(name: str) -> tuple[str, ...]:
    """The spellings a field's own name arrives under: snake_case, spaced, camelCase.

    Carried alongside the workbook headers so a CSV exported by this application - whose
    headers are the field names - re-imports unchanged.
    """
    head, *rest = name.split("_")
    return name, name.replace("_", " "), head + "".join(part.title() for part in rest)


def _build_header_aliases() -> dict[str, str]:
    """Header text (normalized + casefolded) -> field name, retired fields included."""
    aliases = {**_WORKBOOK_HEADERS, **_RETIRED_WORKBOOK_HEADERS}
    for name in (*RECORD_FIELDS, *_RETIRED_FIELDS):
        for alias in _field_name_aliases(name):
            aliases[alias] = name
    return {key.casefold(): value for key, value in aliases.items()}


_HEADER_ALIASES: dict[str, str] = _build_header_aliases()

_DATASETS_SHEET_NAME: str = "datasets"
_RESERVED_SHEET_KEYS: frozenset[str] = frozenset({_DATASETS_SHEET_NAME, "instructions"})
"""Sheets of the template itself, as folded keys: never an indicators sheet, never an ID."""
_XLSX_SIGNATURE = b"PK\x03\x04"
_XLS_SIGNATURE = b"\xd0\xcf\x11\xe0"

_INDICATOR_NAME = "name"
_INDICATOR_UNITS = "units"

_INDICATOR_HEADERS: dict[str, str] = {
    "indicator": _INDICATOR_NAME,
    "indicators": _INDICATOR_NAME,
    "indicator name": _INDICATOR_NAME,
    "units of measure": _INDICATOR_UNITS,
    "unit of measure": _INDICATOR_UNITS,
    "units": _INDICATOR_UNITS,
    "unit": _INDICATOR_UNITS,
}

_INDICATOR_SHEET_WIDTH = 2
"""Columns of an indicator sheet, and the only width the positional fallback accepts."""

_INDICATOR_SEPARATOR = "; "
"""What joins the indicators of one dataset, matching every other list-bearing cell."""


class _DiscoveryFileFormat(StrEnum):
    XLSX = auto()
    CSV = auto()


_FORMAT_FIELDS: dict[_DiscoveryFileFormat, tuple[str, ...]] = {
    _DiscoveryFileFormat.XLSX: DATASETS_SHEET_FIELDS,
    _DiscoveryFileFormat.CSV: RECORD_FIELDS,
}
"""What each format's data columns are, in order. Anything else resolved is refused."""


@dataclass(frozen=True)
class ParsedRow:
    """One data row of an uploaded file, already whitespace-normalized."""

    row_number: int
    """1-based row number in the file, so an error can point at it."""

    values: dict[str, str]
    """Field name -> value. Fields whose column is absent are not present here."""


def _cell_reference(column_letters: dict[str, str], field_name: str, row_number: int) -> str | None:
    letter = column_letters.get(field_name)
    return f"{letter}{row_number}" if letter else None


@dataclass(frozen=True)
class ParsedFile:
    rows: list[ParsedRow] = field(default_factory=list)

    column_letters: dict[str, str] = field(default_factory=dict)
    """Field name -> column letter, so a problem can be reported as a cell reference."""

    problems: list[schemas.DiscoveryPayloadProblem] = field(default_factory=list)
    """What the file says that could not be used, located.

    Carried rather than raised so indicator-sheet trouble reaches the per-cell report
    alongside the schema violations, instead of collapsing the whole upload into one message.
    """

    def cell(self, field_name: str, row_number: int) -> str | None:
        """The cell reference for a field on a row, e.g. 'C14'. None if not a spreadsheet."""
        return _cell_reference(self.column_letters, field_name, row_number)


@dataclass(frozen=True)
class _SheetProblem:
    """Trouble on an indicator sheet, before it is known which dataset row owns it."""

    message: str
    cell: str


@dataclass(frozen=True)
class _IndicatorSheet:
    """One dataset's indicators, as a sheet supplied them."""

    text: str = ""
    """The joined cell this sheet becomes. Empty for a sheet holding no indicators."""

    problems: list[_SheetProblem] = field(default_factory=list)


def _detect_format(data: bytes, filename: str | None = None) -> _DiscoveryFileFormat:
    """Classify an upload by its signature bytes, falling back to the filename extension."""
    if data[:4] == _XLSX_SIGNATURE:
        return _DiscoveryFileFormat.XLSX
    if data[:4] == _XLS_SIGNATURE:
        raise DiscoveryUploadFormatError(
            "The file is a legacy Excel workbook (.xls). Open it in Excel and re-save it"
            " as .xlsx, or export it as CSV."
        )

    name = (filename or "").casefold()
    if name.endswith(".csv") or name.endswith(".txt"):
        return _DiscoveryFileFormat.CSV
    if name.endswith(".xlsx") or name.endswith(".xlsm"):
        raise DiscoveryUploadFormatError(
            "The file is named like an Excel workbook but is not one. Re-save it as .xlsx."
        )
    raise DiscoveryUploadFormatError(
        "Unsupported file type. Upload the discovery workbook as .xlsx, or its data as .csv."
    )


def _coerce_cell(value: Any) -> str:
    """Render a cell as text.

    openpyxl returns typed values for typed cells, and Pydantic's lax mode does not
    coerce ``int -> str``, so the conversion has to happen here. Order matters: ``bool``
    is a subclass of ``int``, and ``1960.0`` has to render as ``1960``, not ``1960.0``.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        # These cells are read by a search layer and by a person, not by Python.
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, datetime.datetime):
        return value.date().isoformat() if value.time() == datetime.time.min else value.isoformat()
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    return str(value)


def _retired_header_error(
    text: str, index: int, field_name: str, file_format: _DiscoveryFileFormat
) -> DiscoveryUploadFormatError:
    """Refuse a file that still carries a column the current template dropped.

    The whole file goes rather than the column: the submitter authored those values, and an
    upload that silently ignored them would look like data loss.
    """
    where = "The 'Datasets' sheet" if file_format is _DiscoveryFileFormat.XLSX else "The file"
    expected = ", ".join(FIELD_LABELS[name] for name in _FORMAT_FIELDS[file_format])
    hint = (
        " Indicators now go on one sheet per dataset, named with the dataset's ID."
        if file_format is _DiscoveryFileFormat.XLSX
        else ""
    )
    return DiscoveryUploadFormatError(
        f"{where} has {text!r} in column {get_column_letter(index + 1)}, which the current"
        f" discovery template does not have: this file follows the old template."
        f" Remove {_RETIRED_LABELS.get(field_name, text)!r} and re-upload."
        f" Expected columns: {expected}.{hint}"
    )


def _resolve_headers(
    header_cells: Sequence[Any], file_format: _DiscoveryFileFormat
) -> dict[int, str]:
    """Column index -> field name, from a header row.

    Unrecognized columns are ignored, so extra columns (including the status columns of an
    exported CSV) are tolerated. Two columns resolving to the same field is an error: one of
    them would be silently dropped. A column of the old template is an error too - see
    `_retired_header_error`.
    """
    accepted = frozenset(_FORMAT_FIELDS[file_format])
    resolved: dict[int, str] = {}
    seen: dict[str, int] = {}
    for index, cell in enumerate(header_cells):
        text = normalize_whitespace(_coerce_cell(cell))
        if not text:
            continue
        field_name = _HEADER_ALIASES.get(text.casefold())
        if field_name is None:
            continue
        if field_name not in accepted:
            raise _retired_header_error(text, index, field_name, file_format)
        if field_name in seen:
            raise DiscoveryUploadFormatError(
                f"Two columns describe {FIELD_LABELS[field_name]!r}"
                f" ({get_column_letter(seen[field_name] + 1)} and"
                f" {get_column_letter(index + 1)}). Remove one of them."
            )
        seen[field_name] = index
        resolved[index] = field_name
    return resolved


def _positional_headers(width: int, file_format: _DiscoveryFileFormat) -> dict[int, str]:
    """Column index -> field name for a file whose header row could not be recognized.

    Only usable when the row is exactly as wide as the template. Row 1 is still treated as
    a header rather than as data: every real file has one, and importing a renamed header
    as a record would create a junk row.
    """
    fields = _FORMAT_FIELDS[file_format]
    if width != len(fields):
        raise DiscoveryUploadFormatError(
            f"The header row does not name any known column. Expected the discovery"
            f" template's columns ({', '.join(FIELD_LABELS[f] for f in REQUIRED_FIELDS)},"
            f" ...) or exactly {len(fields)} columns in template order,"
            f" found {width}."
        )
    _log.warning(
        f"Discovery upload header row unrecognized; falling back to column order"
        f" A-{get_column_letter(len(fields))}."
    )
    return dict(enumerate(fields))


def _resolve_headers_or_positional(
    header_cells: Sequence[Any], file_format: _DiscoveryFileFormat
) -> dict[int, str]:
    resolved = _resolve_headers(header_cells, file_format)
    missing = [name for name in REQUIRED_FIELDS if name not in resolved.values()]
    if not missing:
        return resolved
    if resolved:
        raise DiscoveryUploadFormatError(
            "The file is missing required column(s): "
            + ", ".join(repr(FIELD_LABELS[name]) for name in missing)
            + "."
        )
    return _positional_headers(len(header_cells), file_format)


def _build_rows(
    data_rows: Iterable[tuple[int, Sequence[Any]]],
    headers: dict[int, str],
    max_rows: int,
) -> list[ParsedRow]:
    """Normalize the data rows of a file, stopping as soon as the row cap is exceeded.

    Takes an iterable rather than a list so the cap bounds memory: a sheet is streamed and
    refused while it is being read, instead of after it has been materialized in full. The
    byte cap on the upload is no substitute - xlsx is compressed XML, so a small file can
    expand by orders of magnitude.
    """
    rows: list[ParsedRow] = []

    for row_number, cells in data_rows:
        values = {
            field_name: normalize_whitespace(_coerce_cell(cells[index]))
            for index, field_name in headers.items()
            if index < len(cells)
        }
        if not any(values.values()):
            # The template ships hundreds of formatted-but-empty rows, and `ws.max_row`
            # is inflated in read-only mode, so blanks are passed over rather than stopped
            # at. They are not data, so they are dropped silently, not counted.
            continue
        rows.append(ParsedRow(row_number=row_number, values=values))
        if len(rows) > max_rows:
            raise DiscoveryUploadFormatError(
                f"The file has more than {max_rows} data rows. Split it and upload the parts."
            )

    return rows


def _column_letters(headers: dict[int, str]) -> dict[str, str]:
    return {name: get_column_letter(index + 1) for index, name in headers.items()}


def _get_worksheet(workbook: Workbook) -> ReadOnlyWorksheet:
    """Pick the sheet holding the records."""
    worksheet = None
    for sheet_name in workbook.sheetnames:
        if sheet_name.strip().casefold() == _DATASETS_SHEET_NAME:
            worksheet = workbook[sheet_name]
            break
    if worksheet is None:
        if len(workbook.sheetnames) < 2:
            raise DiscoveryUploadFormatError(
                f"The workbook has no {_DATASETS_SHEET_NAME.title()!r} sheet and no second"
                f" sheet to read instead. Upload the filled discovery template."
            )
        worksheet = workbook[workbook.sheetnames[1]]
    return worksheet


def _check_dataset_ids(
    rows: Sequence[ParsedRow], column_letters: dict[str, str], reserved: Set[str]
) -> list[schemas.DiscoveryPayloadProblem]:
    """Report a dataset ID a template sheet already holds, or a previous row already claimed.

    Workbook-wide and ignoring the agency, unlike the natural key: the ID names a sheet, so
    two records whose IDs agree cannot both carry indicators in one workbook however
    different the datasets are. Length and forbidden characters are not checked here - the
    schema's key type does that on every write path, and its violations already reach the
    per-cell report.
    """
    problems: list[schemas.DiscoveryPayloadProblem] = []
    first_seen: dict[str, ParsedRow] = {}

    for row in rows:
        value = row.values.get("dataset_id", "")
        if not value:
            continue
        key = normalize_key_part(value)
        if key in reserved:
            problems.append(
                schemas.DiscoveryPayloadProblem(
                    message=(
                        f"Dataset ID {value!r} is the name of one of the workbook's own"
                        f" sheets, so no indicators sheet can be named after it. Upload this"
                        f" dataset as a CSV."
                    ),
                    field="dataset_id",
                    row=row.row_number,
                    cell=_cell_reference(column_letters, "dataset_id", row.row_number),
                )
            )
            continue
        first = first_seen.setdefault(key, row)
        if first is row:
            continue
        problems.append(
            schemas.DiscoveryPayloadProblem(
                message=(
                    f"Dataset ID {value!r} is already used on row {first.row_number}."
                    f" It names the dataset's indicators sheet, so it has to be unique in"
                    f" the workbook - across agencies too. Upload them as separate files."
                ),
                field="dataset_id",
                row=row.row_number,
                cell=_cell_reference(column_letters, "dataset_id", row.row_number),
            )
        )

    return problems


def _resolve_indicator_headers(header_cells: Sequence[Any]) -> dict[str, int] | None:
    """Column -> index for an indicator sheet, or None if the sheet cannot be read.

    Falls back to column order only for a sheet exactly as wide as the template's two
    columns, the way the `Datasets` sheet's fallback works.
    """
    resolved: dict[str, int] = {}
    for index, cell in enumerate(header_cells):
        text = normalize_whitespace(_coerce_cell(cell)).casefold()
        column = _INDICATOR_HEADERS.get(text)
        if column is not None and column not in resolved:
            resolved[column] = index

    if _INDICATOR_NAME in resolved:
        return resolved
    # Read-only rows are padded out to the sheet's widest row, so trailing blanks do not count.
    width = len(header_cells)
    while width and not _coerce_cell(header_cells[width - 1]).strip():
        width -= 1
    if width == _INDICATOR_SHEET_WIDTH:
        return {_INDICATOR_NAME: 0, _INDICATOR_UNITS: 1}
    return None


def _indicator_value(cells: Sequence[Any], headers: dict[str, int], column: str) -> str:
    index = headers.get(column)
    if index is None or index >= len(cells):
        return ""
    return normalize_whitespace(_coerce_cell(cells[index]))


def _parse_indicator_sheet(
    worksheet: ReadOnlyWorksheet, max_indicators_per_dataset: int
) -> _IndicatorSheet:
    """Read one dataset's indicators sheet into the joined cell it becomes.

    An empty sheet is an empty list rather than a problem: it says the same thing as a
    dataset row with no sheet at all, and whether a record may have no indicators is
    validation's question, not this one's.
    """
    title = worksheet.title
    rows_iter = worksheet.values

    try:
        header_cells = list(next(rows_iter))
    except StopIteration:
        return _IndicatorSheet()

    headers = _resolve_indicator_headers(header_cells)
    if headers is None:
        return _IndicatorSheet(
            problems=[
                _SheetProblem(
                    message=(
                        f"Sheet {title!r} has no 'Indicator' column, so this dataset's"
                        f" indicators could not be read. Expected 'Indicator' and"
                        f" 'Units of measure', or exactly {_INDICATOR_SHEET_WIDTH} columns in"
                        f" template order."
                    ),
                    cell=f"'{title}'!A1",
                )
            ],
        )

    entries: list[str] = []
    problems: list[_SheetProblem] = []

    for row_number, cells in enumerate(rows_iter, start=2):
        name = _indicator_value(cells, headers, _INDICATOR_NAME)
        units = _indicator_value(cells, headers, _INDICATOR_UNITS)
        if not name and not units:
            # The template ships formatted-but-empty rows here too.
            continue
        if not name:
            letter = get_column_letter(headers.get(_INDICATOR_UNITS, 1) + 1)
            problems.append(
                _SheetProblem(
                    message=(
                        f"Sheet {title!r} gives units {units!r} with no indicator name,"
                        f" so the row says nothing the search could match."
                    ),
                    cell=f"'{title}'!{letter}{row_number}",
                )
            )
            continue
        entries.append(f"{name} ({units})" if units else name)
        if len(entries) > max_indicators_per_dataset:
            # Streamed and refused mid-read, like `max_rows`: a compressed sheet can expand
            # by orders of magnitude, so the byte cap on the upload bounds nothing here.
            raise DiscoveryUploadFormatError(
                f"Sheet {title!r} lists more than {max_indicators_per_dataset} indicators."
                f" Shorten the list, or split the dataset."
            )

    return _IndicatorSheet(text=_INDICATOR_SEPARATOR.join(entries), problems=problems)


def _index_sheet_titles(workbook: Workbook, reserved: Set[str]) -> dict[str, list[str]]:
    """Sheet titles keyed the way a dataset ID folds, the template's own sheets left out.

    Reads names only: a sheet's rows are loaded once a dataset row names it, so a sheet
    nothing refers to costs nothing.
    """
    index: dict[str, list[str]] = {}
    for title in workbook.sheetnames:
        key = normalize_key_part(title)
        if key not in reserved:
            index.setdefault(key, []).append(title)
    return index


def _apply_indicator_sheets(
    workbook: Workbook,
    rows: Sequence[ParsedRow],
    titles: dict[str, list[str]],
    column_letters: dict[str, str],
    max_indicators_per_dataset: int,
) -> tuple[list[ParsedRow], list[schemas.DiscoveryPayloadProblem]]:
    """Give each dataset row the indicators its sheet lists.

    A dataset row with no sheet and one whose sheet holds nothing both get an empty cell:
    the parser records how many indicators a dataset has, and validation decides whether
    that is allowed. A sheet no row names is ignored - a misspelt one leaves its dataset with
    no indicators, which validation reports against the dataset.
    """
    result: list[ParsedRow] = []
    problems: list[schemas.DiscoveryPayloadProblem] = []
    parsed: dict[str, _IndicatorSheet] = {}

    for row in rows:
        key = normalize_key_part(row.values.get("dataset_id", ""))
        matches = titles.get(key, [])
        sheet = _IndicatorSheet()
        if len(matches) > 1:
            problems.append(
                schemas.DiscoveryPayloadProblem(
                    message=(
                        f"Sheets {', '.join(repr(title) for title in matches)} all match"
                        f" Dataset ID {row.values['dataset_id']!r}, so it is unclear which"
                        f" holds its indicators. Keep one of them."
                    ),
                    field="dataset_id",
                    row=row.row_number,
                    cell=_cell_reference(column_letters, "dataset_id", row.row_number),
                )
            )
        elif matches:
            if key not in parsed:
                parsed[key] = _parse_indicator_sheet(
                    workbook[matches[0]], max_indicators_per_dataset
                )
            sheet = parsed[key]
            problems.extend(
                schemas.DiscoveryPayloadProblem(
                    message=problem.message,
                    field="indicators_coverage",
                    row=row.row_number,
                    cell=problem.cell,
                )
                for problem in sheet.problems
            )
        result.append(replace(row, values={**row.values, "indicators_coverage": sheet.text}))

    named = {normalize_key_part(row.values.get("dataset_id", "")) for row in rows}
    if unnamed := [title for key, group in titles.items() if key not in named for title in group]:
        _log.warning(f"Ignoring workbook sheets no Dataset ID names: {unnamed}.")

    return result, problems


def _parse_workbook(data: bytes, max_rows: int, max_indicators_per_dataset: int) -> ParsedFile:
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except (zipfile.BadZipFile, InvalidFileException, KeyError, ValueError) as e:
        raise DiscoveryUploadFormatError(
            "The workbook could not be read. It may be corrupted or password-protected;"
            " re-save it as .xlsx and try again."
        ) from e

    try:
        worksheet = _get_worksheet(workbook)

        rows_iter = worksheet.values
        try:
            header_cells = list(next(rows_iter))
        except StopIteration as e:
            raise DiscoveryUploadFormatError("The sheet is empty.") from e

        headers = _resolve_headers_or_positional(header_cells, _DiscoveryFileFormat.XLSX)
        # Consumed here, inside the `try`: streaming is what makes `max_rows` bound memory,
        # and the rows cannot be read once the `finally` below has closed the workbook.
        rows = _build_rows(enumerate(rows_iter, start=2), headers, max_rows)
        column_letters = _column_letters(headers)

        # The fallback may read another sheet as the datasets one, so its own title is
        # reserved alongside the template's.
        reserved = _RESERVED_SHEET_KEYS | {normalize_key_part(worksheet.title)}
        problems = _check_dataset_ids(rows, column_letters, reserved)
        rows, sheet_problems = _apply_indicator_sheets(
            workbook,
            rows,
            _index_sheet_titles(workbook, reserved),
            column_letters,
            max_indicators_per_dataset,
        )
        problems.extend(sheet_problems)
    finally:
        workbook.close()

    return ParsedFile(rows=rows, column_letters=column_letters, problems=problems)


def _decode_csv(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DiscoveryUploadFormatError(
        "The CSV file's encoding could not be determined. Re-save it as UTF-8."
    )


def _csv_dialect(text: str) -> type[csv.Dialect] | csv.Dialect:
    """Sniff the delimiter, since Excel writes ';' in some locales."""
    sample = text[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return csv.excel


def _parse_csv(data: bytes, max_rows: int) -> ParsedFile:
    text = _decode_csv(data)
    reader = csv.reader(io.StringIO(text, newline=""), dialect=_csv_dialect(text))

    try:
        header_cells = next(reader)
    except StopIteration as e:
        raise DiscoveryUploadFormatError("The CSV file is empty.") from e

    headers = _resolve_headers_or_positional(header_cells, _DiscoveryFileFormat.CSV)
    rows = _build_rows(enumerate(reader, start=2), headers, max_rows)
    # A CSV has no spreadsheet coordinates, so problems are reported by row only.
    return ParsedFile(rows=rows)


def parse_discovery_file(
    data: bytes,
    filename: str | None,
    max_rows: int,
    max_indicators_per_dataset: int,
) -> ParsedFile:
    """Parse an uploaded workbook or CSV into normalized rows.

    Raises `DiscoveryUploadFormatError` for anything that is not a readable discovery
    file - the caller maps that to a 400. Everything a readable file gets wrong about one
    dataset is carried on the result instead, so it can be reported cell by cell.
    """
    file_format = _detect_format(data, filename)
    _log.info(f"Parsing discovery upload {filename!r} as {file_format}.")

    if file_format is _DiscoveryFileFormat.XLSX:
        return _parse_workbook(data, max_rows, max_indicators_per_dataset)
    return _parse_csv(data, max_rows)
