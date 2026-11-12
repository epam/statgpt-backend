"""Discovery dataset helpers shared by the `discovery` commands and `content init`."""

import glob
import os
from collections.abc import Iterable

from rich.table import Table

from statgpt.cli.shared.console import console, print_error, print_warning
from statgpt.common.schemas import DiscoveryPayloadErrorDetail, DiscoveryUploadSummary

UPLOAD_SUFFIXES = (".xlsx", ".csv")
"""What the API accepts: the discovery workbook, or its data exported as CSV."""


def is_upload_file(filename: str) -> bool:
    """Whether a file name looks like a discovery upload.

    Excel lock files (`~$...`) sit next to an open workbook and are named like one, so they
    have to be excluded by name - there is nothing in them to upload.
    """
    name = os.path.basename(filename)
    if name.startswith("~$") or name.startswith("."):
        return False
    return name.casefold().endswith(UPLOAD_SUFFIXES)


_GLOB_CHARS = frozenset("*?[")


def _expand_upload_path(value: str) -> list[str]:
    """The files one `--file` value names: a directory's upload files, a glob's, or itself."""
    if os.path.isdir(value):
        return [
            entry.path
            for entry in os.scandir(value)
            if entry.is_file() and is_upload_file(entry.name)
        ]
    if _GLOB_CHARS & set(value):
        return [path for path in glob.glob(value) if os.path.isfile(path) and is_upload_file(path)]
    if not os.path.exists(value):
        raise ValueError(f"File not found: {value}")
    return [value]


def expand_upload_paths(values: Iterable[str]) -> list[str]:
    """Turn `--file` values into the files of one upload: absolute, de-duplicated, sorted.

    A directory stands for the `.xlsx` and `.csv` files directly inside it, excluding lock
    and hidden files. A value with glob characters is expanded here, for a pattern the shell
    left quoted. Raises `ValueError` naming the first value that is missing or yields no
    file, so nothing is sent for a batch that is not what the admin meant.
    """
    paths: set[str] = set()
    for value in values:
        expanded = _expand_upload_path(value)
        if not expanded:
            raise ValueError(f"No .xlsx or .csv file matches: {value}")
        paths.update(os.path.abspath(path) for path in expanded)
    return sorted(paths)


_SUMMARY_FIELDS = [
    ("created", "Created"),
    ("updated", "Updated"),
    ("unchanged", "Unchanged"),
    ("deleted", "Deleted"),
    ("rows_read", "Rows read"),
]


def summary_line(summary: DiscoveryUploadSummary) -> str:
    """One-line account of what an upload did, for a batch report or a log line."""
    return (
        f"{summary.created} created, {summary.updated} updated,"
        f" {summary.unchanged} unchanged, {summary.deleted} deleted"
    )


def render_upload_summary(summary: DiscoveryUploadSummary, title: str = "Upload results") -> None:
    """Print what an upload did to the records a channel holds."""
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Records")
    table.add_column("Count", justify="right")
    for field, label in _SUMMARY_FIELDS:
        table.add_row(label, str(getattr(summary, field)))
    console.print(table)

    if len(summary.files) > 1:
        files = Table(show_header=True, header_style="bold")
        files.add_column("File", overflow="fold")
        files.add_column("Rows read", justify="right")
        for entry in summary.files:
            files.add_row(entry.file, str(entry.rows_read))
        console.print(files)


def render_payload_problems(detail: DiscoveryPayloadErrorDetail) -> None:
    """Print the per-record report a refused write comes back with.

    Nothing was saved, so the point of the output is telling the submitter which cells to fix.
    """
    print_error(detail.message)

    if not detail.problems:
        return

    # Only a file upload names files, so a JSON payload's report keeps its old columns.
    with_file = any(problem.file for problem in detail.problems)

    table = Table(show_header=True, header_style="bold")
    if with_file:
        table.add_column("File", ratio=1, overflow="fold")
    table.add_column("Row", width=6, justify="right")
    table.add_column("Cell", width=6)
    table.add_column("Field", ratio=1)
    table.add_column("Problem", ratio=3, overflow="fold")

    for problem in detail.problems:
        row = str(problem.row) if problem.row is not None else ""
        if not row and problem.index is not None:
            row = f"#{problem.index}"
        cells = [row, problem.cell or "", problem.field or "", problem.message]
        if with_file:
            cells.insert(0, problem.file or "")
        table.add_row(*cells)

    console.print(table)

    if detail.truncated:
        print_warning("More problems were found than are listed above.")
