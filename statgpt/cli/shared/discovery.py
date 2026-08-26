"""Discovery dataset helpers shared by the `discovery` commands and `content init`."""

import os

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


_SUMMARY_FIELDS = [
    ("created", "Created"),
    ("updated", "Updated"),
    ("unchanged", "Unchanged"),
    ("deleted", "Deleted"),
    ("rows_read", "Rows read"),
    ("rows_skipped", "Rows skipped"),
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


def render_payload_problems(detail: DiscoveryPayloadErrorDetail) -> None:
    """Print the per-record report a refused write comes back with.

    Nothing was saved, so the point of the output is telling the submitter which cells to fix.
    """
    print_error(detail.message)

    if not detail.problems:
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Row", width=6, justify="right")
    table.add_column("Cell", width=6)
    table.add_column("Field", ratio=1)
    table.add_column("Problem", ratio=3)

    for problem in detail.problems:
        row = str(problem.row) if problem.row is not None else ""
        if not row and problem.index is not None:
            row = f"#{problem.index}"
        table.add_row(row, problem.cell or "", problem.field or "", problem.message)

    console.print(table)

    if detail.truncated:
        print_warning("More problems were found than are listed above.")
