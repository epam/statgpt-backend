"""Per-item outcome tracking for batch CLI operations.

Batch commands (``content init``, ``channel reindex``) process many independent items.
A failure in one item must not stop the ones after it, otherwise a single unsupported
dataset silently truncates the batch and which datasets survive depends on the order
the configs happen to be read in. Handlers therefore record an outcome per item and
render one summary at the end, instead of raising on the first error.
"""

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, Field
from rich.markup import escape
from rich.table import Table

from statgpt.cli.shared.console import console, print_error, print_success, print_warning

# Error bodies and titles are arbitrary text; keep a single table cell readable.
_MAX_MESSAGE_LENGTH = 300


class BatchPartialFailureError(Exception):
    """Raised after a batch summary has been rendered, when at least one item failed.

    The summary already explains what went wrong, so the CLI entry points map this to a
    non-zero exit code without printing another error line on top of it.
    """


class BatchItemStatus(StrEnum):
    OK = "ok"
    UNCHANGED = "unchanged"
    SKIPPED = "skipped"
    """Not attempted, because something it depends on failed or is missing."""
    FAILED = "failed"


# Statuses worth listing individually: the ones a user has to act on. Successes are
# already reported inline while the batch runs.
_ACTIONABLE = (BatchItemStatus.FAILED, BatchItemStatus.SKIPPED)

_RENDER_ORDER = {
    BatchItemStatus.FAILED: 0,
    BatchItemStatus.SKIPPED: 1,
    BatchItemStatus.OK: 2,
    BatchItemStatus.UNCHANGED: 3,
}

_STATUS_STYLES = {
    BatchItemStatus.FAILED: "red",
    BatchItemStatus.SKIPPED: "yellow",
    BatchItemStatus.OK: "green",
    BatchItemStatus.UNCHANGED: "dim",
}


def _one_line(message: str | None) -> str:
    """Collapse a message to a single truncated line fit for a table cell."""
    if not message:
        return ""
    collapsed = " ".join(message.split())
    if len(collapsed) > _MAX_MESSAGE_LENGTH:
        collapsed = collapsed[: _MAX_MESSAGE_LENGTH - 1].rstrip() + "…"
    return collapsed


class BatchItemResult(BaseModel):
    kind: str = Field(description="What the item is, e.g. 'dataset' or 'data source'")
    item_id: str = Field(description="How the user identifies the item, e.g. a dataset URN")
    status: BatchItemStatus
    message: str | None = Field(
        default=None, description="Why it failed, or what it was waiting on when skipped"
    )


class BatchReport(BaseModel):
    """Accumulates per-item outcomes of one batch operation."""

    title: str
    items: list[BatchItemResult] = Field(default_factory=list)

    def record(
        self, kind: str, item_id: str, status: BatchItemStatus, message: str | None = None
    ) -> None:
        self.items.append(
            BatchItemResult(kind=kind, item_id=item_id, status=status, message=message)
        )

    def record_ok(self, kind: str, item_id: str) -> None:
        self.record(kind, item_id, BatchItemStatus.OK)

    def record_unchanged(self, kind: str, item_id: str) -> None:
        self.record(kind, item_id, BatchItemStatus.UNCHANGED)

    def record_skipped(self, kind: str, item_id: str, message: str) -> None:
        self.record(kind, item_id, BatchItemStatus.SKIPPED, message)

    def record_failed(self, kind: str, item_id: str, message: str) -> None:
        self.record(kind, item_id, BatchItemStatus.FAILED, message)

    @property
    def failed(self) -> list[BatchItemResult]:
        return [item for item in self.items if item.status == BatchItemStatus.FAILED]

    @property
    def skipped(self) -> list[BatchItemResult]:
        return [item for item in self.items if item.status == BatchItemStatus.SKIPPED]

    @property
    def has_failures(self) -> bool:
        return any(item.status == BatchItemStatus.FAILED for item in self.items)

    def counts(self) -> dict[BatchItemStatus, int]:
        counter = Counter(item.status for item in self.items)
        return {status: counter[status] for status in _RENDER_ORDER if counter[status]}

    def render(self) -> None:
        """Print failures and skips as a table, then a count per status."""
        console.print()
        console.print(f"[bold cyan]{escape(self.title)}[/bold cyan]")

        if not self.items:
            console.print("[dim]Nothing to do.[/dim]")
            return

        actionable = [item for item in self.items if item.status in _ACTIONABLE]
        if actionable:
            table = Table(show_header=True, header_style="bold")
            table.add_column("Status", width=9)
            table.add_column("Type", width=14)
            table.add_column("Item", ratio=2)
            table.add_column("Details", ratio=3)

            for item in sorted(actionable, key=lambda i: (_RENDER_ORDER[i.status], i.item_id)):
                style = _STATUS_STYLES[item.status]
                table.add_row(
                    f"[{style}]{item.status.value.upper()}[/{style}]",
                    escape(item.kind),
                    escape(item.item_id),
                    escape(_one_line(item.message)),
                )

            console.print(table)

        summary = ", ".join(f"{count} {status.value}" for status, count in self.counts().items())
        if self.has_failures:
            print_error(summary)
        elif self.skipped:
            print_warning(summary)
        else:
            print_success(summary)
