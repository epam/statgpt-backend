"""Console output utilities using Rich."""

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table
from rich.text import Text

# Global console instance
console = Console()


def print_banner(version: str = "1.0.0") -> None:
    """Print the CLI welcome banner."""
    banner_text = Text()
    banner_text.append("StatGPT CLI", style="bold cyan")
    banner_text.append(f" v{version}\n", style="dim")
    banner_text.append("Press ", style="dim")
    banner_text.append("Tab", style="bold green")
    banner_text.append(" for suggestions, ", style="dim")
    banner_text.append("help", style="bold green")
    banner_text.append(" for commands, ", style="dim")
    banner_text.append("exit", style="bold green")
    banner_text.append(" to quit", style="dim")

    console.print(Panel(banner_text, border_style="cyan", padding=(0, 1)))


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]\u2713[/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]\u2717[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]\u26a0[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]\u2139[/bold blue] {message}")


def mask_secret(value: str | None, visible_chars: int = 4) -> str:
    """Mask a secret value, showing only last few characters."""
    if value is None:
        return "[dim]\u2717 not set[/dim]"
    if len(value) <= visible_chars:
        return "\u2022" * len(value)
    return "\u2022" * 12 + value[-visible_chars:]


def create_settings_table() -> Table:
    """Create a table for displaying settings."""
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
        expand=True,
    )
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_column("Source", style="dim", justify="right")
    return table


def create_data_table(title: str, columns: list[tuple[str, str]]) -> Table:
    """Create a data table with specified columns.

    Args:
        title: Table title
        columns: List of (column_name, style) tuples
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    for col_name, style in columns:
        table.add_column(col_name, style=style)
    return table


class SpinnerStatus:
    """Helper class for updating spinner message."""

    def __init__(self, progress: Progress, task_id: TaskID):
        self._progress = progress
        self._task_id = task_id
        self._message = ""

    def update(self, message: str) -> None:
        """Update the spinner message."""
        self._message = message
        self._progress.update(self._task_id, description=message)

    @property
    def message(self) -> str:
        """Get the current message."""
        return self._message


@contextmanager
def spinner_status(message: str) -> Generator[SpinnerStatus, None, None]:
    """Show a spinner that displays completion status when done.

    Displays a spinner while the context is active, then replaces it with
    ✓ (green tick) on success or ✗ (red cross) on exception.

    Args:
        message: The message to display next to the spinner

    Yields:
        SpinnerStatus object that can be used to update the message

    Example:
        with spinner_status("Fetching data...") as status:
            # do work
            status.update("Processing results...")
        # Shows ✓ Processing results... on success
        # Shows ✗ Processing results... on exception
    """
    status = None
    error_occurred = False

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(message, total=None)
        status = SpinnerStatus(progress, task_id)
        status._message = message

        try:
            yield status
        except Exception:
            error_occurred = True
            raise

    final_message = status.message if status else message
    if error_occurred:
        console.print(f"[bold red]\u2717[/bold red] {final_message}")
    else:
        console.print(f"[bold green]\u2713[/bold green] {final_message}")
