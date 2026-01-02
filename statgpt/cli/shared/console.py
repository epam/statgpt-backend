"""Console output utilities using Rich."""

from rich.console import Console
from rich.panel import Panel
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
