"""Interactive prompts for StatGPT CLI."""

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.prompt import Confirm

from statgpt.cli.settings import cli_runtime


class NonInteractiveError(Exception):
    """Raised when an interactive prompt is required but non-interactive mode is enabled.

    This exception is caught by command handlers to provide helpful error messages
    indicating which parameters must be provided via command-line arguments.
    """


def confirm_interactive(
    prompt: str,
    default: bool = False,
    error_message: str | None = None,
) -> bool:
    """Show confirmation prompt or raise NonInteractiveError in non-interactive mode.

    Args:
        prompt: The confirmation prompt text
        default: Default value if user just presses Enter
        error_message: Custom error message for non-interactive mode.
                      If None, a generic message is used.

    Returns:
        True if confirmed, False otherwise

    Raises:
        NonInteractiveError: If non-interactive mode is enabled
    """
    if cli_runtime.non_interactive:
        raise NonInteractiveError(
            error_message
            or "Confirmation required but --non-interactive mode is enabled. "
            "Use -y/--yes to skip confirmations."
        )
    return Confirm.ask(prompt, default=default)


class CheckboxSelector:
    """Interactive checkbox selector with optional filtering."""

    def __init__(
        self,
        items: list[tuple[str, str]],
        title: str = "Select items",
        filter_enabled: bool = True,
    ):
        """
        Initialize checkbox selector.

        Args:
            items: List of (value, label) tuples
            title: Title shown at the top
            filter_enabled: Whether to show filter input
        """
        self._items = items
        self._title = title
        self._filter_enabled = filter_enabled
        self._selected: set[str] = set()
        self._cursor = 0
        self._filter_text = ""
        self._filtered_items: list[tuple[str, str]] = list(items)
        self._result: list[str] | None = None

    def _get_filtered_items(self) -> list[tuple[str, str]]:
        """Get items filtered by current filter text."""
        if not self._filter_text:
            return list(self._items)
        filter_lower = self._filter_text.lower()
        return [
            (value, label)
            for value, label in self._items
            if filter_lower in label.lower() or filter_lower in value.lower()
        ]

    def _get_display_text(self) -> list[tuple[str, str]]:
        """Generate formatted text for display."""
        lines: list[tuple[str, str]] = []

        # Title
        lines.append(("class:title", f" {self._title}\n"))
        lines.append(("", "─" * 60 + "\n"))

        # Filter input (if enabled)
        if self._filter_enabled:
            lines.append(("class:label", " Filter: "))
            lines.append(("class:filter", self._filter_text))
            lines.append(("class:cursor", "█\n"))
            lines.append(("", "─" * 60 + "\n"))

        # Items
        self._filtered_items = self._get_filtered_items()
        if not self._filtered_items:
            lines.append(("class:dim", " No matching items\n"))
        else:
            for i, (value, label) in enumerate(self._filtered_items):
                is_selected = value in self._selected
                is_cursor = i == self._cursor

                # Checkbox
                checkbox = "[x]" if is_selected else "[ ]"
                style = "class:selected" if is_selected else ""

                if is_cursor:
                    lines.append(("class:cursor-line", f" > {checkbox} {label}\n"))
                else:
                    lines.append((style, f"   {checkbox} {label}\n"))

        # Help
        lines.append(("", "─" * 60 + "\n"))
        lines.append(("class:dim", " Space: toggle | Enter: confirm | Ctrl+C: cancel\n"))

        return lines

    async def run(self) -> list[str]:
        """Run the interactive selector and return selected values."""
        kb = KeyBindings()

        @kb.add("up")
        def _(event):
            if self._cursor > 0:
                self._cursor -= 1

        @kb.add("down")
        def _(event):
            if self._cursor < len(self._filtered_items) - 1:
                self._cursor += 1

        @kb.add(" ")
        def _(event):
            if self._filtered_items:
                value = self._filtered_items[self._cursor][0]
                if value in self._selected:
                    self._selected.discard(value)
                else:
                    self._selected.add(value)

        @kb.add("enter")
        def _(event):
            self._result = list(self._selected)
            event.app.exit()

        @kb.add("c-c")
        def _(event):
            self._result = []
            event.app.exit()

        @kb.add("backspace")
        def _(event):
            if self._filter_enabled and self._filter_text:
                self._filter_text = self._filter_text[:-1]
                self._cursor = 0

        @kb.add("<any>")
        def _(event):
            if self._filter_enabled:
                char = event.data
                if char.isprintable() and len(char) == 1:
                    self._filter_text += char
                    self._cursor = 0

        style = Style.from_dict(
            {
                "title": "bold cyan",
                "label": "bold",
                "filter": "fg:yellow",
                "cursor": "fg:yellow",
                "cursor-line": "bg:#333333 bold",
                "selected": "fg:green",
                "dim": "fg:#888888",
            }
        )

        def get_formatted_text():
            return self._get_display_text()

        layout = Layout(Window(content=FormattedTextControl(get_formatted_text), wrap_lines=True))

        app: Application[None] = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
        )

        await app.run_async()
        return self._result or []


class RadioSelector:
    """Interactive radio selector for single item selection with optional filtering."""

    def __init__(
        self,
        items: list[tuple[str, str]],
        title: str = "Select item",
        filter_enabled: bool = False,
    ):
        """
        Initialize radio selector.

        Args:
            items: List of (value, label) tuples
            title: Title shown at the top
            filter_enabled: Whether to show filter input
        """
        self._items = items
        self._title = title
        self._filter_enabled = filter_enabled
        self._cursor = 0
        self._filter_text = ""
        self._filtered_items: list[tuple[str, str]] = list(items)
        self._result: str | None = None
        self._cancelled = False

    def _get_filtered_items(self) -> list[tuple[str, str]]:
        """Get items filtered by current filter text."""
        if not self._filter_text:
            return list(self._items)
        filter_lower = self._filter_text.lower()
        return [
            (value, label)
            for value, label in self._items
            if filter_lower in label.lower() or filter_lower in value.lower()
        ]

    def _get_display_text(self) -> list[tuple[str, str]]:
        """Generate formatted text for display."""
        lines: list[tuple[str, str]] = []

        # Title
        lines.append(("class:title", f" {self._title}\n"))
        lines.append(("", "─" * 60 + "\n"))

        # Filter input (if enabled)
        if self._filter_enabled:
            lines.append(("class:label", " Filter: "))
            lines.append(("class:filter", self._filter_text))
            lines.append(("class:cursor", "█\n"))
            lines.append(("", "─" * 60 + "\n"))

        # Items
        self._filtered_items = self._get_filtered_items()
        if not self._filtered_items:
            lines.append(("class:dim", " No matching items\n"))
        else:
            for i, (_, label) in enumerate(self._filtered_items):
                is_cursor = i == self._cursor
                radio = "(●)" if is_cursor else "( )"

                if is_cursor:
                    lines.append(("class:cursor-line", f" > {radio} {label}\n"))
                else:
                    lines.append(("", f"   {radio} {label}\n"))

        # Help
        lines.append(("", "─" * 60 + "\n"))
        lines.append(("class:dim", " Enter: select | Ctrl+C: cancel\n"))

        return lines

    async def run(self) -> str | None:
        """Run the interactive selector and return selected value."""
        kb = KeyBindings()

        @kb.add("up")
        def _(event):
            if self._cursor > 0:
                self._cursor -= 1

        @kb.add("down")
        def _(event):
            if self._cursor < len(self._filtered_items) - 1:
                self._cursor += 1

        @kb.add("enter")
        def _(event):
            if self._filtered_items:
                self._result = self._filtered_items[self._cursor][0]
            event.app.exit()

        @kb.add("c-c")
        def _(event):
            self._cancelled = True
            event.app.exit()

        @kb.add("backspace")
        def _(event):
            if self._filter_enabled and self._filter_text:
                self._filter_text = self._filter_text[:-1]
                self._cursor = 0

        @kb.add("<any>")
        def _(event):
            if self._filter_enabled:
                char = event.data
                if char.isprintable() and len(char) == 1:
                    self._filter_text += char
                    self._cursor = 0

        style = Style.from_dict(
            {
                "title": "bold cyan",
                "label": "bold",
                "filter": "fg:yellow",
                "cursor": "fg:yellow",
                "cursor-line": "bg:#333333 bold",
                "dim": "fg:#888888",
            }
        )

        def get_formatted_text():
            return self._get_display_text()

        layout = Layout(Window(content=FormattedTextControl(get_formatted_text), wrap_lines=True))

        app: Application[None] = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
        )

        await app.run_async()
        return None if self._cancelled else self._result


async def select_item_interactive(
    items: list[tuple[str, str]],
    title: str = "Select item",
    filter_enabled: bool = False,
) -> str | None:
    """
    Show interactive single-item selection.

    Args:
        items: List of (value, label) tuples
        title: Title shown at the top
        filter_enabled: Whether to show filter input for searching

    Returns:
        Selected value, or None if cancelled

    Raises:
        NonInteractiveError: If non-interactive mode is enabled
    """
    if cli_runtime.non_interactive:
        raise NonInteractiveError(
            "Interactive selection required but --non-interactive mode is enabled. "
            "Please provide the required parameter."
        )
    selector = RadioSelector(items, title, filter_enabled)
    return await selector.run()


async def select_items_interactive(
    items: list[tuple[str, str]],
    title: str = "Select items",
    filter_enabled: bool = True,
) -> list[str]:
    """
    Show interactive checkbox selection.

    Args:
        items: List of (value, label) tuples
        title: Title shown at the top
        filter_enabled: Whether to show filter input for searching

    Returns:
        List of selected values (empty if cancelled)

    Raises:
        NonInteractiveError: If non-interactive mode is enabled
    """
    if cli_runtime.non_interactive:
        raise NonInteractiveError(
            "Interactive selection required but --non-interactive mode is enabled. "
            "Please provide the required parameter."
        )
    selector = CheckboxSelector(items, title, filter_enabled)
    return await selector.run()


async def select_clients_interactive(available_clients: list[str]) -> set[str] | None:
    """
    Interactive client selection with "all" option.

    Args:
        available_clients: List of available client names

    Returns:
        Set of selected client IDs, or None if "all" selected.
        Empty set if cancelled.

    Raises:
        NonInteractiveError: If non-interactive mode is enabled
    """
    if cli_runtime.non_interactive:
        raise NonInteractiveError(
            "Interactive client selection required but --non-interactive mode is enabled.\n"
            "  Use --client-id to specify clients.\n"
            "  Usage: statgpt content init --client-id <client1,client2,...>"
        )
    items = [("__all__", "All clients")] + [(c, c) for c in sorted(available_clients)]

    selected = await select_items_interactive(
        items,
        title="Select Clients",
        filter_enabled=False,
    )

    if not selected:
        return set()  # Cancelled

    if "__all__" in selected:
        return None  # All clients

    return set(selected)


async def select_datasets_interactive(
    datasets: list[tuple[str, str]],
) -> set[str]:
    """
    Interactive dataset selection with filtering.

    Args:
        datasets: List of (urn, display_label) tuples

    Returns:
        Set of selected dataset URNs (empty if cancelled)

    Raises:
        NonInteractiveError: If non-interactive mode is enabled
    """
    if cli_runtime.non_interactive:
        raise NonInteractiveError(
            "Interactive dataset selection required but --non-interactive mode is enabled.\n"
            "  Use --datasets to specify datasets.\n"
            "  Usage: statgpt content init --datasets <urn1,urn2,...>"
        )
    selected = await select_items_interactive(
        datasets,
        title="Select Datasets (type to filter)",
        filter_enabled=True,
    )
    return set(selected)
