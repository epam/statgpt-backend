"""Autocomplete support for StatGPT CLI."""

from collections.abc import Iterable

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from statgpt.cli.commands.base import CommandRegistry


class StatGPTCompleter(Completer):
    """Autocomplete provider for StatGPT CLI commands."""

    def __init__(self, registry: CommandRegistry):
        self._registry = registry

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Get completions for current input."""
        text = document.text_before_cursor
        word = document.get_word_before_cursor()

        completions = self._registry.get_completions(text)

        for name, description in completions:
            # Calculate how much of the completion is already typed
            if text.strip():
                # Find the start position for replacement
                start_pos = -len(word) if word else 0

                # For multi-word completions (group commands), adjust
                if " " in name and " " not in text:
                    # User typed part of group name, complete full command
                    start_pos = -len(text.strip())
                elif " " in text and " " in name:
                    # User typed "group partial", complete to "group command"
                    typed_parts = text.strip().split()
                    if len(typed_parts) == 2:
                        start_pos = -len(typed_parts[1])
                        name = name.split()[-1]  # Only complete the subcommand part
                elif " " not in name and text.endswith(" "):
                    # Subcommand completion after "group " - insert at cursor
                    start_pos = 0
            else:
                start_pos = 0

            yield Completion(
                name,
                start_position=start_pos,
                display_meta=description,
            )
