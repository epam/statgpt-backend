"""Interactive REPL for StatGPT CLI."""

import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from statgpt.cli.commands.base import CommandRegistry
from statgpt.cli.completer import StatGPTCompleter
from statgpt.cli.settings import cli_runtime, cli_settings
from statgpt.cli.shared.auth import get_token_info, is_logged_in
from statgpt.cli.shared.console import console, print_banner, print_error

# Prompt style
PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "bold cyan",
        "completion-menu.completion": "bg:#333333 #ffffff",
        "completion-menu.completion.current": "bg:#00aaaa #000000",
        "completion-menu.meta.completion": "bg:#333333 #888888",
        "completion-menu.meta.completion.current": "bg:#00aaaa #000000",
    }
)


def _get_history_path() -> Path:
    """Get path to command history file."""

    history_dir = Path(cli_settings.cli_data_dir)
    history_dir.mkdir(mode=0o700, exist_ok=True)
    return history_dir / "cli_history"


def _print_auth_status() -> None:
    """Print current authentication status."""
    if is_logged_in():
        info = get_token_info()
        if info:
            mins = info["expires_in_seconds"] // 60
            provider = info.get("provider", "unknown")
            console.print(
                f"[green]✓ Authenticated[/green] [dim]({provider}, expires in {mins} min)[/dim]"
            )
    else:
        console.print(
            "[yellow]○ Not authenticated[/yellow] [dim](run 'auth login' to authenticate)[/dim]"
        )


class REPL:
    """Interactive Read-Eval-Print Loop for StatGPT CLI."""

    def __init__(self, registry: CommandRegistry):
        self._registry = registry
        self._session: PromptSession | None = None
        self._running = False

    def _setup_session(self) -> PromptSession:
        """Set up prompt session with history and autocomplete."""
        history_path = _get_history_path()
        completer = StatGPTCompleter(self._registry)

        return PromptSession(
            history=FileHistory(str(history_path)),
            completer=completer,
            style=PROMPT_STYLE,
            complete_while_typing=True,
            enable_history_search=True,
        )

    def _handle_builtin(self, command: str) -> bool:
        """Handle built-in commands.

        Returns:
            True if command was handled, False otherwise
        """
        cmd = command.strip().lower()

        if cmd in ("exit", "quit", "q"):
            self._running = False
            console.print("[dim]Goodbye![/dim]")
            return True

        if cmd == "help":
            console.print(self._registry.get_help())
            return True

        if cmd.startswith("help "):
            cmd_name = cmd[5:].strip()
            # Check for command first
            command_obj = self._registry.get_command(cmd_name)
            if command_obj:
                console.print(command_obj.get_help())
                return True
            # Check for group
            group_obj = self._registry.get_group(cmd_name)
            if group_obj:
                console.print(group_obj.get_help())
                return True
            print_error(f"Unknown command: {cmd_name}")
            return True

        if cmd == "clear":
            console.clear()
            return True

        if cmd == "version":
            console.print(f"StatGPT CLI v{self._registry.version}")
            return True

        return False

    async def run(self) -> None:
        """Run the interactive REPL."""
        self._session = self._setup_session()
        self._running = True

        print_banner(self._registry.version)
        _print_auth_status()
        console.print()

        while self._running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._session.prompt("statgpt> "),  # type: ignore[union-attr]
                )

                if not user_input.strip():
                    continue

                if self._handle_builtin(user_input):
                    continue

                found = await self._registry.execute(user_input)
                if not found:
                    print_error(f"Unknown command: {user_input.split()[0]}")
                    console.print("[dim]Type 'help' for available commands.[/dim]")

            except KeyboardInterrupt:
                console.print("\n[dim]Use 'exit' to quit.[/dim]")
                continue
            except EOFError:
                self._running = False
                console.print("\n[dim]Goodbye![/dim]")
                break
            except Exception as e:
                print_error(f"Error: {e}")
                if cli_runtime.debug:
                    console.print_exception()


async def run_repl(registry: CommandRegistry) -> None:
    """Create and run the REPL."""
    repl = REPL(registry)
    await repl.run()
