"""StatGPT CLI - Interactive command-line interface for StatGPT administration."""

import asyncio
import sys

from statgpt.cli.commands import create_registry
from statgpt.cli.commands.base import CommandRegistry
from statgpt.cli.repl import run_repl
from statgpt.cli.settings import cli_runtime
from statgpt.cli.shared.batch_report import BatchPartialFailureError
from statgpt.cli.shared.console import console, print_error
from statgpt.cli.shared.logging import setup_logging


def _init_runtime() -> list[str]:
    """Parse global flags from sys.argv and initialize cli_runtime.

    Returns:
        Command arguments with global flags removed.
    """
    command_args = []

    for arg in sys.argv[1:]:
        if arg == "--non-interactive":
            cli_runtime.non_interactive = True
        elif arg == "--debug":
            cli_runtime.debug = True
        else:
            command_args.append(arg)

    return command_args


async def _execute_direct(registry: CommandRegistry, args: list[str]) -> int:
    """Execute a command directly from command-line arguments.

    Returns 0 for success, 1 for error.
    """
    command_str = " ".join(args)

    if args[0] == "help":
        if len(args) == 1:
            console.print(registry.get_help())
        else:
            cmd_name = " ".join(args[1:])
            command = registry.get_command(cmd_name)
            if command:
                console.print(command.get_help())
            else:
                group = registry.get_group(args[1])
                if group:
                    console.print(group.get_help())
                else:
                    print_error(f"Unknown command: {cmd_name}")
                    return 1
        return 0

    if args[0] == "version":
        console.print(f"StatGPT CLI v{registry.version}")
        return 0

    try:
        found = await registry.execute(command_str)
        if not found:
            print_error(f"Unknown command: {args[0]}")
            console.print("[dim]Run 'statgpt help' for available commands.[/dim]")
            return 1
        return 0
    except BatchPartialFailureError:
        # Some items in a batch failed. The summary is already on screen; report it through
        # the exit code so a pipeline cannot read a truncated run as a complete one.
        return 1
    except Exception as e:
        print_error(f"Command failed: {e}")
        if cli_runtime.debug:
            raise
        return 1


def main() -> None:
    """Main entry point for the StatGPT CLI."""
    logger = setup_logging()
    logger.info("StatGPT CLI starting")

    try:
        registry = create_registry()
        command_args = _init_runtime()

        if command_args:
            exit_code = asyncio.run(_execute_direct(registry, command_args))
            sys.exit(exit_code)

        asyncio.run(run_repl(registry))
    except KeyboardInterrupt:
        logger.info("CLI interrupted by user")
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        logger.exception("CLI crashed with error")
        print(f"Error: {e}", file=sys.stderr)
        if cli_runtime.debug:
            raise
        sys.exit(1)


__all__ = ["main"]
