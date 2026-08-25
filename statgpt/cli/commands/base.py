"""Base command infrastructure for CLI."""

import shlex
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from statgpt.cli.shared.batch_report import BatchPartialFailureError
from statgpt.cli.shared.console import console, print_error


@dataclass
class CommandArg:
    """Definition of a command argument."""

    name: str
    description: str
    short_name: str | None = None
    required: bool = False
    default: Any = None
    choices: list[str] | None = None
    is_flag: bool = False


@dataclass
class Command:
    """Represents a CLI command."""

    name: str
    description: str
    handler: Callable[..., Coroutine[Any, Any, None]]
    args: list[CommandArg] = field(default_factory=list)
    group: str | None = None

    @property
    def full_name(self) -> str:
        """Get full command name including group."""
        if self.group:
            return f"{self.group} {self.name}"
        return self.name

    def get_help(self) -> str:
        """Generate help text for this command."""
        lines = [f"[bold cyan]{self.full_name}[/bold cyan] - {self.description}"]

        if self.args:
            lines.append("\n[bold]Arguments:[/bold]")
            for arg in self.args:
                if arg.short_name:
                    arg_str = f"  -{arg.short_name}, --{arg.name}"
                else:
                    arg_str = f"  --{arg.name}"
                if arg.choices:
                    arg_str += f" [{'/'.join(arg.choices)}]"
                if arg.required:
                    arg_str += " [red](required)[/red]"
                elif arg.default is not None:
                    arg_str += f" [dim](default: {arg.default})[/dim]"
                arg_str += f"\n      {arg.description}"
                lines.append(arg_str)

        return "\n".join(lines)

    def parse_args(self, args_str: str) -> dict[str, Any]:
        """Parse command arguments from string.

        Args:
            args_str: Command arguments as string

        Returns:
            Dictionary of parsed arguments

        Raises:
            ValueError: If required argument is missing or invalid
        """
        result: dict[str, Any] = {}

        # Set defaults
        for arg in self.args:
            if arg.is_flag:
                result[arg.name.replace("-", "_")] = False
            elif arg.default is not None:
                result[arg.name.replace("-", "_")] = arg.default

        if not args_str.strip():
            # Check required args
            self._validate_required(result)
            return result

        # Parse args
        try:
            tokens = shlex.split(args_str)
        except ValueError as e:
            raise ValueError(f"Invalid argument syntax: {e}")

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.startswith("--"):
                arg_name = token[2:]
                arg_def = self._find_arg(arg_name)

                if arg_def is None:
                    raise ValueError(f"Unknown argument: --{arg_name}")

                key = arg_def.name.replace("-", "_")

                if arg_def.is_flag:
                    result[key] = True
                    i += 1
                else:
                    if i + 1 >= len(tokens):
                        raise ValueError(f"Argument --{arg_name} requires a value")
                    value = tokens[i + 1]

                    if arg_def.choices and value not in arg_def.choices:
                        raise ValueError(
                            f"Invalid value for --{arg_name}: {value}. "
                            f"Must be one of: {', '.join(arg_def.choices)}"
                        )

                    result[key] = value
                    i += 2
            elif token.startswith("-") and len(token) == 2:
                # Short argument (e.g., -c)
                short_name = token[1:]
                arg_def = self._find_arg_by_short(short_name)

                if arg_def is None:
                    raise ValueError(f"Unknown argument: -{short_name}")

                key = arg_def.name.replace("-", "_")

                if arg_def.is_flag:
                    result[key] = True
                    i += 1
                else:
                    if i + 1 >= len(tokens):
                        raise ValueError(f"Argument -{short_name} requires a value")
                    value = tokens[i + 1]

                    if arg_def.choices and value not in arg_def.choices:
                        raise ValueError(
                            f"Invalid value for -{short_name}: {value}. "
                            f"Must be one of: {', '.join(arg_def.choices)}"
                        )

                    result[key] = value
                    i += 2
            else:
                raise ValueError(f"Unexpected token: {token}")

        self._validate_required(result)
        return result

    def _find_arg(self, name: str) -> CommandArg | None:
        """Find argument definition by name."""
        for arg in self.args:
            if arg.name == name or arg.name.replace("-", "_") == name.replace("-", "_"):
                return arg
        return None

    def _find_arg_by_short(self, short_name: str) -> CommandArg | None:
        """Find argument definition by short name."""
        for arg in self.args:
            if arg.short_name == short_name:
                return arg
        return None

    def _validate_required(self, parsed: dict[str, Any]) -> None:
        """Validate that all required arguments are present."""
        for arg in self.args:
            key = arg.name.replace("-", "_")
            if arg.required and key not in parsed:
                raise ValueError(f"Missing required argument: --{arg.name}")

    async def execute(self, args_str: str = "") -> None:
        """Execute the command with given arguments."""
        try:
            parsed_args = self.parse_args(args_str)
            await self.handler(**parsed_args)
        except ValueError as e:
            print_error(str(e))
            console.print(f"\n{self.get_help()}")
        except BatchPartialFailureError:
            # The batch summary already lists what failed; re-raise quietly so the entry
            # point can set a non-zero exit code without repeating it.
            raise
        except Exception as e:
            print_error(f"Command failed: {e}")
            raise


@dataclass
class CommandGroup:
    """Represents a group of related commands."""

    name: str
    description: str
    commands: dict[str, Command] = field(default_factory=dict)

    def add_command(self, command: Command) -> None:
        """Add a command to this group."""
        command.group = self.name
        self.commands[command.name] = command

    def get_help(self) -> str:
        """Generate help text for this group."""
        lines = [f"[bold cyan]{self.name}[/bold cyan] - {self.description}\n"]
        lines.append("[bold]Commands:[/bold]")
        for cmd in self.commands.values():
            lines.append(f"  {cmd.name:<15} {cmd.description}")
        return "\n".join(lines)


class CommandRegistry:
    """Registry for all CLI commands."""

    def __init__(self, version: str = "unknown"):
        self._commands: dict[str, Command] = {}
        self._groups: dict[str, CommandGroup] = {}
        self._version = version

    @property
    def version(self) -> str:
        """Get CLI version."""
        return self._version

    def register_command(self, command: Command) -> None:
        """Register a standalone command."""
        self._commands[command.name] = command

    def register_group(self, group: CommandGroup) -> None:
        """Register a command group."""
        self._groups[group.name] = group
        for cmd in group.commands.values():
            cmd.group = group.name

    def get_command(self, name: str) -> Command | None:
        """Get a command by name (supports 'group command' syntax)."""
        parts = name.strip().split(maxsplit=1)

        if len(parts) == 1:
            # Standalone command
            return self._commands.get(parts[0])
        else:
            # Group command
            group_name, cmd_name = parts
            group = self._groups.get(group_name)
            if group:
                return group.commands.get(cmd_name)
        return None

    def get_all_command_names(self) -> list[str]:
        """Get all command names for autocomplete."""
        names = list(self._commands.keys())
        for group in self._groups.values():
            names.append(group.name)
            for cmd in group.commands.values():
                names.append(f"{group.name} {cmd.name}")
        return sorted(names)

    def get_completions(self, text: str) -> list[tuple[str, str]]:
        """Get completions for partial input.

        Returns:
            List of (completion, description) tuples
        """
        # Check if text ends with space (user wants subcommand completions)
        ends_with_space = text.endswith(" ")
        text = text.strip().lower()
        completions = []

        if not text:
            # Show all top-level commands and groups
            for cmd in self._commands.values():
                completions.append((cmd.name, cmd.description))
            for group in self._groups.values():
                completions.append((group.name, group.description))
            return completions

        parts = text.split(maxsplit=1)

        if len(parts) == 1:
            # Check if this is a complete group name followed by space
            if ends_with_space and text in self._groups:
                # Show all subcommands of this group
                group = self._groups[text]
                for cmd in group.commands.values():
                    completions.append((cmd.name, cmd.description))
                return completions

            # Complete command or group name
            for cmd in self._commands.values():
                if cmd.name.lower().startswith(text):
                    completions.append((cmd.name, cmd.description))
            for group in self._groups.values():
                if group.name.lower().startswith(text):
                    completions.append((group.name, group.description))
                # Also show group commands if typing matches
                for cmd in group.commands.values():
                    full_name = f"{group.name} {cmd.name}"
                    if full_name.lower().startswith(text):
                        completions.append((full_name, cmd.description))
        else:
            # Complete subcommand within group
            group_name, partial_cmd = parts

            matched_group = self._groups.get(group_name)
            if matched_group:
                for cmd in matched_group.commands.values():
                    if cmd.name.lower().startswith(partial_cmd.lower()):
                        completions.append((f"{group_name} {cmd.name}", cmd.description))

        return completions

    def get_help(self) -> str:
        """Generate help text for all commands."""
        lines = [
            f"[bold cyan]StatGPT CLI[/bold cyan] [dim]v{self._version}[/dim]\n",
            "[bold]Built-in:[/bold]",
            "  help                 Show this help message",
            "  help <command>       Show help for a specific command",
            "  clear                Clear the screen",
            "  version              Show CLI version",
            "  exit                 Exit the CLI",
            "",
            "[bold]Commands:[/bold]",
        ]

        if self._commands:
            for cmd in self._commands.values():
                lines.append(f"  {cmd.name:<20} {cmd.description}")

        if self._groups:
            lines.append("")
            for group in self._groups.values():
                lines.append(f"  [bold cyan]{group.name}[/bold cyan]")
                for cmd in group.commands.values():
                    lines.append(f"    {cmd.name:<18} {cmd.description}")

        lines.append("\n[dim]Type 'help <command>' for detailed help.[/dim]")
        return "\n".join(lines)

    def get_group(self, name: str) -> CommandGroup | None:
        """Get a command group by name."""
        return self._groups.get(name)

    async def execute(self, input_str: str) -> bool:
        """Execute a command from input string.

        Returns:
            True if command was found and executed, False otherwise
        """
        input_str = input_str.strip()
        if not input_str:
            return False

        # Split into command and args
        parts = input_str.split(maxsplit=2)
        first_part = parts[0]

        # Check if first part is a group name
        if first_part in self._groups:
            group = self._groups[first_part]

            if len(parts) == 1:
                # Just group name, show group help
                console.print(group.get_help())
                return True

            # Group command (e.g., "content reindex --arg value")
            cmd_name = parts[1]
            args_str = parts[2] if len(parts) > 2 else ""

            command = group.commands.get(cmd_name)
            if command:
                await command.execute(args_str)
                return True
            else:
                # Unknown subcommand
                print_error(f"Unknown command: {first_part} {cmd_name}")
                console.print(group.get_help())
                return True

        # Check for standalone command
        args_str = " ".join(parts[1:]) if len(parts) > 1 else ""

        command = self._commands.get(first_part)
        if command:
            await command.execute(args_str)
            return True

        return False
