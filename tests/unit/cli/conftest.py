"""Shared fixtures for CLI unit tests."""

import os

import pytest

from statgpt.cli.commands.base import Command, CommandArg, CommandGroup, CommandRegistry


@pytest.fixture
def sample_command_args() -> list[CommandArg]:
    """Sample command arguments for testing."""
    return [
        CommandArg(name="name", description="Name argument", short_name="n", required=True),
        CommandArg(name="clean", description="Clean flag", is_flag=True),
        CommandArg(name="mode", description="Mode selection", choices=["a", "b"], default="a"),
    ]


@pytest.fixture
def noop_handler():
    """No-op async handler for command tests."""

    async def handler(**kwargs):
        pass

    return handler


@pytest.fixture
def sample_command(noop_handler, sample_command_args) -> Command:
    """Sample command with various argument types."""
    return Command(
        name="test",
        description="Test command",
        handler=noop_handler,
        args=sample_command_args,
    )


@pytest.fixture
def simple_command(noop_handler) -> Command:
    """Simple command with no arguments."""
    return Command(
        name="simple",
        description="Simple command",
        handler=noop_handler,
    )


@pytest.fixture
def sample_group(noop_handler) -> CommandGroup:
    """Sample command group for testing."""
    group = CommandGroup(name="test", description="Test group")
    group.add_command(Command(name="sub1", description="Sub command 1", handler=noop_handler))
    group.add_command(Command(name="sub2", description="Sub command 2", handler=noop_handler))
    return group


@pytest.fixture
def registry_with_commands(simple_command, sample_group) -> CommandRegistry:
    """Registry with sample commands and groups."""
    registry = CommandRegistry()
    registry.register_command(simple_command)
    registry.register_group(sample_group)
    return registry


@pytest.fixture
def clean_cli_env(monkeypatch):
    """Remove all STATGPT_CLI_ environment variables."""
    for key in list(os.environ.keys()):
        if key.startswith("STATGPT_CLI_"):
            monkeypatch.delenv(key, raising=False)
