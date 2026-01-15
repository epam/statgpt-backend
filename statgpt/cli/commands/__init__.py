"""CLI commands registry."""

from importlib.metadata import version

from statgpt.cli.commands.auth import auth_group
from statgpt.cli.commands.base import Command, CommandGroup, CommandRegistry
from statgpt.cli.commands.channel import channel_group
from statgpt.cli.commands.content import content_group
from statgpt.cli.commands.settings import settings_command

__all__ = [
    "Command",
    "CommandGroup",
    "CommandRegistry",
    "auth_group",
    "channel_group",
    "content_group",
    "settings_command",
    "create_registry",
]


def _get_version() -> str:
    """Get CLI version from package metadata."""
    try:
        return version("statgpt")
    except Exception:
        return "unknown"


def create_registry() -> CommandRegistry:
    """Create and populate the command registry."""
    registry = CommandRegistry(version=_get_version())

    # Register standalone commands
    registry.register_command(settings_command)

    # Register command groups
    registry.register_group(auth_group)
    registry.register_group(channel_group)
    registry.register_group(content_group)

    return registry
