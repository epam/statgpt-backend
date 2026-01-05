"""Settings command for displaying CLI configuration."""

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from statgpt.cli.commands.base import Command
from statgpt.cli.shared.auth import get_available_providers
from statgpt.cli.shared.console import console, mask_secret
from statgpt.cli.shared.settings import cli_settings


def _create_settings_section(title: str) -> Table:
    """Create a table for a settings section."""
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
        expand=True,
    )
    table.add_column("Setting", style="cyan", width=30)
    table.add_column("Value", style="white", ratio=2)
    table.add_column("Source", style="dim", justify="right", width=15)
    return table


def _format_value(value: str | int | None, is_secret: bool = False) -> str:
    """Format a setting value for display."""
    if value is None:
        return "[dim]\u2717 not set[/dim]"
    if is_secret:
        return mask_secret(str(value))
    return str(value)


def _format_source(source: str) -> str:
    """Format source indicator."""
    if source == "env":
        return "[green](env)[/green]"
    elif source == "default":
        return "[dim](default)[/dim]"
    elif source == "fallback: app":
        return "[yellow](fallback: app)[/yellow]"
    else:
        return f"[dim]({source})[/dim]"


async def settings_handler() -> None:
    """Display current CLI settings."""
    # Admin API section
    admin_table = _create_settings_section("Admin API")
    admin_table.add_row(
        "admin_url",
        _format_value(cli_settings.admin_url),
        _format_source(cli_settings.get_setting_source("admin_url")),
    )

    # Auth section - provider selection
    auth_table = _create_settings_section("Authentication")
    providers = get_available_providers()
    provider_value = f"{cli_settings.auth_provider} [dim](available: {', '.join(providers)})[/dim]"
    auth_table.add_row(
        "auth_provider",
        provider_value,
        _format_source(cli_settings.get_setting_source("auth_provider")),
    )

    # Auth section - Azure Entra ID settings
    azure_table = _create_settings_section("Azure Entra ID")
    azure_table.add_row(
        "auth_azure_client_id",
        _format_value(cli_settings.auth_azure_client_id),
        _format_source(cli_settings.get_setting_source("auth_azure_client_id")),
    )
    azure_table.add_row(
        "auth_azure_authority",
        _format_value(cli_settings.auth_azure_authority),
        _format_source(cli_settings.get_setting_source("auth_azure_authority")),
    )
    azure_table.add_row(
        "auth_azure_scope",
        _format_value(cli_settings.auth_azure_scope),
        _format_source(cli_settings.get_setting_source("auth_azure_scope")),
    )
    azure_table.add_row(
        "auth_azure_client_secret",
        _format_value(cli_settings.auth_azure_client_secret, is_secret=True),
        _format_source(cli_settings.get_setting_source("auth_azure_client_secret")),
    )
    azure_table.add_row(
        "auth_azure_username",
        _format_value(cli_settings.auth_azure_username),
        _format_source(cli_settings.get_setting_source("auth_azure_username")),
    )
    azure_table.add_row(
        "auth_azure_password",
        _format_value(cli_settings.auth_azure_password, is_secret=True),
        _format_source(cli_settings.get_setting_source("auth_azure_password")),
    )

    # Content section
    content_table = _create_settings_section("Content")
    content_table.add_row(
        "config_dir",
        _format_value(cli_settings.config_dir),
        _format_source(cli_settings.get_setting_source("config_dir")),
    )
    max_emb_value = (
        str(cli_settings.max_embeddings)
        if cli_settings.max_embeddings
        else "[dim]\u2717 not set (unlimited)[/dim]"
    )
    content_table.add_row(
        "max_embeddings",
        max_emb_value,
        _format_source(cli_settings.get_setting_source("max_embeddings")),
    )

    # DIAL section
    dial_table = _create_settings_section("DIAL")
    dial_table.add_row(
        "dial_url",
        _format_value(cli_settings.dial_url),
        _format_source(cli_settings.get_setting_source("dial_url")),
    )
    dial_table.add_row(
        "dial_api_key",
        _format_value(cli_settings.dial_api_key, is_secret=True),
        _format_source(cli_settings.get_setting_source("dial_api_key")),
    )

    # General section
    general_table = _create_settings_section("General")
    general_table.add_row(
        "log_level",
        _format_value(cli_settings.log_level),
        _format_source(cli_settings.get_setting_source("log_level")),
    )
    data_dir_value = cli_settings.cli_data_dir
    data_dir_source = cli_settings.get_setting_source("data_dir")
    if data_dir_source == "not set":
        data_dir_source = "default"  # Show as default since we have a fallback
    general_table.add_row(
        "data_dir",
        _format_value(data_dir_value),
        _format_source(data_dir_source),
    )

    # Render all sections
    sections = Group(
        Text("Admin API", style="bold white"),
        admin_table,
        Text("\nAuthentication", style="bold white"),
        auth_table,
        Text("\n  Azure Entra ID", style="bold white"),
        azure_table,
        Text("\nContent", style="bold white"),
        content_table,
        Text("\nDIAL", style="bold white"),
        dial_table,
        Text("\nGeneral", style="bold white"),
        general_table,
    )

    panel = Panel(
        sections,
        title="[bold cyan]CLI Settings[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )

    console.print(panel)
    console.print(
        "\n[dim]Tip: Set variables with STATGPT_CLI_ prefix, "
        "e.g. STATGPT_CLI_AUTH_AZURE_CLIENT_ID[/dim]"
    )


# Command definition
settings_command = Command(
    name="settings",
    description="Show current CLI settings and their sources",
    handler=settings_handler,
)
