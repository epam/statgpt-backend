"""Settings command for displaying CLI configuration."""

from pydantic.fields import FieldInfo
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from statgpt.cli.commands.base import Command
from statgpt.cli.settings import CLISettings, FieldMeta, SettingsSection, cli_settings
from statgpt.cli.shared.auth import get_available_providers
from statgpt.cli.shared.console import console, mask_secret

PROVIDER_SECTIONS = {
    SettingsSection.AZURE: "azure",
    SettingsSection.KEYCLOAK: "keycloak",
}


def _get_field_meta(field_info: FieldInfo) -> FieldMeta | None:
    """Parse field metadata from json_schema_extra."""
    extra = field_info.json_schema_extra
    if isinstance(extra, dict):
        try:
            return FieldMeta.model_validate(extra)
        except Exception:
            return None
    return None


def _create_settings_section(title: str) -> Table:
    """Create a table for a settings section."""
    table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    table.add_column("Setting", style="cyan", width=30)
    table.add_column("Value", style="white", ratio=2)
    table.add_column("Source", style="dim", justify="right", width=15)
    return table


def _format_source(source: str) -> str:
    """Format source indicator."""
    if source == "env":
        return "[green](env)[/green]"
    elif source == "default":
        return "[dim](default)[/dim]"
    else:
        return f"[dim]({source})[/dim]"


def _format_value(name: str, value: str | int | None, meta: FieldMeta) -> str:
    """Format a setting value for display."""
    if value is None:
        if name == "max_embeddings":
            return "[dim]\u2717 not set (unlimited)[/dim]"
        return "[dim]\u2717 not set[/dim]"
    if meta.secret:
        return mask_secret(str(value))
    return str(value)


def _get_display_value(name: str, meta: FieldMeta) -> tuple[str, str]:
    """Get display value and source for a field, handling special cases."""
    value = getattr(cli_settings, name)
    source = cli_settings.get_setting_source(name)

    if name == "auth_provider":
        providers = get_available_providers()
        formatted = f"{value} [dim](available: {', '.join(providers)})[/dim]"
        return formatted, source

    if name == "data_dir":
        display_value = cli_settings.cli_data_dir
        if source == "not set":
            source = "default"
        return display_value, source

    return _format_value(name, value, meta), source


def _group_settings_by_section() -> dict[SettingsSection, list[tuple[str, FieldMeta]]]:
    """Group settings by their section annotation."""
    sections: dict[SettingsSection, list[tuple[str, FieldMeta]]] = {}
    for name, field_info in CLISettings.model_fields.items():
        meta = _get_field_meta(field_info)
        if meta:
            sections.setdefault(meta.section, []).append((name, meta))
    return sections


def _should_show_section(section: SettingsSection) -> bool:
    """Check if a provider section should be shown based on current auth_provider."""
    if section not in PROVIDER_SECTIONS:
        return True
    return cli_settings.auth_provider == PROVIDER_SECTIONS[section]


async def settings_handler() -> None:
    """Display current CLI settings."""
    sections_data = _group_settings_by_section()
    render_items: list[Text | Table] = []

    for section in SettingsSection:
        if section not in sections_data:
            continue
        if not _should_show_section(section):
            continue

        fields = sections_data[section]
        table = _create_settings_section(section.value)

        for field_name, meta in fields:
            display_value, source = _get_display_value(field_name, meta)
            table.add_row(field_name, display_value, _format_source(source))

        indent = "  " if section in PROVIDER_SECTIONS else ""
        if render_items:
            render_items.append(Text(f"\n{indent}{section.value}", style="bold white"))
        else:
            render_items.append(Text(f"{indent}{section.value}", style="bold white"))
        render_items.append(table)

    panel = Panel(
        Group(*render_items),
        title="[bold cyan]CLI Settings[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )

    console.print(panel)
    console.print(
        "\n[dim]Tip: Set variables with STATGPT_CLI_ prefix, "
        "e.g. STATGPT_CLI_AUTH_AZURE_CLIENT_ID[/dim]"
    )


settings_command = Command(
    name="settings",
    description="Show current CLI settings and their sources",
    handler=settings_handler,
)
