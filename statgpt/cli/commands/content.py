"""Content management commands for StatGPT CLI."""

import os
from typing import Any

from rich.panel import Panel
from rich.prompt import Confirm

from statgpt.cli.commands.base import Command, CommandArg, CommandGroup
from statgpt.cli.shared import (
    AdminClient,
    cli_settings,
    console,
    get_admin_client,
    print_error,
    print_info,
    print_success,
    print_warning,
    select_clients_interactive,
    select_datasets_interactive,
)
from statgpt.common import utils
from statgpt.common.schemas import (
    Channel,
    ChannelDatasetExpanded,
    DataSet,
    DataSource,
    GlossaryTerm,
)
from statgpt.common.utils import dial_core_factory

VALID_COMPONENTS = {"channels", "datasources", "datasets", "glossaries", "files"}


def _parse_components(only: str | None) -> set[str]:
    """Parse and validate component list from --only flag."""
    if not only:
        return VALID_COMPONENTS.copy()

    components = {c.strip().lower() for c in only.split(",") if c.strip()}

    invalid = components - VALID_COMPONENTS
    if invalid:
        raise ValueError(
            f"Invalid components: {', '.join(invalid)}. "
            f"Valid: {', '.join(sorted(VALID_COMPONENTS))}"
        )

    # datasets implies datasources
    if "datasets" in components:
        components.add("datasources")

    return components


def _confirm_full_init(clients: list[str], components: set[str]) -> bool:
    """Show confirmation dialog for full content initialization."""
    component_lines = []
    if "files" in components:
        component_lines.append("  • Files (DIAL uploads)")
    if "channels" in components:
        component_lines.append("  • Channels")
    if "glossaries" in components:
        component_lines.append("  • Glossaries")
    if "datasources" in components:
        component_lines.append("  • Data sources")
    if "datasets" in components:
        component_lines.append("  • Datasets")

    content = (
        "[bold]You are about to initialize:[/bold]\n\n"
        f"{chr(10).join(component_lines)}\n\n"
        f"[bold]Clients:[/bold] {', '.join(clients)}"
    )
    console.print(Panel(content, title="Content Initialization", border_style="yellow"))
    return Confirm.ask("Proceed?", default=False)


def _load_available_datasets(client_config_dir: str) -> list[tuple[str, str]]:
    """Load dataset URNs from config files."""
    datasets_dir = os.path.join(client_config_dir, "datasets")
    if not os.path.exists(datasets_dir):
        return []

    datasets: list[tuple[str, str]] = []
    for filename in os.listdir(datasets_dir):
        if not filename.endswith(".yaml"):
            continue
        cfg = utils.read_yaml(os.path.join(datasets_dir, filename))
        for ds in cfg.get("dataSets", []):
            urn = ds.get("details", {}).get("urn", "")
            title = ds.get("title", urn)
            if urn:
                datasets.append((urn, f"{title}"))
    return sorted(datasets, key=lambda x: x[1])


async def init_handler(
    client_id: str | None = None,
    datasets: str | None = None,
    only: str | None = None,
    yes: bool = False,
) -> None:
    """Initialize content (channels, data sources, datasets, glossaries)."""
    config_dir = cli_settings.config_dir
    if not config_dir:
        print_error("STATGPT_CLI_CONFIG_DIR environment variable is not set")
        return

    if not os.path.exists(config_dir):
        print_error(f"Config directory not found: {config_dir}")
        return

    # Parse components
    try:
        components = _parse_components(only)
    except ValueError as e:
        print_error(str(e))
        return

    # Find available clients
    clients_dir = os.path.join(config_dir, "clients")
    if not os.path.exists(clients_dir):
        print_error(f"Clients directory not found: {clients_dir}")
        return

    all_available_clients = [
        d for d in os.listdir(clients_dir) if os.path.isdir(os.path.join(clients_dir, d))
    ]

    if not all_available_clients:
        print_error(f"No clients found in {clients_dir}")
        return

    # Client selection
    client_ids: set[str] | None = None
    if client_id:
        # Parse from CLI argument
        client_ids = {cid.strip() for cid in client_id.split(",") if cid.strip()}
    elif not yes:
        # Interactive selection
        client_ids = await select_clients_interactive(all_available_clients)
        if client_ids is not None and len(client_ids) == 0:
            print_info("No clients selected. Aborted.")
            return

    # Filter clients by selection
    if client_ids is None:
        available_clients = all_available_clients
    else:
        available_clients = [c for c in all_available_clients if c in client_ids]
        if not available_clients:
            print_error("No matching clients found")
            return

    # Parse dataset IDs from CLI argument
    dataset_ids: set[str] | None = None
    if datasets:
        dataset_ids = {did.strip() for did in datasets.split(",") if did.strip()}
        print_info(f"Processing datasets: {', '.join(sorted(dataset_ids))}")
    elif "datasets" in components and not yes:
        # Ask if user wants to select specific datasets
        # First, collect all available datasets from selected clients
        available_datasets: list[tuple[str, str]] = []
        for c in available_clients:
            client_config_dir = os.path.join(clients_dir, c)
            available_datasets.extend(_load_available_datasets(client_config_dir))

        if available_datasets:
            # Remove duplicates and sort
            available_datasets = sorted(set(available_datasets), key=lambda x: x[1])

            if Confirm.ask("Would you like to select specific datasets?", default=False):
                selected = await select_datasets_interactive(available_datasets)
                if not selected:
                    print_info("No datasets selected. Aborted.")
                    return
                dataset_ids = selected
                print_info(f"Selected {len(dataset_ids)} datasets")

    # Confirmation for full init (when no --only specified)
    if not only and not yes:
        if not _confirm_full_init(available_clients, components):
            print_info("Aborted.")
            return

    print_info(f"Processing clients: {', '.join(available_clients)}")

    async with get_admin_client() as client:
        if not await client.health_check():
            print_error("Admin API is not available.")
            return

        # Get existing data for updates
        existing_datasets: dict[str, DataSet] = {}
        if "datasets" in components:
            all_datasets = await client.get_datasets()
            existing_datasets = {str(ds.id_): ds for ds in all_datasets}

        for client_name in available_clients:
            print_info(f"\nProcessing client: {client_name}")
            client_config_dir = os.path.join(clients_dir, client_name)

            try:
                await _process_client(
                    admin_client=client,
                    client_id=client_name,
                    client_config_dir=client_config_dir,
                    existing_datasets=existing_datasets,
                    dataset_ids=dataset_ids,
                    components=components,
                )
                print_success(f"Client {client_name} processed successfully")
            except Exception as e:
                print_error(f"Failed to process client {client_name}: {e}")
                raise


async def _process_client(
    admin_client: AdminClient,
    client_id: str,
    client_config_dir: str,
    existing_datasets: dict[str, DataSet],
    dataset_ids: set[str] | None,
    components: set[str],
) -> None:
    """Process a single client configuration."""
    # Load configurations
    tools_cfg = utils.read_yaml(f"{client_config_dir}/tools.yaml")
    channel_cfg = utils.read_yaml(f"{client_config_dir}/channels.yaml")
    onboarding_cfg = utils.optional_read_yaml(f"{client_config_dir}/onboarding.yaml")

    glossaries_dir = f"{client_config_dir}/glossaries"
    dial_files_dir = f"{client_config_dir}/dial_files"

    # Upload DIAL files
    if "files" in components:
        if os.path.exists(dial_files_dir):
            await _upload_dial_files(dial_files_dir)
        else:
            print_info(f"No DIAL files directory found at {dial_files_dir}")

    # Process channels
    channels: dict[str, Channel] = {}
    if "channels" in components:
        process_glossaries = "glossaries" in components
        channels = await _process_channels(
            admin_client, channel_cfg, tools_cfg, onboarding_cfg, process_glossaries, glossaries_dir
        )

    # Process data sources
    data_sources: dict[str, DataSource] = {}
    if "datasources" in components:
        data_sources_cfg = utils.read_yaml(f"{client_config_dir}/data_sources.yaml")
        data_sources = await _process_data_sources(admin_client, data_sources_cfg)

    # Process datasets
    if "datasets" in components:
        await _process_datasets(
            admin_client,
            client_id,
            client_config_dir,
            data_sources,
            channels,
            existing_datasets,
            dataset_ids,
        )


async def _upload_dial_files(dial_files_dir: str) -> None:
    """Upload files to DIAL storage."""
    dial_url = cli_settings.dial_url
    dial_api_key = cli_settings.dial_api_key

    if not dial_url or not dial_api_key:
        print_warning("DIAL credentials not configured, skipping file upload")
        return

    print_info("Uploading files to DIAL...")

    async with dial_core_factory(dial_url, dial_api_key) as dial:
        for root, _, files in os.walk(dial_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                dial_path = os.path.relpath(file_path, dial_files_dir).replace("\\", "/")

                mime_type = _get_mime_type(file)
                print_info(f"  Uploading: {dial_path}")

                with open(file_path, "rb") as f:
                    content = f.read()
                    await dial.put_file(dial_path, mime_type, content)


def _get_mime_type(filename: str) -> str:
    """Get MIME type for file."""
    ext_map = {
        ".yaml": "application/x-yaml",
        ".yml": "application/x-yaml",
        ".json": "application/json",
        ".csv": "text/csv",
        ".md": "text/markdown",
    }
    for ext, mime in ext_map.items():
        if filename.endswith(ext):
            return mime
    return "application/octet-stream"


async def _process_channels(
    client: AdminClient,
    channel_cfg: dict[str, Any],
    tools_cfg: dict[str, Any],
    onboarding_cfg: dict[str, Any] | None,
    process_glossaries: bool,
    glossaries_dir: str,
) -> dict[str, Channel]:
    """Process channel configurations."""
    print_info("Processing channels...")

    existing = {ch.deployment_id: ch for ch in await client.get_channels()}
    channels: dict[str, Channel] = {}

    for ch_cfg in channel_cfg.get("channels", []):
        deployment_id = ch_cfg["deployment_id"]

        # Add tools to channel config
        _add_tools_to_channel(ch_cfg, tools_cfg)
        _add_onboarding_to_channel(ch_cfg, onboarding_cfg)

        if deployment_id in existing:
            # Update existing channel
            channel = await client.update_channel(existing[deployment_id].id, ch_cfg)
            print_info(f"  Updated channel: {deployment_id}")
        else:
            # Create new channel
            channel = await client.create_channel(ch_cfg)
            print_info(f"  Created channel: {deployment_id}")

        channels[deployment_id] = channel

        # Process glossary
        glossary_file = ch_cfg.get("glossary")
        if process_glossaries and glossary_file:
            glossary_path = os.path.join(glossaries_dir, glossary_file)
            if os.path.exists(glossary_path):
                terms = utils.read_csv_as_dict_list(glossary_path)
                await _process_glossary(client, channel, terms)
                print_info(f"  Processed glossary: {glossary_file}")

    return channels


def _add_tools_to_channel(ch_cfg: dict[str, Any], tools_cfg: dict[str, Any]) -> None:
    """Add tool configurations to channel."""
    if "details" not in ch_cfg:
        ch_cfg["details"] = {}

    for tool in tools_cfg.get("tools", []):
        if ch_cfg["deployment_id"] not in tool.get("channels", []):
            continue

        tool_type = tool["type"]
        ch_cfg["details"][tool_type] = {
            k: v for k, v in tool.items() if k in ("name", "description", "details")
        }


def _add_onboarding_to_channel(
    ch_cfg: dict[str, Any], onboarding_cfg: dict[str, Any] | None
) -> None:
    """Add onboarding configuration to channel."""
    if not onboarding_cfg:
        return

    if "details" not in ch_cfg:
        ch_cfg["details"] = {}

    for ob_cfg in onboarding_cfg.get("configurations", []):
        if ch_cfg["deployment_id"] not in ob_cfg.get("channels", []):
            continue

        ch_cfg["details"]["onboarding"] = {k: v for k, v in ob_cfg.items() if k != "channels"}


async def _process_glossary(
    client: AdminClient,
    channel: Channel,
    terms: list[dict[str, str]],
) -> None:
    """Process glossary terms for a channel."""
    existing = await client.get_glossary_terms(channel.id)
    existing_map: dict[tuple[str, str | None, str | None], GlossaryTerm] = {
        (t.term, t.domain, t.source): t for t in existing
    }

    add_terms = []
    update_terms = []
    found_ids: set[int] = set()

    for term in terms:
        key = (term["term"], term.get("domain"), term.get("source"))
        if key in existing_map:
            existing_term = existing_map[key]
            found_ids.add(existing_term.id)
            if term.get("definition") != existing_term.definition:
                update_terms.append({"id": existing_term.id, "definition": term["definition"]})
        else:
            add_terms.append(term)

    if add_terms:
        await client.create_glossary_terms_bulk(channel.id, add_terms)
    if update_terms:
        await client.update_glossary_terms_bulk(update_terms)

    # Delete removed terms
    delete_ids = [t.id for t in existing if t.id not in found_ids]
    if delete_ids:
        await client.delete_glossary_terms_bulk(delete_ids)


async def _process_data_sources(
    client: AdminClient,
    data_sources_cfg: dict[str, Any],
) -> dict[str, DataSource]:
    """Process data source configurations."""
    print_info("Processing data sources...")

    existing = {ds.title: ds for ds in await client.get_data_sources()}
    ds_types = {t.name: t.id for t in await client.get_data_source_types()}
    data_sources: dict[str, DataSource] = {}

    for ds_cfg in data_sources_cfg.get("dataSources", []):
        title = ds_cfg["title"]

        # Replace type name with type_id
        type_name = ds_cfg.pop("type", None)
        if type_name:
            ds_cfg["type_id"] = ds_types.get(type_name)

        if title in existing:
            ds = await client.update_data_source(existing[title].id, ds_cfg)
            print_info(f"  Updated data source: {title}")
        else:
            ds = await client.create_data_source(ds_cfg)
            print_info(f"  Created data source: {title}")

        data_sources[title] = ds

    return data_sources


async def _process_datasets(
    client: AdminClient,
    client_id: str,
    client_config_dir: str,
    data_sources: dict[str, DataSource],
    channels: dict[str, Channel],
    existing_datasets: dict[str, DataSet],
    dataset_ids: set[str] | None,
) -> None:
    """Process dataset configurations."""
    print_info("Processing datasets...")

    datasets_dir = os.path.join(client_config_dir, "datasets")
    if not os.path.exists(datasets_dir):
        print_warning(f"No datasets directory found: {datasets_dir}")
        return

    datasets_cfg: list[dict[str, Any]] = []
    for filename in os.listdir(datasets_dir):
        if not filename.endswith(".yaml"):
            continue
        file_path = os.path.join(datasets_dir, filename)
        cfg = utils.read_yaml(file_path)
        datasets_cfg.extend(cfg.get("dataSets", []))

    channel_datasets: dict[int, list[ChannelDatasetExpanded]] = {}

    for ds_cfg in datasets_cfg:
        urn = ds_cfg.get("details", {}).get("urn")

        # Filter by dataset IDs if specified
        if dataset_ids and urn not in dataset_ids:
            continue

        # Link to data source
        ds_name = ds_cfg.pop("dataSource", None)
        if ds_name and ds_name in data_sources:
            ds_cfg["data_source_id"] = data_sources[ds_name].id

        dataset_id_str = ds_cfg.get("id_")

        if dataset_id_str and dataset_id_str in existing_datasets:
            dataset = await client.update_dataset(existing_datasets[dataset_id_str].id, ds_cfg)
            print_info(f"  Updated dataset: {urn}")
        else:
            dataset = await client.create_dataset(ds_cfg)
            print_info(f"  Created dataset: {urn}")

        # Link to channels
        for ch_name in ds_cfg.get("channels", []):
            if ch_name not in channels:
                continue
            channel = channels[ch_name]
            ch_id = channel.id

            if ch_id not in channel_datasets:
                channel_datasets[ch_id] = await client.get_channel_datasets(ch_id)

            # Check if already linked
            if not any(cd.dataset_id == dataset.id for cd in channel_datasets[ch_id]):
                await client.add_dataset_to_channel(ch_id, dataset.id)
                print_info(f"    Linked to channel: {ch_name}")


init_command = Command(
    name="init",
    description="Initialize content (channels, data sources, datasets, glossaries)",
    handler=init_handler,
    args=[
        CommandArg(
            name="client-id",
            description="Comma-separated list of client IDs to process",
        ),
        CommandArg(
            name="datasets",
            description="Comma-separated list of dataset URNs to process",
        ),
        CommandArg(
            name="only",
            short_name="o",
            description="Components to process: channels,datasources,datasets,glossaries,files",
        ),
        CommandArg(
            name="yes",
            short_name="y",
            description="Skip confirmation prompt",
            is_flag=True,
        ),
    ],
)

# Command group
content_group = CommandGroup(
    name="content",
    description="Content initialization and management",
)
content_group.add_command(init_command)
