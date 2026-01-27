"""Channel commands for StatGPT CLI."""

import asyncio
import csv
import os
from collections import Counter

import questionary
from rich.table import Table

from statgpt.cli.commands.base import Command, CommandArg, CommandGroup
from statgpt.cli.settings import cli_runtime, cli_settings
from statgpt.cli.shared import (
    NonInteractiveError,
    console,
    get_admin_client,
    print_error,
    print_info,
    print_success,
    select_item_interactive,
    spinner_status,
)
from statgpt.common.data.sdmx.common import UrnReference
from statgpt.common.schemas import Channel, ChannelDatasetExpanded, ChannelDatasetVersion, DataSet

POLL_INTERVAL = 1  # seconds


def _get_urn_display(dataset: DataSet | None) -> str:
    """Get URN string for display from dataset details."""
    if not dataset:
        return "N/A"
    urn_data = dataset.details.get("urn")
    if not urn_data:
        return "N/A"
    return UrnReference.model_validate(urn_data).short_urn()


async def import_handler(
    file: str | None = None,
    clean: bool = False,
    update_datasets: bool = False,
    update_data_sources: bool = False,
    admin_url: str | None = None,
) -> None:
    """Import a channel from a zip archive."""
    # Interactive file selection if not provided
    if not file:
        if cli_runtime.non_interactive:
            raise NonInteractiveError(
                "Missing required parameter: --file\n"
                "  Usage: statgpt channel import --file <path.zip>"
            )
        file = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: questionary.path(
                "Select channel archive file:",
                only_directories=False,
                file_filter=lambda f: f.endswith(".zip") or os.path.isdir(f),
            ).ask(),
        )
        if not file:
            print_error("No file selected")
            return

    # Validate file exists
    if not os.path.exists(file):
        print_error(f"File not found: {file}")
        return

    # Interactive options if not provided via args
    if not any([clean, update_datasets, update_data_sources]) and not cli_runtime.non_interactive:
        options = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: questionary.checkbox(
                "Select import options:",
                choices=[
                    questionary.Choice("Clean up existing channels before import", value="clean"),
                    questionary.Choice("Update existing datasets", value="update_datasets"),
                    questionary.Choice("Update existing data sources", value="update_data_sources"),
                ],
            ).ask(),
        )
        if options:
            clean = "clean" in options
            update_datasets = "update_datasets" in options
            update_data_sources = "update_data_sources" in options

    print_info(f"Importing channel from: {file}")
    print_info(
        f"Options: clean={clean}, update_datasets={update_datasets}, "
        f"update_data_sources={update_data_sources}"
    )

    async with get_admin_client(base_url=admin_url) as client:
        # Health check
        if not await client.health_check():
            print_error("Admin API is not available. Make sure the server is running.")
            return

        # Start import
        with spinner_status("Starting import...") as status:
            try:
                job = await client.import_channel(
                    file_path=file,
                    clean_up=clean,
                    update_datasets=update_datasets,
                    update_data_sources=update_data_sources,
                )
                job_id = job.id
                print_info(f"Import job started: {job_id}")

                # Poll for completion
                while True:
                    job_status = await client.get_import_job_status(str(job_id))
                    status.update(f"Import status: {job_status.status}")

                    if job_status.status == "COMPLETED":
                        print_success(f"Channel imported successfully! ID: {job_status.channel_id}")
                        return
                    elif job_status.status == "FAILED":
                        reason = job_status.reason_for_failure or "Unknown error"
                        print_error(f"Import failed: {reason}")
                        return

                    await asyncio.sleep(POLL_INTERVAL)

            except Exception as e:
                print_error(f"Import failed: {e}")
                raise


async def _select_channel_interactive(channels: list[Channel]) -> Channel | None:
    """Interactive channel selection with filtering."""
    channel_map = {ch.deployment_id: ch for ch in channels}
    items = [
        (ch.deployment_id, f"{ch.deployment_id} - {ch.title}")
        for ch in sorted(channels, key=lambda ch: ch.deployment_id)
    ]

    try:
        selected = await select_item_interactive(
            items,
            title="Select Channel (type to filter)",
            filter_enabled=True,
        )
    except NonInteractiveError:
        available = ", ".join(
            ch.deployment_id for ch in sorted(channels, key=lambda c: c.deployment_id)
        )
        raise NonInteractiveError(
            f"Missing required parameter: -c/--channel\n"
            f"  Available channels: {available}\n"
            f"  Usage: statgpt channel <command> -c <channel>"
        ) from None

    if not selected:
        return None
    return channel_map.get(selected)


def _get_status_style(status: str) -> str:
    """Get Rich style for status."""
    status_styles = {
        "COMPLETED": "green",
        "PROCESSING": "yellow",
        "PENDING": "blue",
        "FAILED": "red",
        "UNKNOWN": "dim",
    }
    return status_styles.get(status, "white")


def _get_version_hashes(version: ChannelDatasetVersion | None) -> dict[str, str | None]:
    """Extract hashes from version data."""
    if not version:
        return {}
    return {
        "hash_structure": version.structure_hash,
        "hash_indicator_dims": version.indicator_dimensions_hash,
        "hash_non_indicator_dims": version.non_indicator_dimensions_hash,
        "hash_special_dims": version.special_dimensions_hash,
    }


def _export_status_csv(
    channel: Channel,
    datasets: list[ChannelDatasetExpanded],
    output_path: str,
) -> None:
    """Export status to CSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Collect all hash columns from the data
    hash_columns = [
        "hash_structure",
        "hash_indicator_dims",
        "hash_non_indicator_dims",
        "hash_special_dims",
    ]
    header = ["channel", "dataset_urn", "dataset_title", "status"] + hash_columns

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for cd in datasets:
            dataset = cd.dataset
            version = cd.last_completed_version

            # Use version status if available, otherwise fall back to preprocessing_status
            status = str(version.preprocessing_status) if version else str(cd.preprocessing_status)
            hashes = _get_version_hashes(version)

            row = [
                channel.deployment_id,
                _get_urn_display(dataset),
                dataset.title if dataset else "",
                status,
            ]
            # Add hash values in order
            for col in hash_columns:
                row.append(hashes.get(col, "") or "")

            writer.writerow(row)


async def status_handler(
    channel: str | None = None,
    output_path: str | None = None,
) -> None:
    """Show preprocessing status of datasets for a channel."""
    async with get_admin_client() as client:
        if not await client.health_check():
            print_error("Admin API is not available.")
            return

        # Fetch channels
        channels = await client.get_channels()
        if not channels:
            print_error("No channels found.")
            return

        # Select channel
        if channel:
            selected_channel = next((ch for ch in channels if ch.deployment_id == channel), None)
            if not selected_channel:
                print_error(f"Channel not found: {channel}")
                return
        else:
            selected_channel = await _select_channel_interactive(channels)
            if not selected_channel:
                return

        # Fetch datasets for channel
        with spinner_status("Fetching data..."):
            ch_datasets = await client.get_channel_datasets(selected_channel.id)
            index_status = await client.get_channel_index_status(selected_channel.id)

        # Print channel info
        console.print(f"\n[bold cyan]Channel:[/bold cyan] {selected_channel.title}")
        console.print(f"[dim]Deployment ID: {selected_channel.deployment_id}[/dim]")

        # Print datasets status table
        console.print("\n[bold cyan]Datasets Status[/bold cyan]")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Status", width=15)
        table.add_column("Dataset", ratio=2)
        table.add_column("URN", ratio=2)

        # Sort by status priority: FAILED first, then NOT_STARTED, IN_PROGRESS, QUEUED, COMPLETED
        status_order = {
            "FAILED": 0,
            "NOT_STARTED": 1,
            "IN_PROGRESS": 2,
            "QUEUED": 3,
            "COMPLETED": 4,
        }
        ch_datasets.sort(
            key=lambda cd: (
                status_order.get(str(cd.preprocessing_status), 99),
                _get_urn_display(cd.dataset),
            )
        )

        for cd in ch_datasets:
            status = str(cd.preprocessing_status)
            dataset = cd.dataset
            dataset_title = dataset.title if dataset else "Unknown"
            dataset_urn = _get_urn_display(dataset)
            status_style = _get_status_style(status)
            table.add_row(
                f"[{status_style}]{status}[/{status_style}]",
                dataset_title,
                dataset_urn,
            )

        console.print(table)

        # Print summary
        console.print("\n[bold cyan]Status Summary[/bold cyan]")
        counter: Counter[str] = Counter(str(cd.preprocessing_status) for cd in ch_datasets)
        for status, count in sorted(counter.items()):
            style = _get_status_style(status)
            console.print(f"  [{style}]{status}[/{style}]: {count}")

        # Print index status
        try:
            console.print("\n[bold cyan]Index Status[/bold cyan]")
            vs = index_status.vector_store
            sizes = vs.sizes
            dedup = vs.deduplication

            console.print(f"  [dim]Scope:[/dim] {index_status.scope}")
            console.print("  [dim]Vector Store index:[/dim]")
            console.print(
                f"    [dim]Available dimensions index:[/dim] {sizes.available_dimensions_size}"
            )
            console.print(
                f"    [dim]Special dimensions index:[/dim] {sizes.special_dimensions_size}"
            )
            console.print(
                f"    [dim]Indicator dimensions index:[/dim] {sizes.indicator_dimensions_size}"
            )

            if dedup.deduplication_required:
                console.print(
                    f"    [yellow]Duplicates:[/yellow] {dedup.total_duplicate_count} "
                    f"(deduplication required)"
                )
                console.print(
                    f"    [dim]Run:[/dim] [cyan]channel deduplicate "
                    f"-c {selected_channel.deployment_id}[/cyan]"
                )
            else:
                console.print(
                    f"    [green]Duplicates:[/green] {dedup.total_duplicate_count} "
                    f"(no deduplication required)"
                )
        except Exception as e:
            print_error(f"Failed to fetch index status: {e}")

        # Export to CSV if requested
        if output_path:
            _export_status_csv(selected_channel, ch_datasets, output_path)
            print_success(f"Status report saved to: {output_path}")


async def list_handler() -> None:
    """List all available channels."""
    async with get_admin_client() as client:
        if not await client.health_check():
            print_error("Admin API is not available.")
            return

        with spinner_status("Fetching channels..."):
            channels = await client.get_channels()

            # Count datasets per channel in parallel
            async def get_dataset_count(ch: Channel) -> tuple[int, int]:
                ch_datasets = await client.get_channel_datasets(ch.id)
                return ch.id, len(ch_datasets)

            results = await asyncio.gather(*[get_dataset_count(ch) for ch in channels])
            dataset_counts: dict[int, int] = dict(results)

        if not channels:
            print_info("No channels found.")
            return

        # Sort by deployment_id
        channels = sorted(channels, key=lambda ch: ch.deployment_id)

        console.print(f"\n[bold cyan]Channels[/bold cyan] ({len(channels)} total)\n")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Deployment ID", style="cyan", ratio=1)
        table.add_column("Title", ratio=2)
        table.add_column("Datasets", width=8, justify="right")
        table.add_column("ID", width=6, justify="right")

        for ch in channels:
            dataset_count = dataset_counts.get(ch.id, 0)
            table.add_row(
                ch.deployment_id,
                ch.title,
                str(dataset_count),
                str(ch.id),
            )

        console.print(table)


async def deduplicate_handler(channel: str | None = None) -> None:
    """Deduplicate embeddings for a channel."""
    async with get_admin_client() as client:
        if not await client.health_check():
            print_error("Admin API is not available.")
            return

        # Fetch channels
        channels = await client.get_channels()
        if not channels:
            print_error("No channels found.")
            return

        # Select channel
        if channel:
            selected_channel = next((ch for ch in channels if ch.deployment_id == channel), None)
            if not selected_channel:
                print_error(f"Channel not found: {channel}")
                return
        else:
            selected_channel = await _select_channel_interactive(channels)
            if not selected_channel:
                return

        print_info(f"Deduplicating channel: {selected_channel.title}")

        with spinner_status("Deduplicating embeddings..."):
            await client.deduplicate_channel(selected_channel.id)

        print_success("Deduplication completed")


async def _select_reindex_mode_interactive(datasets: list[DataSet], channel: Channel) -> str | None:
    """Interactive mode selection."""
    # Show datasets info
    console.print(f"\n[bold]Datasets in {channel.title}:[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("URN", ratio=2)
    table.add_column("Title", ratio=2)
    table.add_column("Status", width=12)

    for ds in datasets:
        urn = _get_urn_display(ds)
        status = ds.status.status if ds.status else "UNKNOWN"
        table.add_row(urn, ds.title, status)

    console.print(table)
    console.print()

    items = [
        ("all", "Reindex all datasets"),
        ("channel", "Reindex entire channel (slower)"),
        ("dataset", "Reindex specific dataset"),
    ]

    try:
        return await select_item_interactive(
            items,
            title="Select Reindex Mode",
            filter_enabled=False,
        )
    except NonInteractiveError:
        raise NonInteractiveError(
            "Missing required parameter: --mode\n"
            "  Available modes: all, channel, dataset\n"
            "  Usage: statgpt channel reindex -c <channel> --mode <mode>"
        ) from None


async def _select_dataset_interactive(datasets: list[DataSet]) -> str | None:
    """Interactive dataset selection with filtering."""
    items = [
        (_get_urn_display(ds), f"{_get_urn_display(ds)} - {ds.title}")
        for ds in datasets
        if ds.details.get("urn")
    ]

    try:
        return await select_item_interactive(
            items,
            title="Select Dataset (type to filter)",
            filter_enabled=True,
        )
    except NonInteractiveError:
        available = ", ".join(_get_urn_display(ds) for ds in datasets if ds.details.get("urn"))
        raise NonInteractiveError(
            f"Missing required parameter: --dataset-urn\n"
            f"  Available datasets: {available}\n"
            f"  Usage: statgpt channel reindex -c <channel> --mode dataset --dataset-urn <urn>"
        ) from None


async def reindex_handler(
    channel: str | None = None,
    mode: str | None = None,
    dataset_urn: str | None = None,
) -> None:
    """Reindex dataset embeddings for a channel."""
    async with get_admin_client() as client:
        if not await client.health_check():
            print_error("Admin API is not available.")
            return

        # Fetch channels
        channels = await client.get_channels()
        if not channels:
            print_error("No channels found.")
            return

        # Select channel
        if channel:
            selected_channel = next((ch for ch in channels if ch.deployment_id == channel), None)
            if not selected_channel:
                print_error(f"Channel not found: {channel}")
                return
        else:
            selected_channel = await _select_channel_interactive(channels)
            if not selected_channel:
                return

        print_info(f"Selected channel: {selected_channel.title}")

        # Get datasets for channel
        datasets = await client.get_datasets(channel_id=selected_channel.id)

        # Select mode
        if not mode:
            mode = await _select_reindex_mode_interactive(datasets, selected_channel)
            if not mode:
                return

        # Handle dataset selection
        dataset_ids: list[int] | None = None

        if mode == "dataset":
            if not dataset_urn:
                dataset_urn = await _select_dataset_interactive(datasets)
                if not dataset_urn:
                    return

            selected_ds = next(
                (ds for ds in datasets if _get_urn_display(ds) == dataset_urn),
                None,
            )
            if not selected_ds:
                print_error(f"Dataset not found: {dataset_urn}")
                return
            dataset_ids = [selected_ds.id]
        elif mode == "all":
            dataset_ids = [ds.id for ds in datasets]

        # Reindex
        max_embeddings = cli_settings.max_embeddings

        if dataset_ids:
            msg = f"Reindexing {len(dataset_ids)} dataset(s)..."
        else:
            msg = "Reindexing all datasets in channel..."

        with spinner_status(msg):
            await client.reload_channel_indicators(
                channel_id=selected_channel.id,
                dataset_ids=dataset_ids,
                max_n_embeddings=max_embeddings,
            )

        print_success("Reindex operation started")


list_command = Command(
    name="list",
    description="List all available channels",
    handler=list_handler,
)

import_command = Command(
    name="import",
    description="Import a channel from a zip archive",
    handler=import_handler,
    args=[
        CommandArg(
            name="file",
            description="Path to the channel archive file (.zip)",
            required=False,
        ),
        CommandArg(
            name="clean",
            description="Clean up existing channels before import",
            is_flag=True,
        ),
        CommandArg(
            name="update-datasets",
            description="Update existing datasets if they already exist",
            is_flag=True,
        ),
        CommandArg(
            name="update-data-sources",
            description="Update existing data sources if they already exist",
            is_flag=True,
        ),
        CommandArg(
            name="admin-url",
            description="Override Admin API URL",
        ),
    ],
)

status_command = Command(
    name="status",
    description="Show preprocessing status of datasets for a channel",
    handler=status_handler,
    args=[
        CommandArg(
            name="channel",
            short_name="c",
            description="Channel deployment ID",
        ),
        CommandArg(
            name="output-path",
            short_name="o",
            description="Save status report to CSV file",
        ),
    ],
)


deduplicate_command = Command(
    name="deduplicate",
    description="Deduplicate embeddings for a channel",
    handler=deduplicate_handler,
    args=[
        CommandArg(
            name="channel",
            short_name="c",
            description="Channel deployment ID",
        ),
    ],
)

reindex_command = Command(
    name="reindex",
    description="Reindex dataset embeddings for a channel",
    handler=reindex_handler,
    args=[
        CommandArg(
            name="channel",
            short_name="c",
            description="Channel deployment ID",
        ),
        CommandArg(
            name="mode",
            description="Reindex mode (all/channel/dataset)",
            choices=["all", "channel", "dataset"],
        ),
        CommandArg(
            name="dataset-urn",
            description="Dataset URN (required when mode=dataset)",
        ),
    ],
)


# Command group
channel_group = CommandGroup(
    name="channel",
    description="Channel management commands",
)
channel_group.add_command(list_command)
channel_group.add_command(import_command)
channel_group.add_command(status_command)
channel_group.add_command(deduplicate_command)
channel_group.add_command(reindex_command)
