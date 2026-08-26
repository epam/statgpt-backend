"""Discovery dataset (Grade C) commands for StatGPT CLI."""

import asyncio
import os

import questionary
from rich.table import Table

from statgpt.cli.commands.base import Command, CommandArg, CommandGroup
from statgpt.cli.settings import cli_runtime
from statgpt.cli.shared import (
    BatchPartialFailureError,
    DiscoveryPayloadError,
    NonInteractiveError,
    confirm_interactive,
    console,
    get_admin_client,
    is_upload_file,
    print_error,
    print_info,
    print_success,
    render_payload_problems,
    render_upload_summary,
    select_channel,
    spinner_status,
)
from statgpt.common.schemas import (
    DiscoveryIndexingJob,
    DiscoveryUploadMode,
    PreprocessingStatusEnum,
)

POLL_INTERVAL = 1  # seconds


async def _select_file_interactive() -> str | None:
    """Ask for the file to upload, in the shape `channel import` uses."""
    if cli_runtime.non_interactive:
        raise NonInteractiveError(
            "Missing required parameter: --file\n"
            "  Usage: statgpt discovery upload -c <channel> --file <path.xlsx|path.csv>"
        )
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: questionary.path(
            "Select discovery datasets file:",
            only_directories=False,
            file_filter=lambda f: is_upload_file(f) or os.path.isdir(f),
        ).ask(),
    )


async def upload_handler(
    channel: str | None = None,
    file: str | None = None,
    mode: str = DiscoveryUploadMode.UPSERT.value,
) -> None:
    """Upload a discovery datasets workbook or CSV to a channel."""
    upload_mode = DiscoveryUploadMode(mode)

    async with get_admin_client() as client:
        selected_channel = await select_channel(client, channel)
        if not selected_channel:
            return

        if not file:
            file = await _select_file_interactive()
            if not file:
                print_error("No file selected")
                return

        if not os.path.exists(file):
            print_error(f"File not found: {file}")
            return

        if upload_mode is DiscoveryUploadMode.REPLACE and not confirm_interactive(
            f"Replace mode deletes records absent from the file. Continue with"
            f" {selected_channel.deployment_id}?",
            default=False,
            error_message=(
                "Replace mode requires confirmation.\n"
                "  Use --mode upsert to keep records the file does not mention."
            ),
        ):
            print_info("Aborted.")
            return

        print_info(f"Uploading {os.path.basename(file)} to {selected_channel.deployment_id}...")

        try:
            summary = await client.upload_discovery_datasets(selected_channel.id, file, upload_mode)
        except DiscoveryPayloadError as e:
            render_payload_problems(e.detail)
            print_error("Nothing was saved.")
            raise BatchPartialFailureError("Discovery upload was refused") from e

        render_upload_summary(summary)
        print_success(f"Uploaded {summary.rows_read} record(s)")
        print_info(
            f"Run: [cyan]discovery reindex -c {selected_channel.deployment_id}[/cyan]"
            f" to publish the records"
        )


def _render_job_results(job: DiscoveryIndexingJob) -> None:
    """Print what a completed indexing run did."""
    table = Table(title="Discovery indexing results", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Count", justify="right")

    rows = [
        ("Records evaluated", job.records_total),
        ("Records valid", job.records_valid),
        ("Records invalid", job.records_invalid),
        ("Documents published", job.documents_upserted),
        ("Documents removed", job.documents_deleted),
    ]
    for label, value in rows:
        table.add_row(label, str(value) if value is not None else "-")

    console.print(table)
    if job.details:
        console.print(f"[dim]{job.details}[/dim]")


async def reindex_handler(channel: str | None = None, force: bool = False) -> None:
    """Re-validate and re-publish a channel's discovery datasets."""
    async with get_admin_client() as client:
        selected_channel = await select_channel(client, channel)
        if not selected_channel:
            return

        print_info(f"Reindexing discovery datasets of: {selected_channel.title}")

        with spinner_status("Starting discovery indexing...") as status:
            job = await client.trigger_discovery_indexing(selected_channel.id, force=force)
            print_info(f"Discovery indexing job started: {job.id}")

            while True:
                job = await client.get_discovery_indexing_job(job.id)
                status.update(f"Indexing status: {job.status}")

                if job.status == PreprocessingStatusEnum.COMPLETED:
                    break
                if job.status == PreprocessingStatusEnum.FAILED:
                    reason = job.reason_for_failure or "Unknown error"
                    print_error(f"Discovery indexing failed: {reason}")
                    return

                await asyncio.sleep(POLL_INTERVAL)

        _render_job_results(job)
        print_success("Discovery indexing completed")


async def clear_handler(channel: str | None = None, yes: bool = False) -> None:
    """Delete every discovery dataset of a channel, and its published documents."""
    async with get_admin_client() as client:
        selected_channel = await select_channel(client, channel)
        if not selected_channel:
            return

        stats = await client.get_discovery_stats(selected_channel.id)
        if not stats.total:
            print_info(f"{selected_channel.deployment_id} holds no discovery datasets.")
            return

        if not yes and not confirm_interactive(
            f"Delete all {stats.total} discovery dataset(s) of"
            f" {selected_channel.deployment_id} and their published documents?",
            default=False,
            error_message=(
                "Deleting a channel's discovery datasets requires confirmation.\n"
                "  Use -y/--yes to skip it.\n"
                "  Usage: statgpt discovery clear -c <channel> -y"
            ),
        ):
            print_info("Aborted.")
            return

        # Not instant: the records' documents are withdrawn from the RAG channel first.
        with spinner_status(f"Deleting discovery datasets of {selected_channel.title}..."):
            deleted = await client.clear_discovery_datasets(selected_channel.id)

        print_success(f"Deleted {len(deleted)} record(s) and their published documents")


upload_command = Command(
    name="upload",
    description="Upload a discovery datasets workbook (.xlsx) or CSV to a channel",
    handler=upload_handler,
    args=[
        CommandArg(
            name="channel",
            short_name="c",
            description="Channel deployment ID",
        ),
        CommandArg(
            name="file",
            short_name="f",
            description="Path to the discovery datasets file (.xlsx or .csv)",
        ),
        CommandArg(
            name="mode",
            description=(
                "How the file is reconciled with the channel's records:"
                " upsert keeps records the file does not mention, replace deletes them"
            ),
            choices=[mode.value for mode in DiscoveryUploadMode],
            default=DiscoveryUploadMode.UPSERT.value,
        ),
    ],
)

reindex_command = Command(
    name="reindex",
    description="Re-validate and re-publish the discovery datasets of a channel",
    handler=reindex_handler,
    args=[
        CommandArg(
            name="channel",
            short_name="c",
            description="Channel deployment ID",
        ),
        CommandArg(
            name="force",
            description=(
                "Rebuild every document even if nothing changed;"
                " each rebuilt record is briefly absent from the index"
            ),
            is_flag=True,
        ),
    ],
)


clear_command = Command(
    name="clear",
    description="Delete all discovery datasets of a channel and their published documents",
    handler=clear_handler,
    args=[
        CommandArg(
            name="channel",
            short_name="c",
            description="Channel deployment ID",
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
discovery_group = CommandGroup(
    name="discovery",
    description="Discovery dataset (Grade C) management commands",
)
discovery_group.add_command(upload_command)
discovery_group.add_command(reindex_command)
discovery_group.add_command(clear_command)
