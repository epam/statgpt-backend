"""Channel resolution shared by the channel-scoped commands."""

from statgpt.cli.shared.admin_client import AdminClient
from statgpt.cli.shared.console import print_error
from statgpt.cli.shared.prompts import NonInteractiveError, select_item_interactive
from statgpt.common.schemas import Channel


async def select_channel_interactive(channels: list[Channel]) -> Channel | None:
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
            f"  Usage: statgpt <group> <command> -c <channel>"
        ) from None

    if not selected:
        return None
    return channel_map.get(selected)


async def select_channel(client: AdminClient, deployment_id: str | None) -> Channel | None:
    """Resolve the channel a command works on, or None when there is nothing to work on.

    The whole preamble every channel-scoped command needs: check the API is up, list the
    channels, then either match the given deployment id or let the user pick one. Returns
    None - rather than raising - for every "nothing to do" case, since each of them has
    already been reported to the user.
    """
    if not await client.health_check():
        print_error("Admin API is not available.")
        return None

    channels = await client.get_channels()
    if not channels:
        print_error("No channels found.")
        return None

    if deployment_id:
        channel = next((ch for ch in channels if ch.deployment_id == deployment_id), None)
        if not channel:
            print_error(f"Channel not found: {deployment_id}")
        return channel

    return await select_channel_interactive(channels)
