"""Authentication commands for CLI."""

from datetime import datetime

import questionary

from statgpt.cli.commands.base import Command, CommandArg, CommandGroup
from statgpt.cli.settings import cli_settings
from statgpt.cli.shared.auth import (
    AuthConfigError,
    AuthenticationError,
    get_token_info,
    is_logged_in,
    login,
    logout,
)
from statgpt.cli.shared.console import console, print_error, print_info, print_success


async def login_handler(method: str) -> None:
    """Handle login command."""
    if method not in ("interactive", "system_user"):
        print_error(f"Invalid login method: {method}. Use 'interactive' or 'system_user'.")
        return

    # Check if already logged in
    if is_logged_in():
        info = get_token_info()
        if info:
            mins = info["expires_in_seconds"] // 60
            print_info(f"Already logged in via {info['provider']} (token expires in {mins} min)")
            console.print(
                "[dim]Use 'auth logout' to clear the session, or 'auth login --method ...' to refresh.[/dim]"
            )

            # Ask if they want to refresh

            refresh = await questionary.confirm(
                "Do you want to refresh the token?",
                default=False,
            ).ask_async()

            if not refresh:
                return

    try:
        result = login(method)  # type: ignore[arg-type]
        mins = result.expires_in // 60
        print_success(f"Successfully authenticated! Token expires in {mins} minutes.")
    except (AuthenticationError, AuthConfigError) as e:
        print_error(str(e))


async def logout_handler() -> None:
    """Handle logout command."""
    if logout():
        print_success("Successfully logged out. Token cache cleared.")
    else:
        print_info("Not currently logged in.")


async def status_handler() -> None:
    """Handle auth status command."""

    info = get_token_info()

    if info is None:
        console.print("[yellow]Not logged in[/yellow]")
        console.print(f"[dim]Provider configured: {cli_settings.auth_provider}[/dim]")
        console.print("[dim]Use 'auth login --method interactive' to authenticate.[/dim]")
        return

    # Calculate expiration
    expires_in = info["expires_in_seconds"]
    mins, secs = divmod(expires_in, 60)
    hours, mins = divmod(mins, 60)

    if hours > 0:
        time_str = f"{hours}h {mins}m"
    elif mins > 0:
        time_str = f"{mins}m {secs}s"
    else:
        time_str = f"{secs}s"

    expires_at = datetime.fromtimestamp(info["expires_at"])

    console.print("[green]Logged in[/green]")
    console.print(f"  Provider: [cyan]{info['provider']}[/cyan]")
    console.print(f"  Expires in: [cyan]{time_str}[/cyan]")
    console.print(f"  Expires at: [dim]{expires_at.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")


login_command = Command(
    name="login",
    description="Authenticate with the admin API",
    handler=login_handler,
    args=[
        CommandArg(
            name="method",
            description="Authentication method",
            required=False,
            default="interactive",
            choices=["interactive", "system_user"],
        ),
    ],
)

logout_command = Command(
    name="logout",
    description="Clear cached authentication token",
    handler=logout_handler,
)

status_command = Command(
    name="status",
    description="Show current authentication status",
    handler=status_handler,
)

auth_group = CommandGroup(
    name="auth",
    description="Authentication management",
)
auth_group.add_command(login_command)
auth_group.add_command(logout_command)
auth_group.add_command(status_command)
