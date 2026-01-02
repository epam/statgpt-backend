"""Shared utilities for StatGPT CLI."""

from statgpt.cli.shared.admin_client import (
    AdminAPIError,
    AdminClient,
    AuthenticationRequired,
    get_admin_client,
)
from statgpt.cli.shared.auth import (
    AuthConfigError,
    AuthenticationError,
    AuthProvider,
    LoginMethod,
    get_auth_headers,
    get_available_providers,
    get_provider,
    login,
    register_provider,
)
from statgpt.cli.shared.console import (
    console,
    create_data_table,
    create_settings_table,
    mask_secret,
    print_banner,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from statgpt.cli.shared.logging import get_logger, setup_logging
from statgpt.cli.shared.settings import CLISettings, cli_settings

__all__ = [
    # Settings
    "CLISettings",
    "cli_settings",
    # Console
    "console",
    "print_banner",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "mask_secret",
    "create_settings_table",
    "create_data_table",
    # Auth
    "AuthProvider",
    "AuthenticationError",
    "AuthConfigError",
    "LoginMethod",
    "login",
    "get_auth_headers",
    "register_provider",
    "get_provider",
    "get_available_providers",
    # Admin Client
    "AdminAPIError",
    "AdminClient",
    "AuthenticationRequired",
    "get_admin_client",
    # Logging
    "setup_logging",
    "get_logger",
]
