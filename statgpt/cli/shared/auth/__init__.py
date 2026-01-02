"""Authentication module with pluggable provider support.

This module provides a registry-based authentication system that supports
multiple identity providers (Azure Entra ID, Keycloak, etc.).

Usage:
    from statgpt.cli.shared.auth import login, get_auth_headers

    # Get auth headers for HTTP requests (uses cached token if available)
    headers = get_auth_headers("interactive")

    # Or login directly (always performs fresh login)
    result = login("system_user")

    # Check if already logged in
    if is_logged_in():
        print("Already authenticated")

Adding a new provider:
    1. Create a new module (e.g., keycloak.py)
    2. Implement AuthProvider interface
    3. Register it: register_provider("keycloak", KeycloakProvider)
    4. Add settings to CLISettings with auth_keycloak_* prefix
"""

from typing import Literal

from statgpt.cli.shared.auth.azure import AzureEntraIDProvider
from statgpt.cli.shared.auth.base import (
    AuthConfigError,
    AuthenticationError,
    AuthProvider,
    AuthResult,
)
from statgpt.cli.shared.token_cache import token_cache

# Type alias for login methods
LoginMethod = Literal["interactive", "system_user"]

# Provider registry
_providers: dict[str, type[AuthProvider]] = {}


def register_provider(name: str, provider_class: type[AuthProvider]) -> None:
    """Register an authentication provider.

    Args:
        name: Provider identifier (e.g., 'azure', 'keycloak')
        provider_class: Provider class implementing AuthProvider
    """
    _providers[name] = provider_class


def get_provider(name: str) -> AuthProvider:
    """Get an authentication provider instance by name.

    Args:
        name: Provider identifier

    Returns:
        AuthProvider instance

    Raises:
        AuthenticationError: If provider is not registered
    """
    if name not in _providers:
        available = ", ".join(_providers.keys()) or "none"
        raise AuthenticationError(
            f"Unknown authentication provider: '{name}'. " f"Available providers: {available}"
        )
    return _providers[name]()


def get_available_providers() -> list[str]:
    """Get list of registered provider names."""
    return list(_providers.keys())


def is_logged_in() -> bool:
    """Check if there is a valid cached token (or can be refreshed).

    Returns:
        True if a valid (non-expired) token exists or was successfully refreshed
    """
    return get_cached_token() is not None


def get_token_info() -> dict | None:
    """Get information about the current cached token.

    Returns:
        Dictionary with provider and expiration info, or None if not logged in
    """
    return token_cache.get_token_info()


def login(method: LoginMethod, force: bool = False) -> AuthResult:
    """Perform authentication using the configured provider.

    Args:
        method: Either 'interactive' or 'system_user'
        force: If True, always perform fresh login even if cached token exists

    Returns:
        AuthResult with access token and expiration

    Raises:
        AuthenticationError: If authentication fails
        AuthConfigError: If provider is not configured
    """
    from statgpt.cli.shared.console import print_info
    from statgpt.cli.shared.settings import cli_settings

    provider = get_provider(cli_settings.auth_provider)

    if method == "interactive":
        print_info(f"Authenticating via {provider.name} (interactive)...")
        result = provider.interactive_login(cli_settings)
    elif method == "system_user":
        result = provider.system_user_login(cli_settings)
    else:
        raise AuthenticationError(f"Unknown login method: {method}")

    # Cache the token
    token_cache.save_token(
        access_token=result.access_token,
        expires_in=result.expires_in,
        provider=provider.name,
        refresh_token=result.refresh_token,
    )

    return result


def logout() -> bool:
    """Clear the cached authentication token.

    Returns:
        True if a token was cleared, False if no token was cached
    """
    was_logged_in = is_logged_in()
    token_cache.clear()
    return was_logged_in


def _try_refresh_token() -> AuthResult | None:
    """Try to refresh the access token using the cached refresh token.

    Returns:
        AuthResult if refresh successful, None otherwise
    """
    import logging

    from statgpt.cli.shared.settings import cli_settings

    _log = logging.getLogger(__name__)

    cached = token_cache.get_token_raw()
    if not cached or not cached.has_refresh_token():
        return None

    try:
        provider = get_provider(cached.provider)
        _log.debug("Attempting token refresh via %s", provider.name)

        result = provider.refresh_token(cli_settings, cached.refresh_token)  # type: ignore[arg-type]

        # Cache the new token
        token_cache.save_token(
            access_token=result.access_token,
            expires_in=result.expires_in,
            provider=cached.provider,
            refresh_token=result.refresh_token,
        )

        _log.debug("Token refresh successful")
        return result
    except AuthenticationError as e:
        _log.debug("Token refresh failed: %s", e)
        # Clear the cache since refresh failed
        token_cache.clear()
        return None


def get_cached_token() -> str | None:
    """Get the cached access token if valid, attempting refresh if expired.

    If the access token is expired but a refresh token is available,
    this will attempt to refresh the token automatically.

    Returns:
        Access token string if cached and valid (or refreshed), None otherwise
    """
    cached = token_cache.get_token()
    if cached:
        return cached.access_token

    # Token expired or missing - try to refresh
    result = _try_refresh_token()
    if result:
        return result.access_token

    return None


def get_auth_headers(method: LoginMethod | None) -> dict[str, str]:
    """Get authorization headers, using cached token if available.

    Args:
        method: Login method or None for no auth

    Returns:
        Dictionary with Authorization header if method is provided,
        empty dictionary otherwise

    Raises:
        AuthenticationError: If authentication fails
    """
    if method is None:
        return {}

    from statgpt.cli.shared.console import print_error, print_info

    try:
        # Check for cached token first
        cached_token = get_cached_token()
        if cached_token:
            info = get_token_info()
            if info:
                mins = info["expires_in_seconds"] // 60
                print_info(f"Using cached token (expires in {mins} min)")
            return {"Authorization": f"Bearer {cached_token}"}

        # No cached token, perform login
        result = login(method)
        return {"Authorization": f"Bearer {result.access_token}"}
    except (AuthenticationError, AuthConfigError) as e:
        print_error(str(e))
        raise


# Register built-in providers
register_provider("azure", AzureEntraIDProvider)

# Export public API
__all__ = [
    # Base classes
    "AuthProvider",
    "AuthenticationError",
    "AuthConfigError",
    "AuthResult",
    # Registry functions
    "register_provider",
    "get_provider",
    "get_available_providers",
    # Auth functions
    "LoginMethod",
    "login",
    "logout",
    "get_auth_headers",
    "is_logged_in",
    "get_token_info",
    "get_cached_token",
]
