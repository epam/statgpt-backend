"""Base authentication provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from statgpt.cli.shared.settings import CLISettings


@dataclass
class AuthResult:
    """Result of a successful authentication."""

    access_token: str
    expires_in: int  # Token lifetime in seconds
    refresh_token: str | None = None


class AuthenticationError(Exception):
    """Raised when authentication fails."""


class AuthConfigError(AuthenticationError):
    """Raised when authentication configuration is invalid or incomplete."""

    def __init__(self, provider: str, missing_vars: list[str]):
        self.provider = provider
        self.missing_vars = missing_vars
        vars_list = "\n  - ".join(missing_vars)
        super().__init__(
            f"Authentication provider '{provider}' is not configured.\n"
            f"Missing environment variables:\n  - {vars_list}"
        )


class AuthProvider(ABC):
    """Abstract base class for authentication providers.

    To implement a new provider:
    1. Create a new module in statgpt/cli/shared/auth/
    2. Implement this interface
    3. Register the provider in __init__.py

    Example:
        class KeycloakProvider(AuthProvider):
            name = "keycloak"

            def validate_config(self, settings, interactive: bool) -> None:
                ...

            def interactive_login(self, settings) -> str:
                ...

            def system_user_login(self, settings) -> str:
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'azure', 'keycloak').

        This should match the value used in STATGPT_CLI_AUTH_PROVIDER.
        """

    @abstractmethod
    def validate_config(self, settings: "CLISettings", interactive: bool) -> None:
        """Validate that required configuration is present.

        Args:
            settings: CLI settings instance
            interactive: True if interactive login, False if system user login

        Raises:
            AuthConfigError: If required settings are missing
        """

    @abstractmethod
    def interactive_login(self, settings: "CLISettings") -> AuthResult:
        """Perform interactive login (browser-based).

        Args:
            settings: CLI settings instance

        Returns:
            AuthResult with access token and expiration

        Raises:
            AuthenticationError: If authentication fails
        """

    @abstractmethod
    def system_user_login(self, settings: "CLISettings") -> AuthResult:
        """Perform system user login (non-interactive).

        Args:
            settings: CLI settings instance

        Returns:
            AuthResult with access token and expiration

        Raises:
            AuthenticationError: If authentication fails
        """

    def refresh_token(self, settings: "CLISettings", refresh_token: str) -> AuthResult:
        """Refresh an access token using a refresh token.

        Override this method to support token refresh. Default implementation
        raises AuthenticationError indicating refresh is not supported.

        Args:
            settings: CLI settings instance
            refresh_token: The refresh token to use

        Returns:
            AuthResult with new access token and expiration

        Raises:
            AuthenticationError: If refresh fails or is not supported
        """
        raise AuthenticationError(f"Token refresh not supported by {self.name} provider")
