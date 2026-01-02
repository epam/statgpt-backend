"""Azure Entra ID (formerly Azure AD) authentication provider."""

from typing import TYPE_CHECKING

import msal

from statgpt.cli.shared.auth.base import (
    AuthConfigError,
    AuthenticationError,
    AuthProvider,
    AuthResult,
)

if TYPE_CHECKING:
    from statgpt.cli.shared.settings import CLISettings

# Default token lifetime if not provided by MSAL (1 hour)
DEFAULT_EXPIRES_IN = 3600


class AzureEntraIDProvider(AuthProvider):
    """Authentication provider for Azure Entra ID using MSAL.

    Required environment variables for interactive login:
        - STATGPT_CLI_AUTH_AZURE_CLIENT_ID
        - STATGPT_CLI_AUTH_AZURE_AUTHORITY
        - STATGPT_CLI_AUTH_AZURE_SCOPE

    Additional variables for system user login:
        - STATGPT_CLI_AUTH_AZURE_CLIENT_SECRET
        - STATGPT_CLI_AUTH_AZURE_USERNAME
        - STATGPT_CLI_AUTH_AZURE_PASSWORD
    """

    @property
    def name(self) -> str:
        return "azure"

    def validate_config(self, settings: "CLISettings", interactive: bool) -> None:
        """Validate Azure Entra ID configuration."""
        missing = []

        # Required for both login types
        if not settings.auth_azure_client_id:
            missing.append("STATGPT_CLI_AUTH_AZURE_CLIENT_ID")
        if not settings.auth_azure_authority:
            missing.append("STATGPT_CLI_AUTH_AZURE_AUTHORITY")
        if not settings.auth_azure_scope:
            missing.append("STATGPT_CLI_AUTH_AZURE_SCOPE")

        # Additional requirements for system user login
        if not interactive:
            if not settings.auth_azure_client_secret:
                missing.append("STATGPT_CLI_AUTH_AZURE_CLIENT_SECRET")
            if not settings.auth_azure_username:
                missing.append("STATGPT_CLI_AUTH_AZURE_USERNAME")
            if not settings.auth_azure_password:
                missing.append("STATGPT_CLI_AUTH_AZURE_PASSWORD")

        if missing:
            raise AuthConfigError(self.name, missing)

    def _parse_result(self, result: dict) -> AuthResult:
        """Parse MSAL result into AuthResult."""
        if not result.get("access_token"):
            error_desc = result.get("error_description", result.get("error", "Unknown error"))
            raise AuthenticationError(f"Azure Entra ID authentication failed: {error_desc}")

        return AuthResult(
            access_token=result["access_token"],
            expires_in=result.get("expires_in", DEFAULT_EXPIRES_IN),
            refresh_token=result.get("refresh_token"),
        )

    def interactive_login(self, settings: "CLISettings") -> AuthResult:
        """Perform interactive Azure Entra ID login via browser."""
        self.validate_config(settings, interactive=True)

        app = msal.PublicClientApplication(
            client_id=settings.auth_azure_client_id,
            authority=settings.auth_azure_authority,
        )

        result = app.acquire_token_interactive(
            scopes=[settings.auth_azure_scope],  # type: ignore[list-item]
        )

        return self._parse_result(result)

    def system_user_login(self, settings: "CLISettings") -> AuthResult:
        """Perform Azure Entra ID system user login."""
        self.validate_config(settings, interactive=False)

        app = msal.ConfidentialClientApplication(
            client_id=settings.auth_azure_client_id,
            authority=settings.auth_azure_authority,
            client_credential=settings.auth_azure_client_secret,
        )

        result = app.acquire_token_by_username_password(
            username=settings.auth_azure_username,  # type: ignore[arg-type]
            password=settings.auth_azure_password,  # type: ignore[arg-type]
            scopes=[settings.auth_azure_scope],  # type: ignore[list-item]
        )

        return self._parse_result(result)

    def refresh_token(self, settings: "CLISettings", refresh_token: str) -> AuthResult:
        """Refresh an access token using a refresh token.

        Args:
            settings: CLI settings instance
            refresh_token: The refresh token to use

        Returns:
            AuthResult with new access token and expiration

        Raises:
            AuthenticationError: If refresh fails
        """
        # Only need basic config for refresh
        if not settings.auth_azure_client_id:
            raise AuthConfigError(self.name, ["STATGPT_CLI_AUTH_AZURE_CLIENT_ID"])
        if not settings.auth_azure_scope:
            raise AuthConfigError(self.name, ["STATGPT_CLI_AUTH_AZURE_SCOPE"])

        app = msal.PublicClientApplication(
            client_id=settings.auth_azure_client_id,
            authority=settings.auth_azure_authority,
        )

        result = app.acquire_token_by_refresh_token(
            refresh_token=refresh_token,
            scopes=[settings.auth_azure_scope],  # type: ignore[list-item]
        )

        return self._parse_result(result)
