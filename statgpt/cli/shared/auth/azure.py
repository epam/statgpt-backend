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
    from statgpt.cli.settings import CLISettings

DEFAULT_EXPIRES_IN = 3600


class AzureEntraIDProvider(AuthProvider):
    """Authentication provider for Azure Entra ID using MSAL.

    Supports two authentication flows:

    1. Interactive login (for developers):
       - Uses MSAL PublicClientApplication
       - Opens browser for user authentication
       - No client_secret required

    2. System user login (for CI/CD - Machine-to-Machine):
       - Uses MSAL ConfidentialClientApplication with Client Credentials Grant
       - Requires client_secret
       - No browser interaction

    Required settings for interactive login:
        - auth_azure_client_id
        - auth_azure_authority
        - auth_azure_scope

    Additional settings for M2M (system_user) login:
        - auth_azure_client_secret
    """

    @property
    def name(self) -> str:
        return "azure"

    def _validate_base_config(self, settings: "CLISettings") -> None:
        """Validate base Azure Entra ID configuration required for all operations."""
        missing = []
        if not settings.auth_azure_client_id:
            missing.append("STATGPT_CLI_AUTH_AZURE_CLIENT_ID")
        if not settings.auth_azure_authority:
            missing.append("STATGPT_CLI_AUTH_AZURE_AUTHORITY")
        if not settings.auth_azure_scope:
            missing.append("STATGPT_CLI_AUTH_AZURE_SCOPE")
        if missing:
            raise AuthConfigError(self.name, missing)

    def validate_config(self, settings: "CLISettings", interactive: bool) -> None:
        """Validate Azure Entra ID configuration."""
        self._validate_base_config(settings)

        if not interactive:
            if not settings.auth_azure_client_secret:
                raise AuthConfigError(self.name, ["STATGPT_CLI_AUTH_AZURE_CLIENT_SECRET"])

    def _parse_token_response(self, result: dict) -> AuthResult:
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

        return self._parse_token_response(result)

    def system_user_login(self, settings: "CLISettings") -> AuthResult:
        """Perform Azure Entra ID M2M login using Client Credentials Grant."""
        self.validate_config(settings, interactive=False)

        app = msal.ConfidentialClientApplication(
            client_id=settings.auth_azure_client_id,
            authority=settings.auth_azure_authority,
            client_credential=settings.auth_azure_client_secret,
        )

        result = app.acquire_token_for_client(
            scopes=[settings.auth_azure_scope],  # type: ignore[list-item]
        )

        if result is None:
            raise AuthenticationError("Azure Entra ID authentication failed: No response received")

        return self._parse_token_response(result)

    def refresh_token(self, settings: "CLISettings", refresh_token: str) -> AuthResult:
        """Refresh an access token using a refresh token."""
        self._validate_base_config(settings)

        app = msal.PublicClientApplication(
            client_id=settings.auth_azure_client_id,
            authority=settings.auth_azure_authority,
        )

        result = app.acquire_token_by_refresh_token(
            refresh_token=refresh_token,
            scopes=[settings.auth_azure_scope],  # type: ignore[list-item]
        )

        return self._parse_token_response(result)
