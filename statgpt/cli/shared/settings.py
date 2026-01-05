"""CLI Settings module with environment variable configuration."""

import os
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CLISettings(BaseSettings):
    """StatGPT CLI settings loaded from environment variables.

    All environment variables are prefixed with STATGPT_CLI_.
    Example: STATGPT_CLI_ADMIN_URL=http://localhost:8000
    """

    model_config = SettingsConfigDict(
        env_prefix="STATGPT_CLI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Admin API settings
    admin_url: str = Field(
        default="http://localhost:8000",
        description="URL of the StatGPT Admin API",
    )

    # Content initialization settings
    config_dir: str | None = Field(
        default=None,
        description="Path to the configuration directory for content initialization",
    )

    # DIAL settings (for file uploads during content init)
    dial_url: str | None = Field(
        default=None,
        description="DIAL URL for file uploads",
    )
    dial_api_key: str | None = Field(
        default=None,
        description="DIAL API key for file uploads",
    )

    # Reindex settings
    max_embeddings: int | None = Field(
        default=None,
        description="Maximum number of embeddings for reindex operations",
    )

    # =========================================================================
    # Authentication settings
    # =========================================================================

    # Provider selection
    auth_provider: str = Field(
        default="azure",
        description="Authentication provider to use (azure, keycloak, etc.)",
    )

    # Azure Entra ID settings
    auth_azure_client_id: str | None = Field(
        default=None,
        description="Azure Entra ID client/application ID",
    )
    auth_azure_authority: str | None = Field(
        default=None,
        description="Azure Entra ID authority URL (e.g., https://login.microsoftonline.com/{tenant})",
    )
    auth_azure_scope: str | None = Field(
        default=None,
        description="Azure Entra ID scope for token request",
    )
    auth_azure_client_secret: str | None = Field(
        default=None,
        description="Azure Entra ID client secret (for system user login)",
    )
    auth_azure_username: str | None = Field(
        default=None,
        description="Username for Azure Entra ID system user login",
    )
    auth_azure_password: str | None = Field(
        default=None,
        description="Password for Azure Entra ID system user login",
    )

    # Keycloak settings (for future use)
    # auth_keycloak_server_url: str | None = Field(
    #     default=None,
    #     description="Keycloak server URL",
    # )
    # auth_keycloak_realm: str | None = Field(
    #     default=None,
    #     description="Keycloak realm name",
    # )
    # auth_keycloak_client_id: str | None = Field(
    #     default=None,
    #     description="Keycloak client ID",
    # )
    # ... etc

    # General settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level for CLI operations",
    )

    data_dir: str | None = Field(
        default=None,
        description="Directory for CLI data (token cache, history). Defaults to ~/.statgpt",
    )

    @property
    def cli_data_dir(self) -> str:
        """Get the CLI data directory path.

        Returns:
            Configured data_dir or ~/.statgpt if not set
        """
        if self.data_dir:
            return os.path.expanduser(self.data_dir)
        return os.path.expanduser("~/.statgpt")

    def get_setting_source(self, field_name: str) -> str:
        """Determine the source of a setting value.

        Returns:
            'env' if set via environment variable
            'default' if using default value
            'not set' if None and no default
        """
        env_var = f"STATGPT_CLI_{field_name.upper()}"
        if os.environ.get(env_var) is not None:
            return "env"

        field_info = self.model_fields.get(field_name)
        if field_info is None:
            return "unknown"

        value = getattr(self, field_name)
        if value is None:
            return "not set"

        # Check if it's using the default
        default = field_info.default
        if default is not None and value == default:
            return "default"

        return "env"

    def get_auth_settings_for_provider(self, provider: str) -> dict[str, str | None]:
        """Get all auth settings for a specific provider.

        Args:
            provider: Provider name (e.g., 'azure', 'keycloak')

        Returns:
            Dictionary of setting names to values for that provider
        """
        prefix = f"auth_{provider}_"
        return {
            name[len(prefix) :]: getattr(self, name)
            for name in self.model_fields
            if name.startswith(prefix)
        }


# Global settings instance
cli_settings = CLISettings()
