"""CLI Settings module with environment variable configuration."""

import os
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingsSection(StrEnum):
    """Settings sections for grouping in the settings command."""

    ADMIN_API = "Admin API"
    CONTENT = "Content"
    DIAL = "DIAL"
    AUTHENTICATION = "Authentication"
    AZURE = "Azure Entra ID"
    KEYCLOAK = "Keycloak"
    AUTH0 = "Auth0"
    GENERAL = "General"


class FieldMeta(BaseModel):
    """Metadata for CLI settings fields."""

    section: SettingsSection
    secret: bool = False


def field_meta(section: SettingsSection, secret: bool = False) -> dict[str, Any]:
    """Create field metadata dict for json_schema_extra."""
    return FieldMeta(section=section, secret=secret).model_dump()


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

    admin_url: str = Field(
        default="http://localhost:8000",
        description="URL of the StatGPT Admin API",
        json_schema_extra=field_meta(SettingsSection.ADMIN_API),
    )

    config_dir: str | None = Field(
        default=None,
        description="Path to the configuration directory for content initialization",
        json_schema_extra=field_meta(SettingsSection.CONTENT),
    )

    dial_url: str | None = Field(
        default=None,
        description="DIAL URL for file uploads",
        json_schema_extra=field_meta(SettingsSection.DIAL),
    )
    dial_api_key: str | None = Field(
        default=None,
        description="DIAL API key for file uploads",
        json_schema_extra=field_meta(SettingsSection.DIAL, secret=True),
    )

    max_embeddings: int | None = Field(
        default=None,
        description="Maximum number of embeddings for reindex operations",
        json_schema_extra=field_meta(SettingsSection.CONTENT),
    )

    @field_validator("max_embeddings", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        """Convert empty strings to None for optional int fields."""
        if v == "":
            return None
        return v

    auth_provider: str | None = Field(
        default=None,
        description="Authentication provider to use (azure, keycloak, auth0). Optional for local development.",
        json_schema_extra=field_meta(SettingsSection.AUTHENTICATION),
    )
    auth_callback_port: int | None = Field(
        default=None,
        description="Fixed port for OAuth callback server (use when provider requires exact redirect URI)",
        json_schema_extra=field_meta(SettingsSection.AUTHENTICATION),
    )

    auth_azure_client_id: str | None = Field(
        default=None,
        description="Azure Entra ID client/application ID",
        json_schema_extra=field_meta(SettingsSection.AZURE),
    )
    auth_azure_authority: str | None = Field(
        default=None,
        description="Azure Entra ID authority URL",
        json_schema_extra=field_meta(SettingsSection.AZURE),
    )
    auth_azure_scope: str | None = Field(
        default=None,
        description="Azure Entra ID scope for token request",
        json_schema_extra=field_meta(SettingsSection.AZURE),
    )
    auth_azure_client_secret: str | None = Field(
        default=None,
        description="Azure Entra ID client secret (required for M2M/CI authentication)",
        json_schema_extra=field_meta(SettingsSection.AZURE, secret=True),
    )

    auth_keycloak_server_url: str | None = Field(
        default=None,
        description="Keycloak server URL",
        json_schema_extra=field_meta(SettingsSection.KEYCLOAK),
    )
    auth_keycloak_realm: str | None = Field(
        default=None,
        description="Keycloak realm name",
        json_schema_extra=field_meta(SettingsSection.KEYCLOAK),
    )
    auth_keycloak_client_id: str | None = Field(
        default=None,
        description="Keycloak client ID",
        json_schema_extra=field_meta(SettingsSection.KEYCLOAK),
    )
    auth_keycloak_client_secret: str | None = Field(
        default=None,
        description="Keycloak client secret (required for M2M/CI authentication)",
        json_schema_extra=field_meta(SettingsSection.KEYCLOAK, secret=True),
    )
    auth_keycloak_scope: str | None = Field(
        default=None,
        description="OAuth scope for Keycloak (default: openid)",
        json_schema_extra=field_meta(SettingsSection.KEYCLOAK),
    )

    auth_auth0_domain: str | None = Field(
        default=None,
        description="Auth0 tenant domain (e.g., mytenant.us.auth0.com)",
        json_schema_extra=field_meta(SettingsSection.AUTH0),
    )
    auth_auth0_client_id: str | None = Field(
        default=None,
        description="Auth0 application client ID",
        json_schema_extra=field_meta(SettingsSection.AUTH0),
    )
    auth_auth0_client_secret: str | None = Field(
        default=None,
        description="Auth0 client secret (required for M2M/CI authentication)",
        json_schema_extra=field_meta(SettingsSection.AUTH0, secret=True),
    )
    auth_auth0_audience: str | None = Field(
        default=None,
        description="Auth0 API identifier (audience)",
        json_schema_extra=field_meta(SettingsSection.AUTH0),
    )
    auth_auth0_scope: str | None = Field(
        default=None,
        description="OAuth scope for Auth0 (default: openid profile email offline_access)",
        json_schema_extra=field_meta(SettingsSection.AUTH0),
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level for CLI operations",
        json_schema_extra=field_meta(SettingsSection.GENERAL),
    )

    data_dir: str | None = Field(
        default=None,
        description="Directory for CLI data (token cache, history). Defaults to ~/.statgpt",
        json_schema_extra=field_meta(SettingsSection.GENERAL),
    )

    @property
    def cli_data_dir(self) -> str:
        """Get the CLI data directory path."""
        if self.data_dir:
            return os.path.expanduser(self.data_dir)
        return os.path.expanduser("~/.statgpt")

    def get_setting_source(self, field_name: str) -> str:
        """Determine the source of a setting value."""
        env_var = f"STATGPT_CLI_{field_name.upper()}"
        if os.environ.get(env_var) is not None:
            return "env"

        field_info = CLISettings.model_fields.get(field_name)
        if field_info is None:
            return "unknown"

        value = getattr(self, field_name)
        if value is None:
            return "not set"

        default = field_info.default
        if default is not None and value == default:
            return "default"

        return "env"

    def get_auth_settings_for_provider(self, provider: str) -> dict[str, str | None]:
        """Get all auth settings for a specific provider."""
        prefix = f"auth_{provider}_"
        return {
            name[len(prefix) :]: getattr(self, name)
            for name in CLISettings.model_fields
            if name.startswith(prefix)
        }


cli_settings = CLISettings()


class CLIRuntimeState:
    """Runtime state for CLI session (not persisted).

    This is intentionally mutable global state to track session-wide flags
    that need to be accessible across the CLI codebase without threading
    through function parameters.

    Attributes:
        non_interactive: When True, interactive prompts raise NonInteractiveError
            instead of displaying UI selectors.
        debug: When True, full stack traces are shown on errors.
    """

    non_interactive: bool = False
    debug: bool = False


cli_runtime = CLIRuntimeState()
