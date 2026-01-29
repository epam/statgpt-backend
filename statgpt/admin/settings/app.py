from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Settings for the admin backend application."""

    model_config = SettingsConfigDict()

    otel_service_name: str = Field(
        default="statgpt-admin-backend",
        alias="OTEL_APP_SERVICE_NAME",
        description="OpenTelemetry service name",
    )
    beta_mcp_enabled: bool = Field(
        default=False,
        alias="BETA_MCP_ENABLED",
        description="Flag to enable the beta MCP features",
    )


APP_SETTINGS = AppSettings()
