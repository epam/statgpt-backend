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


APP_SETTINGS = AppSettings()
