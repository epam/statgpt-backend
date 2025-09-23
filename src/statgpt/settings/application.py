from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppMode(str, Enum):
    LOCAL = "LOCAL"
    DIAL = "DIAL"


class ApplicationSettings(BaseSettings):
    """
    Application mode and debug settings
    """

    model_config = SettingsConfigDict(env_prefix="app_")

    mode: AppMode = Field(default=AppMode.DIAL, description="Application mode (LOCAL or DIAL)")

    memory_debug: bool = Field(default=False, description="Enable memory debugging")


# Create singleton instance
application_settings = ApplicationSettings()
