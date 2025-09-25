from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    """
    Application mode and debug settings
    """

    model_config = SettingsConfigDict(env_prefix="app_")

    memory_debug: bool = Field(default=False, description="Enable memory debugging")
    gc_debug: bool = Field(default=False, description="Enable garbage collection debugging")


# Create singleton instance
application_settings = ApplicationSettings()
