from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    """
    Settings for logging
    """

    model_config = SettingsConfigDict(env_prefix="log_")

    level: str = Field("INFO", description="Logging level")
    level_openai: str = Field("WARNING", description="OpenAI logging level")
    level_uvicorn: str = Field("INFO", description="Uvicorn logging level")
    level_httpcore: str = Field("WARNING", description="HTTPCore logging level")
    format: str = Field(
        "%(levelprefix)s | %(asctime)s | %(process)d | %(name)s | %(message)s",
        description="Logging format",
    )
    date_format: str = Field("%Y-%m-%d %H:%M:%S", description="Logging date format")
    multiline_mode_enabled: bool = Field(False, description="Enable multiline logging mode")
