from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackgroundTasksSettings(BaseSettings):
    """
    Settings for background tasks
    """

    model_config = SettingsConfigDict(env_prefix="background_tasks_")

    max_concurrent: int = Field(5, description="Maximum number of concurrent background tasks")
    task_timeout: float | None = Field(
        3600.0, description="Timeout in seconds for a single background task (None to disable)"
    )
