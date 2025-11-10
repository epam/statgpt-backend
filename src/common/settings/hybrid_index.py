from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HybridIndexSettings(BaseSettings):
    """
    Settings for hybrid indexer
    """

    model_config = SettingsConfigDict(env_prefix="hybrid_index_")

    concurrency_limit: int = Field(default=20, description="Maximum concurrency of hybrid indexing")
