from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.config import LLMModelsConfig


class HybridIndexSettings(BaseSettings):
    """
    Settings for hybrid indexer
    """

    model_config = SettingsConfigDict(env_prefix="hybrid_index_")

    concurrency_limit: int = Field(20, description="Maximum concurrency of hybrid indexing")
    normalize_model: str = Field(
        LLMModelsConfig.GPT_4_1_MINI_2025_04_14, description="LLM Model used for normalization"
    )
    harmonize_model: str = Field(
        LLMModelsConfig.GPT_4_1_MINI_2025_04_14, description="LLM Model used for harmonization"
    )
