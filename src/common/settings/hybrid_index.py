from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.config import LLMModelsEnum
from common.schemas import LLMModelConfig


class HybridIndexHarmonizeModelSettings(BaseSettings, LLMModelConfig):
    model_config = SettingsConfigDict(env_prefix="hybrid_index_harmonize_model_")

    deployment: LLMModelsEnum = Field(
        default=LLMModelsEnum.GPT_4_1_MINI_2025_04_14,
        description="The deployment of the model in DIAL",
        alias="hybrid_index_harmonize_model",
    )


class HybridIndexNormalizeModelSettings(BaseSettings, LLMModelConfig):
    model_config = SettingsConfigDict(env_prefix="hybrid_index_normalize_model_")

    deployment: LLMModelsEnum = Field(
        default=LLMModelsEnum.GPT_4_1_MINI_2025_04_14,
        description="The deployment of the model in DIAL",
        alias="hybrid_index_normalize_model",
    )


class HybridIndexSettings(BaseSettings):
    """
    Settings for hybrid indexer
    """

    model_config = SettingsConfigDict(env_prefix="hybrid_index_")

    concurrency_limit: int = Field(20, description="Maximum concurrency of hybrid indexing")
    normalize_model_config: HybridIndexNormalizeModelSettings = Field(
        description="LLM Model used for normalization",
        default_factory=HybridIndexNormalizeModelSettings,
    )
    harmonize_model_config: HybridIndexHarmonizeModelSettings = Field(
        description="LLM Model used for harmonization",
        default_factory=HybridIndexHarmonizeModelSettings,
    )
    searcher_model_config: LLMModelConfig = Field(
        description="LLM Model used for search",
        default_factory=LLMModelConfig,
    )
