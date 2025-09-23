from typing import Optional

from langchain import globals as lc_globals
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.config.llm_models import EmbeddingModelsEnum, LLMModelsEnum


class LangChainSettings(BaseSettings):
    """
    LangChain configuration settings
    """

    model_config = SettingsConfigDict(env_prefix="langchain_")

    # Default chat completion and embeddings settings
    embedding_default_model: EmbeddingModelsEnum = Field(
        default=EmbeddingModelsEnum.TEXT_EMBEDDING_3_LARGE,
        description="Default embeddings model",
    )

    default_model: LLMModelsEnum = Field(
        default=LLMModelsEnum.GPT_4_1_2025_04_14,
        description="Default LLM model for LangChain",
    )

    default_temperature: float = Field(
        default=0.0,
        description="Default temperature for LLM",
    )

    default_api_version: str = Field(
        default="2024-08-01-preview",
        description="Default API version for Azure OpenAI",
    )

    default_seed: Optional[int] = Field(
        default=None,
        description="Default seed for reproducible outputs",
    )

    # Debugging settings
    verbose: bool = Field(default=False, description="Enable verbose mode for LangChain")

    debug: bool = Field(default=False, description="Enable debug mode for LangChain")

    use_custom_logger_callback: bool = Field(
        default=False,
        description="Use custom logger callback for LangChain",
    )

    def configure(self):
        lc_globals.set_verbose(self.verbose)
        lc_globals.set_debug(self.debug)


# Create singleton instance
langchain_settings = LangChainSettings()
langchain_settings.configure()
