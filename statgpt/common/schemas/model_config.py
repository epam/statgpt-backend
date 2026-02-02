from pydantic import Field

from statgpt.common.config import (
    EmbeddingModelsEnum,
    LLMModelsEnum,
    ReasoningEffortEnum,
    VerbosityEnum,
)
from statgpt.common.settings.langchain import langchain_settings

from .base import BaseYamlModel


class BaseModelConfig(BaseYamlModel):
    """Base config for LLM and embeddings models configs."""

    api_version: str = Field(
        default=langchain_settings.default_api_version,
        description="API version for the model",
    )


class EmbeddingsModelConfig(BaseModelConfig):
    """Config for embeddings models."""

    deployment: EmbeddingModelsEnum = Field(
        default=langchain_settings.embedding_default_model,
        description="The deployment of the model in DIAL",
    )


class LLMModelConfig(BaseModelConfig):
    """Config for LLM models."""

    deployment: LLMModelsEnum = Field(
        default=langchain_settings.default_model,
        description="The deployment of the model in DIAL",
    )
    temperature: float = Field(
        default=langchain_settings.default_temperature,
        description=(
            "The temperature of the model. 0.0 means deterministic output, higher values mean more"
            " randomness. Note: Not supported by GPT-5 models."
        ),
    )
    seed: int | None = Field(
        default=langchain_settings.default_seed,
        description=(
            "The seed of the model. If set, the model will produce the same output for the same input. "
            "Note: Not supported by GPT-5 models."
        ),
    )
    reasoning_effort: ReasoningEffortEnum | None = Field(
        default=langchain_settings.default_reasoning_effort,
        description=(
            "Reasoning effort level for GPT-5 models. Ignored for non-reasoning models. "
            "Supports: none, minimal, low, medium, high, xhigh."
        ),
    )
    verbosity: VerbosityEnum | None = Field(
        default=langchain_settings.default_verbosity,
        description=("Output verbosity for GPT-5 models (low/medium/high). "),
    )
