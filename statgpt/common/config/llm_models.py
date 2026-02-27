import os
from enum import StrEnum


class EmbeddingModelsEnum(StrEnum):
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"


class ReasoningEffortEnum(StrEnum):
    """Reasoning effort levels for GPT-5 models."""

    NONE = "none"
    """No reasoning mode - standard inference."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class VerbosityEnum(StrEnum):
    """Output verbosity levels for GPT-5 models."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LLMModelsEnum(StrEnum):
    # Gemini models
    GEMINI_2_0_FLASH_LITE_001 = "gemini-2.0-flash-lite-001"

    # Legacy models
    # These models are added to the enum temporarily
    GPT_4_1106_PREVIEW = "gpt-4-1106-preview"

    # GPT-4 Turbo models
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"

    # GPT-4 Omni models
    GPT_4_O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4_O_2024_08_06 = "gpt-4o-2024-08-06"
    GPT_4_O_2024_11_20 = "gpt-4o-2024-11-20"

    # GPT-4 Omni Mini models
    GPT_4_O_MINI_2024_07_18 = "gpt-4o-mini-2024-07-18"

    # GPT-4.1 models
    GPT_4_1_2025_04_14 = "gpt-4.1-2025-04-14"
    GPT_4_1_MINI_2025_04_14 = "gpt-4.1-mini-2025-04-14"
    GPT_4_1_NANO_2025_04_14 = "gpt-4.1-nano-2025-04-14"

    GPT_4_1_2025_04_14_HF = "gpt-4.1-2025-04-14-hf"
    """GPT-4.1 models with high content filters."""

    # GPT-5 models
    GPT_5_MINI_2025_08_07 = "gpt-5-mini-2025-08-07"
    GPT_5_1_2025_11_13 = "gpt-5.1-2025-11-13"
    GPT_5_2_2025_12_11 = "gpt-5.2-2025-12-11"

    @property
    def deployment_id(self) -> str:
        return os.getenv(f"LLM_MODELS_{self.name}", self.value)

    @property
    def is_gpt_41_family(self) -> bool:
        """Check if the model belongs to the GPT-4.1 family."""
        return self in {
            LLMModelsEnum.GPT_4_1_2025_04_14,
            LLMModelsEnum.GPT_4_1_MINI_2025_04_14,
            LLMModelsEnum.GPT_4_1_NANO_2025_04_14,
        }

    @property
    def is_gpt_5_family(self) -> bool:
        """Check if the model belongs to the GPT-5 family."""
        return self in {
            LLMModelsEnum.GPT_5_MINI_2025_08_07,
            LLMModelsEnum.GPT_5_1_2025_11_13,
            LLMModelsEnum.GPT_5_2_2025_12_11,
        }
