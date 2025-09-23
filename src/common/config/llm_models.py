import os
from enum import StrEnum


class EmbeddingModelsEnum(StrEnum):
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"


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

    @property
    def deployment_id(self) -> str:
        return os.getenv(f"LLM_MODELS_{self.name}", self.value)
