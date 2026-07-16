import pytest

from statgpt.common.config import LLMModelsEnum, ReasoningEffortEnum, VerbosityEnum
from statgpt.common.schemas.model_config import LLMModelConfig


class TestLLMModelConfigValidation:
    @pytest.mark.parametrize(
        "deployment, seed, reasoning_effort, temperature, verbosity, should_raise",
        [
            # Valid non-GPT-5 configs
            (LLMModelsEnum.GPT_4_O_2024_11_20, 42, None, 0.5, None, False),
            (LLMModelsEnum.GPT_4_O_MINI_2024_07_18, None, None, 0, None, False),
            (LLMModelsEnum.GPT_4_1_2025_04_14, 42, None, 0, None, False),
            # Valid GPT-5 configs
            (LLMModelsEnum.GPT_5_MINI_2025_08_07, None, ReasoningEffortEnum.MEDIUM, 1, None, False),
            (
                LLMModelsEnum.GPT_5_MINI_2025_08_07,
                None,
                ReasoningEffortEnum.LOW,
                1,
                VerbosityEnum.HIGH,
                False,
            ),
            (LLMModelsEnum.GPT_5_1_2025_11_13, None, ReasoningEffortEnum.NONE, 0.5, None, False),
            (
                LLMModelsEnum.GPT_5_1_2025_11_13,
                None,
                ReasoningEffortEnum.HIGH,
                1,
                VerbosityEnum.LOW,
                False,
            ),
            (LLMModelsEnum.GPT_5_2_2025_12_11, None, ReasoningEffortEnum.NONE, 0, None, False),
            (
                LLMModelsEnum.GPT_5_6_TERRA_2026_07_09,
                None,
                ReasoningEffortEnum.NONE,
                0,
                None,
                False,
            ),
            (
                LLMModelsEnum.GPT_5_6_TERRA_2026_07_09_REASONING,
                None,
                ReasoningEffortEnum.MEDIUM,
                1,
                None,
                False,
            ),
            # GPT-5: seed not supported
            (LLMModelsEnum.GPT_5_MINI_2025_08_07, 42, ReasoningEffortEnum.MEDIUM, 1, None, True),
            (LLMModelsEnum.GPT_5_1_2025_11_13, 42, ReasoningEffortEnum.MEDIUM, 1, None, True),
            # GPT-5: reasoning_effort required
            (LLMModelsEnum.GPT_5_MINI_2025_08_07, None, None, 1, None, True),
            (LLMModelsEnum.GPT_5_1_2025_11_13, None, None, 1, None, True),
            (LLMModelsEnum.GPT_5_6_LUNA_2026_07_09_REASONING, None, None, 1, None, True),
            # GPT-5 Mini: reasoning_effort=none not supported
            (LLMModelsEnum.GPT_5_MINI_2025_08_07, None, ReasoningEffortEnum.NONE, 0.5, None, True),
            # '-reasoning' deployments: reasoning_effort=none not supported
            (
                LLMModelsEnum.GPT_5_6_TERRA_2026_07_09_REASONING,
                None,
                ReasoningEffortEnum.NONE,
                0,
                None,
                True,
            ),
            (
                LLMModelsEnum.GPT_5_6_LUNA_2026_07_09_REASONING,
                None,
                ReasoningEffortEnum.NONE,
                1,
                None,
                True,
            ),
            # GPT-5: temperature must be 1 when reasoning enabled
            (
                LLMModelsEnum.GPT_5_MINI_2025_08_07,
                None,
                ReasoningEffortEnum.MEDIUM,
                0.5,
                None,
                True,
            ),
            (LLMModelsEnum.GPT_5_1_2025_11_13, None, ReasoningEffortEnum.HIGH, 0, None, True),
            # Non-GPT-5: reasoning_effort not supported
            (LLMModelsEnum.GPT_4_O_2024_11_20, None, ReasoningEffortEnum.MEDIUM, 0.5, None, True),
            # Non-GPT-5: verbosity not supported
            (LLMModelsEnum.GPT_4_O_2024_11_20, None, None, 0.5, VerbosityEnum.MEDIUM, True),
        ],
    )
    def test_llm_model_config_validation(
        self, deployment, seed, reasoning_effort, temperature, verbosity, should_raise
    ):
        kwargs = {
            "deployment": deployment,
            "seed": seed,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "verbosity": verbosity,
        }
        if should_raise:
            with pytest.raises(ValueError):
                LLMModelConfig(**kwargs)
        else:
            config = LLMModelConfig(**kwargs)
            assert config.deployment is deployment
            assert config.reasoning_effort is reasoning_effort
