import os

from common.schemas.base import SystemUserPrompt
from common.utils.files import read_yaml


class DataQueryDefaultPrompts:
    """
    Default statgpt prompts.
    Have the lowest priority, used if no other prompts are provided.
    """

    # NOTE: fields here MUST have the same name as:
    # - uppercased fields in PromptsConfigV2
    # - uppercased fields in ChannelPromptsV2

    # TODO: can use pydantic BaseModel and:
    # - use it as a constant var instead of a class with constant fields
    # - create it dynamically from ChannelPromptsV2 fields

    DATETIME_PROMPT: str
    GROUP_EXPANDER_PROMPT: str
    GROUP_EXPANDER_FALLBACK_PROMPT: str
    NORMALIZATION_PROMPT: str
    NAMED_ENTITIES_PROMPT: str
    DATASET_SELECTION_PROMPTS: SystemUserPrompt
    INDICATORS_SELECTION_SYSTEM_PROMPT: str
    VALIDATION_SYSTEM_PROMPT: str
    VALIDATION_USER_PROMPT: str
    INCOMPLETE_QUERIES_PROMPT: str
    SUMMARIZE_QUERIES_PROMPT: str


def load_prompts():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(script_dir, "assets")
    prompts_fp = os.path.join(config_dir, "data_query.yaml")
    dimension_config = read_yaml(prompts_fp)

    DataQueryDefaultPrompts.DATETIME_PROMPT = dimension_config["datetimePrompt"]
    DataQueryDefaultPrompts.GROUP_EXPANDER_PROMPT = dimension_config["groupExpanderPrompt"]
    DataQueryDefaultPrompts.GROUP_EXPANDER_FALLBACK_PROMPT = dimension_config[
        "groupExpanderFallbackPrompt"
    ]
    DataQueryDefaultPrompts.NORMALIZATION_PROMPT = dimension_config["normalizationPrompt"]
    DataQueryDefaultPrompts.NAMED_ENTITIES_PROMPT = dimension_config["namedEntitiesPrompt"]
    DataQueryDefaultPrompts.DATASET_SELECTION_PROMPTS = SystemUserPrompt(
        system_message=dimension_config["datasetSelectionPrompts"]["systemMessage"],
        user_message=dimension_config["datasetSelectionPrompts"]["userMessage"],
    )
    DataQueryDefaultPrompts.INDICATORS_SELECTION_SYSTEM_PROMPT = dimension_config[
        "indicatorsSelectionSystemPrompt"
    ]
    DataQueryDefaultPrompts.VALIDATION_SYSTEM_PROMPT = dimension_config["validationSystemPrompt"]
    DataQueryDefaultPrompts.VALIDATION_USER_PROMPT = dimension_config["validationUserPrompt"]
    DataQueryDefaultPrompts.INCOMPLETE_QUERIES_PROMPT = dimension_config["incompleteQueriesPrompt"]
    DataQueryDefaultPrompts.SUMMARIZE_QUERIES_PROMPT = dimension_config["summarizeQueriesPrompt"]


load_prompts()
