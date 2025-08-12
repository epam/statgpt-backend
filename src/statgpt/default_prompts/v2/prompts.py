import os

from common.utils.files import read_yaml


class DefaultPrompts:
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
    DATASET_SELECTION_PROMPT: str
    INDICATORS_SELECTION_SYSTEM_PROMPT: str
    VALIDATION_SYSTEM_PROMPT: str
    VALIDATION_USER_PROMPT: str
    INCOMPLETE_QUERIES_PROMPT: str


def load_prompts():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(script_dir, "assets")
    prompts_fp = os.path.join(config_dir, "v2_query_builder.yaml")
    dimension_config = read_yaml(prompts_fp)

    DefaultPrompts.DATETIME_PROMPT = dimension_config["datetimePrompt"]
    DefaultPrompts.GROUP_EXPANDER_PROMPT = dimension_config["groupExpanderPrompt"]
    DefaultPrompts.GROUP_EXPANDER_FALLBACK_PROMPT = dimension_config["groupExpanderFallbackPrompt"]
    DefaultPrompts.NORMALIZATION_PROMPT = dimension_config["normalizationPrompt"]
    DefaultPrompts.NAMED_ENTITIES_PROMPT = dimension_config["namedEntitiesPrompt"]
    DefaultPrompts.DATASET_SELECTION_PROMPT = dimension_config["datasetSelectionPrompt"]
    DefaultPrompts.INDICATORS_SELECTION_SYSTEM_PROMPT = dimension_config[
        "indicatorsSelectionSystemPrompt"
    ]
    DefaultPrompts.VALIDATION_SYSTEM_PROMPT = dimension_config["validationSystemPrompt"]
    DefaultPrompts.VALIDATION_USER_PROMPT = dimension_config["validationUserPrompt"]
    DefaultPrompts.INCOMPLETE_QUERIES_PROMPT = dimension_config["incompleteQueriesPrompt"]


load_prompts()
