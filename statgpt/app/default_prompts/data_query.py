import os

from statgpt.common.schemas import DefaltPromptsBase, SystemUserPrompt


class DataQueryDefaultPrompts(DefaltPromptsBase):
    """
    Default statgpt prompts.
    Have the lowest priority, used if no other prompts are provided.

    NOTE: can rename to DataQueryPrompts, move to `common` and reuse in data query tool config model.
    """

    datetime_prompt: str
    group_expander_prompt: str
    group_expander_fallback_prompt: str
    normalization_prompt: str
    named_entities_prompt: str
    indicators_selection_system_prompt: str
    validation_system_prompt: str
    validation_user_prompt: str
    dataset_selection_prompt: SystemUserPrompt
    incomplete_queries_prompt: str
    summarize_queries_prompt: str


yaml_fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "data_query.yaml")
data_query_default_prompts = DataQueryDefaultPrompts.from_yaml(fp=yaml_fp)
