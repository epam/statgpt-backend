import os

from statgpt.common.schemas import DefaltPromptsBase, SystemUserPrompt


class DiscoveryIndexingDefaultPrompts(DefaltPromptsBase):
    chunk_summary_prompt: SystemUserPrompt
    merge_summary_prompt: SystemUserPrompt
    shared_unit_instruction: str
    """Given as `{units_instruction}` when every indicator shares a unit, formatted with `{unit}`."""
    per_indicator_units_instruction: str
    """Given as `{units_instruction}` when the indicators keep their own units."""


yaml_fp = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "assets", "discovery_indexing.yaml"
)
discovery_indexing_default_prompts = DiscoveryIndexingDefaultPrompts.from_yaml(fp=yaml_fp)
