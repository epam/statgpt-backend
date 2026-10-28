import os

from statgpt.common.schemas import DefaltPromptsBase, SystemUserPrompt


class DiscoveryIndexingDefaultPrompts(DefaltPromptsBase):
    chunk_summary_prompt: SystemUserPrompt
    merge_summary_prompt: SystemUserPrompt


yaml_fp = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "assets", "discovery_indexing.yaml"
)
discovery_indexing_default_prompts = DiscoveryIndexingDefaultPrompts.from_yaml(fp=yaml_fp)
