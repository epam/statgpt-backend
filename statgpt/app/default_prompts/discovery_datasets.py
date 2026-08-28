import os

from statgpt.common.schemas import DefaltPromptsBase, SystemUserPrompt


class DiscoveryDatasetsDefaultPrompts(DefaltPromptsBase):
    relevance_prompt: SystemUserPrompt


yaml_fp = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "assets", "discovery_datasets.yaml"
)
discovery_datasets_default_prompts = DiscoveryDatasetsDefaultPrompts.from_yaml(fp=yaml_fp)
