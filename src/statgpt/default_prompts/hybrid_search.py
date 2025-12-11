import os

from common.schemas import DefaltPromptsBase, SystemUserPrompt


class HybridSearchDefaultPrompts(DefaltPromptsBase):
    normalization_prompt: SystemUserPrompt
    separate_subjects_prompt: SystemUserPrompt
    relevancy_prompt: SystemUserPrompt


yaml_fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "hybrid_search.yaml")
hybrid_search_default_prompts = HybridSearchDefaultPrompts.from_yaml(fp=yaml_fp)
