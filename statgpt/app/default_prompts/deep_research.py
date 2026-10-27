import os

from statgpt.common.schemas import DefaltPromptsBase


class DeepResearchDefaultPrompts(DefaltPromptsBase):
    query_check_prompt: str
    default_excluded_topics: list[str]


yaml_fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "deep_research.yaml")
deep_research_default_prompts = DeepResearchDefaultPrompts.from_yaml(fp=yaml_fp)
