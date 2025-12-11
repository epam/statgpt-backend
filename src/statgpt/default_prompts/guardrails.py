import os

from common.schemas import DefaltPromptsBase


class GuardrailsDefaultPrompts(DefaltPromptsBase):
    checker_prompt: str
    general_topics_blacklist: list[str]
    response_prompt: str


yaml_fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "guardrails.yaml")
guardrails_default_prompts = GuardrailsDefaultPrompts.from_yaml(fp=yaml_fp)
