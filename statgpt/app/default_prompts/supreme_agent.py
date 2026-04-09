import os

from statgpt.common.schemas import DefaltPromptsBase


class SupremeAgentDefaultPrompts(DefaltPromptsBase):
    """
    Default prompt for Supreme Agent.
    Has the lowest priority, used if no other prompts are found.
    """

    system_prompt: str
    additional_context_wrapper_section: str
    default_general: str
    default_no_calculations_and_analytics: str


fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "supreme_agent.yaml")
supreme_agent_default_prompts = SupremeAgentDefaultPrompts.from_yaml(fp=fp)
