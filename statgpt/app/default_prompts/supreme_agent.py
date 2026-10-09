import os

from statgpt.common.schemas import DefaltPromptsBase


class SupremeAgentDefaultPrompts(DefaltPromptsBase):
    """
    Default prompt for Supreme Agent.
    Has the lowest priority, used if no other prompts are found.
    """

    system_prompt: str
    additional_context_wrapper_section: str
    default_general_section: str
    default_tool_usage_section: str
    default_no_calculations_section: str
    default_data_presentation_section: str
    deep_research_section: str
    deep_research_enabled_message_to_agent: str
    deep_research_disabled_message_to_agent: str
    deep_research_report_delivered_message_to_agent: str
    default_user_ui_context_section: str


fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "supreme_agent.yaml")
supreme_agent_default_prompts = SupremeAgentDefaultPrompts.from_yaml(fp=fp)
