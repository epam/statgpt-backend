from statgpt.app.chains.supreme_agent import SupremeAgent
from statgpt.common.schemas.channel import ChannelConfig, SupremeAgentConfig


def _render_system_prompt(supreme_agent: SupremeAgentConfig) -> str:
    channel_config = ChannelConfig(supreme_agent=supreme_agent)
    template = SupremeAgent._create_prompt_template(channel_config)
    messages = template.format_messages(today_date="2026-08-21")
    return str(messages[0].content)


def test_system_prompt_uses_default_sections():
    prompt = _render_system_prompt(
        SupremeAgentConfig(
            name="StatGPT",
            domain="official statistics",
            terminology_domain="official statistics",
        )
    )

    # Default UI context describes the interactive data widget.
    assert "shown to the user in an interactive widget" in prompt
    # Default tool usage section also references the widget.
    assert "already visible to the user in a separate widget" in prompt


def test_system_prompt_section_overrides_replace_defaults():
    prompt = _render_system_prompt(
        SupremeAgentConfig(
            name="ask sigma",
            domain="insurance",
            terminology_domain="insurance",
            user_ui_context_section="The user sees only your text reply.",
            tool_usage_section="Custom tool usage rules.",
        )
    )

    assert "The user sees only your text reply." in prompt
    assert "Custom tool usage rules." in prompt
    # No widget mention may survive anywhere in the prompt.
    assert "widget" not in prompt.lower()
