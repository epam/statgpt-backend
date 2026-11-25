from statgpt.app.chains.supreme_agent import SupremeAgent
from statgpt.app.default_prompts import supreme_agent_default_prompts
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

    assert supreme_agent_default_prompts.default_user_ui_context_section in prompt
    assert supreme_agent_default_prompts.default_tool_usage_section in prompt
    # The data presentation section carries the {today_date} placeholder, which must still be
    # resolved when the section is injected.
    assert (
        supreme_agent_default_prompts.default_data_presentation_section.format(
            today_date="2026-08-21"
        )
        in prompt
    )


def test_system_prompt_section_overrides_replace_defaults():
    prompt = _render_system_prompt(
        SupremeAgentConfig(
            name="ask sigma",
            domain="insurance",
            terminology_domain="insurance",
            user_ui_context_section="The user sees only your text reply.",
            tool_usage_section="Custom tool usage rules.",
            data_presentation_section=(
                "Cite everything. You are {chat_bot_name} and today is {today_date}."
            ),
        )
    )

    assert "The user sees only your text reply." in prompt
    assert "Custom tool usage rules." in prompt
    # An override may reference any of the prompt's placeholders, not just `{today_date}`.
    assert "Cite everything. You are ask sigma and today is 2026-08-21." in prompt
    # Overrides must fully replace the defaults, not merely be appended.
    assert supreme_agent_default_prompts.default_user_ui_context_section not in prompt
    assert supreme_agent_default_prompts.default_tool_usage_section not in prompt
    assert "coverage statements MISLEAD THE USER" not in prompt


def test_deep_research_section_is_appended_only_on_deep_research_turns():
    channel_config = ChannelConfig(
        supreme_agent=SupremeAgentConfig(
            name="StatGPT", domain="official statistics", terminology_domain="official statistics"
        )
    )
    section = supreme_agent_default_prompts.deep_research_section

    def render(deep_research: bool) -> str:
        template = SupremeAgent._create_prompt_template(channel_config, deep_research=deep_research)
        return str(template.format_messages(today_date="2026-08-21")[0].content)

    # The section carries no placeholders, so it is rendered verbatim.
    assert section in render(deep_research=True)
    assert section not in render(deep_research=False)


def test_deep_research_section_forwards_plan_changes_to_deep_research():
    """Plan amendments go through the resume tool; the agent never writes a plan itself (#835)."""
    section = supreme_agent_default_prompts.deep_research_section

    assert "**requested changes**" in section
    assert "**Never write a query or plan yourself.**" in section
    assert "Only Deep Research can update **both** the research" in section
    # A plan amendment is not an "unrelated" request to redirect.
    assert "is a requested change to the plan, so forward it." in section
    assert "in doubt whether a message is about the research, forward it." in section


def test_deep_research_section_resolves_references_from_the_conversation():
    """Deep Research does not see the chat, so references to it are written out in full (#835)."""
    section = supreme_agent_default_prompts.deep_research_section

    assert "**Make every message to Deep Research self-contained.**" in section
    assert "including earlier turns, with the concrete details" in section
    assert '"add the publication I asked about", name that publication.' in section
