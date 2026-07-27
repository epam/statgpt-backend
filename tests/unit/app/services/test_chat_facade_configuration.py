"""Unit tests for ChannelServiceFacade.get_dial_channel_configuration.

Focus: the `deep_research` property is advertised in the DIAL configuration schema
only when the channel has the Deep Research tool configured and enabled.
"""

from unittest.mock import MagicMock

import pytest

from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.common.schemas.channel import (
    ChannelConfig,
    ConversationStarterConfig,
    ConversationStartersConfig,
    SupremeAgentConfig,
)
from statgpt.common.schemas.tools import DeepResearchTool


def _channel_config(
    *, deep_research: DeepResearchTool | None = None, starters=None
) -> ChannelConfig:
    return ChannelConfig(
        supreme_agent=SupremeAgentConfig(
            name="StatGPT", domain="statistics", terminology_domain="statistics"
        ),
        deep_research=deep_research,
        conversation_starters=starters,
    )


def _deep_research_tool(*, enabled: bool = True) -> DeepResearchTool:
    return DeepResearchTool(
        name="deep_research",
        description="Deep Research tool",
        enabled=enabled,
        details={"deployment_id": "deep-research-app"},
    )


def _facade(config: ChannelConfig) -> ChannelServiceFacade:
    channel = MagicMock()
    channel.title = "Test Channel"
    channel.details = config
    return ChannelServiceFacade(channel=channel)


async def _get_schema(config: ChannelConfig) -> dict:
    facade = _facade(config)
    return await facade.get_dial_channel_configuration(auth_context=MagicMock())


class TestDeepResearchConfiguration:

    @pytest.mark.asyncio
    async def test_omitted_when_tool_absent(self) -> None:
        schema = await _get_schema(_channel_config())

        assert "deep_research" not in schema["properties"]
        # base fields are still advertised
        assert "timezone" in schema["properties"]
        assert "enable_debug_attachments" in schema["properties"]

    @pytest.mark.asyncio
    async def test_omitted_when_tool_disabled(self) -> None:
        schema = await _get_schema(
            _channel_config(deep_research=_deep_research_tool(enabled=False))
        )

        assert "deep_research" not in schema["properties"]

    @pytest.mark.asyncio
    async def test_advertised_when_tool_enabled(self) -> None:
        schema = await _get_schema(_channel_config(deep_research=_deep_research_tool()))

        prop = schema["properties"]["deep_research"]
        assert prop["type"] == "boolean"
        assert prop["title"] == "Enable Deep research"
        assert prop["default"] is False
        # the base schema stays intact aside from the added toggle
        assert schema.get("dial:chatMessageInputDisabled") is False
        assert schema["additionalProperties"] is False

    @pytest.mark.asyncio
    async def test_advertised_alongside_conversation_starters(self) -> None:
        starters = ConversationStartersConfig(
            intro_text="Welcome",
            buttons=[ConversationStarterConfig(title="Ask", text="Ask something")],
        )
        schema = await _get_schema(
            _channel_config(deep_research=_deep_research_tool(), starters=starters)
        )

        assert schema["properties"]["deep_research"]["title"] == "Enable Deep research"
        assert "starter" in schema["properties"]
