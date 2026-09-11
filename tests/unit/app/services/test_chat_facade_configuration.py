"""Unit tests for ChannelServiceFacade.get_dial_channel_configuration.

Focus: the two gated properties of the DIAL configuration schema. `deep_research` is
advertised only when the channel has the Deep Research tool configured and enabled, and —
when the tool is gated on an `access_claim_value` role — only for callers whose DIAL roles
include it. `enable_debug_attachments` is advertised only when
`DIAL_ALLOW_DEBUG_ATTACHMENTS_TOGGLE` is on.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.app.settings.dial_app import dial_app_settings
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


def _deep_research_tool(
    *,
    enabled: bool = True,
    access_claim_value: str | None = None,
) -> DeepResearchTool:
    details: dict = {"deployment_id": "deep-research-app"}
    if access_claim_value is not None:
        details["access_claim_value"] = access_claim_value
    return DeepResearchTool(
        name="deep_research",
        description="Deep Research tool",
        enabled=enabled,
        details=details,
    )


def _auth_context(*, has_role: bool = True, is_system: bool = False) -> MagicMock:
    auth_context = MagicMock()
    auth_context.is_system = is_system
    auth_context.has_role = AsyncMock(return_value=has_role)
    return auth_context


def _facade(config: ChannelConfig) -> ChannelServiceFacade:
    channel = MagicMock()
    channel.title = "Test Channel"
    channel.details = config
    return ChannelServiceFacade(channel=channel)


async def _get_schema(config: ChannelConfig, auth_context=None) -> dict:
    facade = _facade(config)
    return await facade.get_dial_channel_configuration(
        auth_context=auth_context if auth_context is not None else _auth_context()
    )


class TestDeepResearchConfiguration:

    @pytest.mark.asyncio
    async def test_omitted_when_tool_absent(self) -> None:
        schema = await _get_schema(_channel_config())

        assert "deep_research" not in schema["properties"]
        # base fields are still advertised
        assert "timezone" in schema["properties"]

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
        assert prop["title"] == "Deep research"
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

        assert schema["properties"]["deep_research"]["title"] == "Deep research"
        assert "starter" in schema["properties"]

    @pytest.mark.asyncio
    async def test_advertised_when_role_gated_and_caller_has_role(self) -> None:
        auth_context = _auth_context(has_role=True)
        schema = await _get_schema(
            _channel_config(deep_research=_deep_research_tool(access_claim_value="dr_access")),
            auth_context=auth_context,
        )

        assert "deep_research" in schema["properties"]
        auth_context.has_role.assert_awaited_once_with("dr_access")

    @pytest.mark.asyncio
    async def test_omitted_when_role_gated_and_caller_lacks_role(self) -> None:
        auth_context = _auth_context(has_role=False)
        schema = await _get_schema(
            _channel_config(deep_research=_deep_research_tool(access_claim_value="dr_access")),
            auth_context=auth_context,
        )

        assert "deep_research" not in schema["properties"]
        auth_context.has_role.assert_awaited_once_with("dr_access")

    @pytest.mark.asyncio
    async def test_advertised_for_system_user_even_when_role_gated(self) -> None:
        auth_context = _auth_context(has_role=False, is_system=True)
        schema = await _get_schema(
            _channel_config(deep_research=_deep_research_tool(access_claim_value="dr_access")),
            auth_context=auth_context,
        )

        assert "deep_research" in schema["properties"]
        # System users bypass the role gate, so DIAL roles are never consulted.
        auth_context.has_role.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_advertised_when_role_unset_regardless_of_token(self) -> None:
        auth_context = _auth_context(has_role=False)
        schema = await _get_schema(
            _channel_config(deep_research=_deep_research_tool()),
            auth_context=auth_context,
        )

        assert "deep_research" in schema["properties"]
        # No role configured -> DIAL roles are never consulted.
        auth_context.has_role.assert_not_awaited()


@pytest.fixture
def debug_attachments_toggle_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dial_app_settings, "dial_allow_debug_attachments_toggle", True)


class TestDebugAttachmentsConfiguration:

    @pytest.mark.asyncio
    async def test_omitted_by_default(self) -> None:
        schema = await _get_schema(_channel_config())

        assert "enable_debug_attachments" not in schema["properties"]

    @pytest.mark.asyncio
    async def test_advertised_when_allowed(self, debug_attachments_toggle_allowed: None) -> None:
        schema = await _get_schema(_channel_config())

        prop = schema["properties"]["enable_debug_attachments"]
        assert prop["type"] == "boolean"
        assert prop["default"] is False

    @pytest.mark.asyncio
    async def test_advertised_alongside_deep_research(
        self, debug_attachments_toggle_allowed: None
    ) -> None:
        schema = await _get_schema(_channel_config(deep_research=_deep_research_tool()))

        assert list(schema["properties"]) == [
            "timezone",
            "enable_debug_attachments",
            "deep_research",
        ]
        assert schema.get("dial:chatMessageInputDisabled") is False
        assert schema["additionalProperties"] is False

    @pytest.mark.asyncio
    async def test_advertised_alongside_conversation_starters(
        self, debug_attachments_toggle_allowed: None
    ) -> None:
        starters = ConversationStartersConfig(
            intro_text="Welcome",
            buttons=[ConversationStarterConfig(title="Ask", text="Ask something")],
        )
        schema = await _get_schema(_channel_config(starters=starters))

        assert "enable_debug_attachments" in schema["properties"]
        assert schema["properties"]["starter"]["dial:widget"] == "buttons"
