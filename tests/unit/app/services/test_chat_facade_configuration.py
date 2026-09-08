"""Unit tests for ChannelServiceFacade.get_dial_channel_configuration.

Focus: the `deep_research` property is advertised in the DIAL configuration schema
only when the channel has the Deep Research tool configured and enabled, and — when the
tool is gated on an `access_claim` (optionally requiring a specific `access_claim_value`) —
only for callers whose token satisfies that claim.
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


def _deep_research_tool(
    *,
    enabled: bool = True,
    access_claim: str | None = None,
    access_claim_value: str | None = None,
) -> DeepResearchTool:
    details: dict = {"deployment_id": "deep-research-app"}
    if access_claim is not None:
        details["access_claim"] = access_claim
    if access_claim_value is not None:
        details["access_claim_value"] = access_claim_value
    return DeepResearchTool(
        name="deep_research",
        description="Deep Research tool",
        enabled=enabled,
        details=details,
    )


def _auth_context(*, has_claim: bool = True, is_system: bool = False) -> MagicMock:
    auth_context = MagicMock()
    auth_context.is_system = is_system
    auth_context.has_claim_value.return_value = has_claim
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
    async def test_advertised_when_claim_gated_and_caller_has_claim(self) -> None:
        schema = await _get_schema(
            _channel_config(deep_research=_deep_research_tool(access_claim="dr_access")),
            auth_context=_auth_context(has_claim=True),
        )

        assert "deep_research" in schema["properties"]

    @pytest.mark.asyncio
    async def test_omitted_when_claim_gated_and_caller_lacks_claim(self) -> None:
        auth_context = _auth_context(has_claim=False)
        schema = await _get_schema(
            _channel_config(deep_research=_deep_research_tool(access_claim="dr_access")),
            auth_context=auth_context,
        )

        assert "deep_research" not in schema["properties"]
        auth_context.has_claim_value.assert_called_once_with("dr_access", None)

    @pytest.mark.asyncio
    async def test_claim_value_gate_passes_required_value(self) -> None:
        """`access_claim_value` is forwarded so the caller's claim (e.g. `roles`) is checked for
        that specific value rather than mere presence."""
        auth_context = _auth_context(has_claim=True)
        schema = await _get_schema(
            _channel_config(
                deep_research=_deep_research_tool(
                    access_claim="roles", access_claim_value="dr_access"
                )
            ),
            auth_context=auth_context,
        )

        assert "deep_research" in schema["properties"]
        auth_context.has_claim_value.assert_called_once_with("roles", "dr_access")

    @pytest.mark.asyncio
    async def test_advertised_for_system_user_even_when_claim_gated(self) -> None:
        auth_context = _auth_context(has_claim=False, is_system=True)
        schema = await _get_schema(
            _channel_config(deep_research=_deep_research_tool(access_claim="dr_access")),
            auth_context=auth_context,
        )

        assert "deep_research" in schema["properties"]
        # System users bypass the claim gate, so the token is never consulted.
        auth_context.has_claim_value.assert_not_called()

    @pytest.mark.asyncio
    async def test_advertised_when_claim_unset_regardless_of_token(self) -> None:
        auth_context = _auth_context(has_claim=False)
        schema = await _get_schema(
            _channel_config(deep_research=_deep_research_tool()),
            auth_context=auth_context,
        )

        assert "deep_research" in schema["properties"]
        # No claim configured -> the token is never consulted.
        auth_context.has_claim_value.assert_not_called()
