"""The cross-cutting MCP guards as wired into `StatGptMcpTool.run()`: rate limiting (#600) and the
payload budget (#601) run for model-facing tools and are skipped for app-only tools."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from statgpt.app.chains.tools import StatGptTool
from statgpt.app.mcp.tools import StatGptMcpTool
from statgpt.app.settings.mcp import mcp_settings
from statgpt.common.schemas.tools import AvailablePublicationsTool


def _channel_config() -> SimpleNamespace:
    # out_of_scope=None disables the guardrail, so run() proceeds straight through.
    return SimpleNamespace(mcp=SimpleNamespace(tool_name_prefix=""), out_of_scope=None)


def _build(monkeypatch, *, visibility) -> StatGptMcpTool:
    """A LangChain-backed MCP tool whose config carries `visibility` (None => model-facing,
    ["app"] => app-only)."""
    fake = SimpleNamespace(
        name="fake_tool",
        ainvoke=AsyncMock(return_value=SimpleNamespace(content="ok", artifact=None)),
    )
    monkeypatch.setattr(
        "statgpt.app.mcp.tools.base.StatGptTool",
        SimpleNamespace(
            from_config=lambda tool_config, channel_config: fake,
            implementation_for=StatGptTool.implementation_for,
        ),
    )
    config = AvailablePublicationsTool(
        name="pubs", description="Publications.", mcp_visibility=visibility
    )
    return StatGptMcpTool.from_config(
        config,
        _channel_config(),
        inputs={},
        auth_context=SimpleNamespace(dial_access_token="tok"),
    )


async def test_rate_limit_enforced_for_model_facing_tool(monkeypatch):
    monkeypatch.setattr(mcp_settings, "mcp_rate_limit_enabled", True)
    enforce = MagicMock()
    monkeypatch.setattr("statgpt.app.mcp.tools.base.enforce_rate_limit", enforce)

    await _build(monkeypatch, visibility=None).run({})

    enforce.assert_called_once()


async def test_rate_limit_skipped_for_app_only_tool(monkeypatch):
    monkeypatch.setattr(mcp_settings, "mcp_rate_limit_enabled", True)
    enforce = MagicMock()
    monkeypatch.setattr("statgpt.app.mcp.tools.base.enforce_rate_limit", enforce)

    await _build(monkeypatch, visibility=["app"]).run({})

    enforce.assert_not_called()


async def test_payload_budget_enforced_for_model_facing_tool(monkeypatch):
    monkeypatch.setattr(mcp_settings, "mcp_payload_budget_enabled", True)
    enforce = MagicMock(side_effect=lambda result, **kwargs: result)
    monkeypatch.setattr("statgpt.app.mcp.tools.base.enforce_payload_budget", enforce)

    await _build(monkeypatch, visibility=None).run({})

    enforce.assert_called_once()


async def test_payload_budget_skipped_for_app_only_tool(monkeypatch):
    monkeypatch.setattr(mcp_settings, "mcp_payload_budget_enabled", True)
    enforce = MagicMock(side_effect=lambda result, **kwargs: result)
    monkeypatch.setattr("statgpt.app.mcp.tools.base.enforce_payload_budget", enforce)

    await _build(monkeypatch, visibility=["app"]).run({})

    enforce.assert_not_called()
