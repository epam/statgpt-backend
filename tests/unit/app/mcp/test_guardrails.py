from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError

from statgpt.app.mcp import guardrails
from statgpt.app.mcp.guardrails import enforce_input_guardrail


def _tool(query):
    """A minimal stand-in StatGptTool exposing the given guardrail input."""
    return SimpleNamespace(name="data_query", get_guardrail_input=lambda arguments: query)


def _channel_config(*, out_of_scope):
    return SimpleNamespace(out_of_scope=out_of_scope)


def _auth_context():
    return SimpleNamespace(api_key="key")


def _patch_checker(monkeypatch, classify: AsyncMock) -> None:
    checker = SimpleNamespace(classify=classify)
    monkeypatch.setattr(guardrails, "OutOfScopeChecker", lambda channel_config: checker)


async def test_skips_when_globally_disabled(monkeypatch):
    classify = AsyncMock()
    _patch_checker(monkeypatch, classify)
    monkeypatch.setattr(guardrails.dial_app_settings, "skip_out_of_scope_check", True)

    await enforce_input_guardrail(
        _tool("anything"),
        {"query": "anything"},
        _channel_config(out_of_scope=SimpleNamespace()),
        _auth_context(),
    )

    classify.assert_not_called()


async def test_skips_when_channel_guardrails_disabled(monkeypatch):
    classify = AsyncMock()
    _patch_checker(monkeypatch, classify)

    await enforce_input_guardrail(
        _tool("anything"),
        {"query": "anything"},
        _channel_config(out_of_scope=None),
        _auth_context(),
    )

    classify.assert_not_called()


async def test_skips_when_tool_has_no_guardrail_input(monkeypatch):
    classify = AsyncMock()
    _patch_checker(monkeypatch, classify)

    await enforce_input_guardrail(
        _tool(None),
        {},
        _channel_config(out_of_scope=SimpleNamespace()),
        _auth_context(),
    )

    classify.assert_not_called()


async def test_raises_with_reasoning_when_out_of_scope(monkeypatch):
    decision = SimpleNamespace(out_of_scope=True, reasoning="off-domain weather request")
    classify = AsyncMock(return_value=decision)
    _patch_checker(monkeypatch, classify)

    with pytest.raises(ToolError, match="off-domain weather request"):
        await enforce_input_guardrail(
            _tool("weather in London"),
            {"query": "weather in London"},
            _channel_config(out_of_scope=SimpleNamespace()),
            _auth_context(),
        )

    classify.assert_awaited_once()


async def test_passes_when_in_scope(monkeypatch):
    decision = SimpleNamespace(out_of_scope=False, reasoning="statistics query")
    classify = AsyncMock(return_value=decision)
    _patch_checker(monkeypatch, classify)

    await enforce_input_guardrail(
        _tool("GDP of France"),
        {"query": "GDP of France"},
        _channel_config(out_of_scope=SimpleNamespace()),
        _auth_context(),
    )

    classify.assert_awaited_once()


async def test_fails_closed_when_classify_errors(monkeypatch):
    classify = AsyncMock(side_effect=RuntimeError("model timeout"))
    _patch_checker(monkeypatch, classify)

    with pytest.raises(ToolError, match="safety check could not be completed"):
        await enforce_input_guardrail(
            _tool("GDP of France"),
            {"query": "GDP of France"},
            _channel_config(out_of_scope=SimpleNamespace()),
            _auth_context(),
        )
