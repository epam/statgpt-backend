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


def _patch_checker(
    monkeypatch, checker_invoke: AsyncMock, response_invoke: AsyncMock | None = None
) -> SimpleNamespace:
    """Stub OutOfScopeChecker with build-only methods returning fake chains.

    The checker now exposes ``build_checker_chain`` / ``build_response_chain`` and the
    guardrail invokes the returned chains; ``*_invoke`` stand in for ``chain.ainvoke``.
    """
    response_invoke = response_invoke or AsyncMock(
        return_value=SimpleNamespace(content="generated message")
    )
    checker = SimpleNamespace(
        build_checker_chain=lambda messages, auth_context: SimpleNamespace(ainvoke=checker_invoke),
        build_response_chain=lambda messages, reasoning, auth_context: SimpleNamespace(
            ainvoke=response_invoke
        ),
    )
    monkeypatch.setattr(guardrails, "OutOfScopeChecker", lambda channel_config: checker)
    return checker


async def test_skips_when_globally_disabled(monkeypatch):
    checker_invoke = AsyncMock()
    _patch_checker(monkeypatch, checker_invoke)
    monkeypatch.setattr(guardrails.dial_app_settings, "skip_out_of_scope_check", True)

    await enforce_input_guardrail(
        _tool("anything"),
        {"query": "anything"},
        _channel_config(out_of_scope=SimpleNamespace()),
        _auth_context(),
    )

    checker_invoke.assert_not_called()


async def test_skips_when_channel_guardrails_disabled(monkeypatch):
    checker_invoke = AsyncMock()
    _patch_checker(monkeypatch, checker_invoke)

    await enforce_input_guardrail(
        _tool("anything"),
        {"query": "anything"},
        _channel_config(out_of_scope=None),
        _auth_context(),
    )

    checker_invoke.assert_not_called()


async def test_skips_when_tool_has_no_guardrail_input(monkeypatch):
    checker_invoke = AsyncMock()
    _patch_checker(monkeypatch, checker_invoke)

    await enforce_input_guardrail(
        _tool(None),
        {},
        _channel_config(out_of_scope=SimpleNamespace()),
        _auth_context(),
    )

    checker_invoke.assert_not_called()


async def test_raises_with_generated_message_when_out_of_scope(monkeypatch):
    decision = SimpleNamespace(out_of_scope=True, reasoning="off-domain weather request")
    checker_invoke = AsyncMock(return_value=decision)
    response_invoke = AsyncMock(
        return_value=SimpleNamespace(content="I can only help with official statistics.")
    )
    _patch_checker(monkeypatch, checker_invoke, response_invoke)

    with pytest.raises(ToolError, match="I can only help with official statistics."):
        await enforce_input_guardrail(
            _tool("weather in London"),
            {"query": "weather in London"},
            _channel_config(out_of_scope=SimpleNamespace()),
            _auth_context(),
        )

    checker_invoke.assert_awaited_once()
    response_invoke.assert_awaited_once()


async def test_falls_back_to_reasoning_when_response_generation_fails(monkeypatch):
    decision = SimpleNamespace(out_of_scope=True, reasoning="off-domain weather request")
    checker_invoke = AsyncMock(return_value=decision)
    response_invoke = AsyncMock(side_effect=RuntimeError("model timeout"))
    _patch_checker(monkeypatch, checker_invoke, response_invoke)

    with pytest.raises(ToolError, match="off-domain weather request"):
        await enforce_input_guardrail(
            _tool("weather in London"),
            {"query": "weather in London"},
            _channel_config(out_of_scope=SimpleNamespace()),
            _auth_context(),
        )

    response_invoke.assert_awaited_once()


async def test_passes_when_in_scope(monkeypatch):
    decision = SimpleNamespace(out_of_scope=False, reasoning="statistics query")
    checker_invoke = AsyncMock(return_value=decision)
    _patch_checker(monkeypatch, checker_invoke)

    await enforce_input_guardrail(
        _tool("GDP of France"),
        {"query": "GDP of France"},
        _channel_config(out_of_scope=SimpleNamespace()),
        _auth_context(),
    )

    checker_invoke.assert_awaited_once()


async def test_fails_closed_when_checker_chain_errors(monkeypatch):
    checker_invoke = AsyncMock(side_effect=RuntimeError("model timeout"))
    _patch_checker(monkeypatch, checker_invoke)

    with pytest.raises(ToolError, match="safety check could not be completed"):
        await enforce_input_guardrail(
            _tool("GDP of France"),
            {"query": "GDP of France"},
            _channel_config(out_of_scope=SimpleNamespace()),
            _auth_context(),
        )
