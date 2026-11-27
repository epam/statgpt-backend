"""The data attachments of the queries a turn ran are added whenever the agent loop ends: with the
final answer, at the iteration cap, and when the turn fails (#831). A later turn ("Continue") can
answer from the saved tool responses without running the queries again, so the failed turn is the
only one that can show their data."""

import itertools
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableLambda

from statgpt.app.chains import supreme_agent as supreme_agent_module
from statgpt.app.chains.supreme_agent import SupremeAgentExecutor, ToolCaller
from statgpt.app.config import ChainParametersConfig, StateVarsConfig
from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.app.utils.dial_stages import NullChoice
from statgpt.app.utils.message_history import History
from statgpt.common.schemas.channel import ChannelConfig, SupremeAgentConfig
from statgpt.common.schemas.tools import DataQueryTool

_RATE_LIMIT_ERROR = openai.RateLimitError(
    "Error code: 429 - Day limit: 120000 / 100000 tokens",
    response=httpx.Response(
        429, request=httpx.Request("POST", "http://dial/openai/deployments/gpt/chat/completions")
    ),
    body=None,
)


def _channel_config(max_agent_iterations: int = 10) -> ChannelConfig:
    return ChannelConfig(
        supreme_agent=SupremeAgentConfig(
            name="X",
            domain="d",
            terminology_domain="t",
            max_agent_iterations=max_agent_iterations,
        ),
        data_query=DataQueryTool(name="data_query", description="DQ", enabled=True, details={}),
    )


def _query_call(call_id: str = "c1") -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_calls=[{"name": "data_query", "args": {}, "id": call_id, "type": "tool_call"}],
    )


def _patch_scripted_agent(monkeypatch, responses: list[AIMessageChunk | Exception]) -> None:
    """Drive the agent's model with one scripted response per run; an exception is raised."""
    scripted = iter(responses)

    def _respond(prompt_value):
        response = next(scripted)
        if isinstance(response, Exception):
            raise response
        return response

    class _Model:
        def bind_tools(self, tools, **kwargs):
            return RunnableLambda(_respond)

    monkeypatch.setattr(supreme_agent_module, "get_chat_model", lambda **kwargs: _Model())


def _patch_data_query(monkeypatch) -> MagicMock:
    """Make every tool call a data query that returned rows; return the patched displayer."""
    artifact = DataQueryArtifact.model_construct(data_responses={})

    async def call_tool(self, tool_call, inputs, **kwargs) -> ToolMessage:
        return ToolMessage(content="rows", tool_call_id=tool_call["id"], artifact=artifact)

    monkeypatch.setattr(ToolCaller, "call_tool", call_tool)
    monkeypatch.setattr(
        SupremeAgentExecutor, "_fake_tool_calls", AsyncMock(return_value=History.create_empty())
    )
    displayer = MagicMock()
    displayer.display = AsyncMock()
    displayer.get_system_message_content = AsyncMock(return_value="data")
    monkeypatch.setattr(
        supreme_agent_module, "DataQueryArtifactDisplayer", MagicMock(return_value=displayer)
    )
    return displayer


def _inputs() -> dict:
    return {
        ChainParametersConfig.STATE: {StateVarsConfig.SHOW_DEBUG_STAGES: False},
        ChainParametersConfig.CHOICE: NullChoice(),
        ChainParametersConfig.AUTH_CONTEXT: MagicMock(api_key="k", is_system=False),
        ChainParametersConfig.HISTORY: History(
            messages=[DialMessage(role=Role.USER, content="US GDP")]
        ),
        ChainParametersConfig.CONFIGURATION: StatGPTConfiguration(),
        ChainParametersConfig.ANNOTATION_INDEX_SPACE: itertools.count(),
        ChainParametersConfig.PERFORMANCE_STAGE: None,
    }


def _displayed_call_ids(displayer: MagicMock) -> list[list[str]]:
    return [list(call.args[0]) for call in displayer.display.await_args_list]


async def test_rate_limit_after_a_query_adds_the_attachments_and_raises(monkeypatch):
    _patch_scripted_agent(monkeypatch, [_query_call(), _RATE_LIMIT_ERROR])
    displayer = _patch_data_query(monkeypatch)

    with pytest.raises(openai.RateLimitError) as exc_info:
        await SupremeAgentExecutor(_channel_config()).stream_response(_inputs())

    assert exc_info.value is _RATE_LIMIT_ERROR
    assert _displayed_call_ids(displayer) == [["c1"]]


async def test_failing_attachments_do_not_replace_the_turn_error(monkeypatch):
    _patch_scripted_agent(monkeypatch, [_query_call(), _RATE_LIMIT_ERROR])
    displayer = _patch_data_query(monkeypatch)
    displayer.display.side_effect = RuntimeError("attachments failed")
    log_exception = MagicMock()
    monkeypatch.setattr(supreme_agent_module.logger, "exception", log_exception)

    with pytest.raises(openai.RateLimitError) as exc_info:
        await SupremeAgentExecutor(_channel_config()).stream_response(_inputs())

    assert exc_info.value is _RATE_LIMIT_ERROR
    log_exception.assert_called_once_with("Failed to add the data attachments of a failed turn")


async def test_failure_before_any_query_adds_no_attachments(monkeypatch):
    _patch_scripted_agent(monkeypatch, [_RATE_LIMIT_ERROR])
    displayer = _patch_data_query(monkeypatch)

    with pytest.raises(openai.RateLimitError):
        await SupremeAgentExecutor(_channel_config()).stream_response(_inputs())

    displayer.display.assert_not_awaited()


async def test_iteration_cap_adds_the_attachments(monkeypatch):
    _patch_scripted_agent(monkeypatch, [_query_call("c1"), _query_call("c2")])
    displayer = _patch_data_query(monkeypatch)

    answer = await SupremeAgentExecutor(_channel_config(max_agent_iterations=2)).stream_response(
        _inputs()
    )

    assert "Maximum number of agent loop iterations reached" in answer
    assert _displayed_call_ids(displayer) == [["c1", "c2"]]


async def test_successful_turn_adds_the_attachments_once(monkeypatch):
    _patch_scripted_agent(monkeypatch, [_query_call(), AIMessageChunk(content="US GDP grew.")])
    displayer = _patch_data_query(monkeypatch)

    answer = await SupremeAgentExecutor(_channel_config()).stream_response(_inputs())

    assert answer == "US GDP grew."
    assert _displayed_call_ids(displayer) == [["c1"]]
