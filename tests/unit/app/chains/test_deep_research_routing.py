"""Routing tests for the Deep Research force flow in the Supreme Agent.

Deep Research is excluded from ``ChannelConfig.tool_fields`` and is therefore absent from
the config-scoped ``ToolCaller`` (``tool_executor``). The forced-start turn must dispatch
its single DR tool call through a caller that knows the built-on-demand DR tool; dispatching
it through ``tool_executor`` raises ``KeyError`` and the turn dies before the session is ever
saved or streamed back in DIAL state (so the next turn re-plans instead of resuming). These
tests pin that dispatch and the session round-trip.
"""

import json
from unittest.mock import MagicMock

from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableLambda
from openai import OpenAIError
from openai.types.chat import ChatCompletionChunk

from statgpt.app.chains import supreme_agent as supreme_agent_module
from statgpt.app.chains.deep_research import DEEP_RESEARCH_ERROR_MESSAGE
from statgpt.app.chains.deep_research import deep_research_tool as deep_research_module
from statgpt.app.chains.supreme_agent import SupremeAgentExecutor
from statgpt.app.config import ChainParametersConfig, StateVarsConfig
from statgpt.app.schemas import DeepResearchSession, DeepResearchTurn
from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.utils.dial_stages import NullChoice
from statgpt.app.utils.message_history import History
from statgpt.common.schemas.channel import ChannelConfig, SupremeAgentConfig
from statgpt.common.schemas.tools import DataQueryTool, DeepResearchTool


def _channel_config() -> ChannelConfig:
    return ChannelConfig(
        supreme_agent=SupremeAgentConfig(name="X", domain="d", terminology_domain="t"),
        deep_research=DeepResearchTool(
            name="deep_research",
            description="DR",
            enabled=True,
            details={"deployment_id": "dr-app"},
        ),
        data_query=DataQueryTool(name="data_query", description="DQ", enabled=True, details={}),
    )


class _FakeStream:
    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


def _dr_chunk(content: str | None = None, state: dict | None = None) -> ChatCompletionChunk:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if state is not None:
        delta["custom_content"] = {"state": state}
    return ChatCompletionChunk.model_validate(
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "dr",
            "choices": [{"index": 0, "finish_reason": None, "delta": delta}],
        }
    )


def _patch_supreme_agent_forced_dr(monkeypatch) -> None:
    """Force the Supreme Agent LLM to emit exactly one Deep Research tool call."""

    def _forced(_):
        return AIMessageChunk(
            content="",
            tool_calls=[
                {"name": "deep_research", "args": {"query": "q"}, "id": "c1", "type": "tool_call"}
            ],
        )

    class _FakeModel:
        def bind_tools(self, tools, **kwargs):
            return RunnableLambda(_forced)

    monkeypatch.setattr(supreme_agent_module, "get_chat_model", lambda **kwargs: _FakeModel())


def _patch_deep_research_client(monkeypatch, chunks: list[ChatCompletionChunk], captured: dict):
    class _FakeCompletions:
        async def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return _FakeStream(chunks)

    class _FakeClient:
        chat = type("_Chat", (), {"completions": _FakeCompletions()})()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        deep_research_module.openai, "get_async_client", lambda api_key=None, **k: _FakeClient()
    )


class _RecordingChoice(NullChoice):
    """A NullChoice that records what is streamed to the user, to assert the failure message is
    surfaced exactly once (not double-appended by both the tool and the Supreme Agent)."""

    def __init__(self) -> None:
        self.appended: list[str] = []

    def append_content(self, content: str) -> None:
        self.appended.append(content)


def _inputs(state: dict, user_text: str, choice=None) -> dict:
    return {
        ChainParametersConfig.STATE: state,
        ChainParametersConfig.CHOICE: choice if choice is not None else NullChoice(),
        ChainParametersConfig.AUTH_CONTEXT: MagicMock(api_key="k"),
        ChainParametersConfig.HISTORY: History(
            messages=[DialMessage(role=Role.USER, content=user_text)]
        ),
        ChainParametersConfig.CONFIGURATION: StatGPTConfiguration(deep_research=True),
    }


async def test_forced_start_dispatches_dr_and_saves_session(monkeypatch):
    """Regression: the forced DR call must not be routed through the config-scoped executor
    (which excludes DR and would raise KeyError). It must run and persist the session so the
    next turn can resume."""
    _patch_supreme_agent_forced_dr(monkeypatch)
    captured: dict = {}
    _patch_deep_research_client(
        monkeypatch,
        [
            _dr_chunk(content="Plan:\n1. step\nDoes this look good?"),
            _dr_chunk(state={"preparation": {"research_started": False}}),
        ],
        captured,
    )

    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    content = await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs(state, "give me US GDP")
    )

    assert "Does this look good?" in content
    session = DeepResearchSession.from_state(state)
    assert session is not None, "forced-start turn must persist the Deep Research session"
    # The start turn forwards the LLM's `query` argument (not the raw user text); the turn
    # carries Deep Research's own state for the next turn to resume from.
    assert len(session.turns) == 1
    assert session.turns[0].user_message == "q"
    assert session.turns[0].deep_research_state == {"preparation": {"research_started": False}}
    # Regression: the persisted session must not embed a `custom_content` key, which the DIAL
    # chat client strips from message-shaped objects while round-tripping state.
    assert "custom_content" not in json.dumps(state[StateVarsConfig.DEEP_RESEARCH_SESSION])


async def test_resume_forwards_latest_message_verbatim(monkeypatch):
    """With a session in progress and the toggle on, the turn resumes by forwarding the user's
    latest message verbatim to Deep Research (no Supreme Agent re-composition)."""
    _patch_supreme_agent_forced_dr(monkeypatch)  # must NOT be used on the resume path
    captured: dict = {}
    _patch_deep_research_client(
        monkeypatch,
        [
            _dr_chunk(content="Final report."),
            _dr_chunk(state={"preparation": {"research_started": True}}),
        ],
        captured,
    )

    prior = DeepResearchSession(
        turns=[
            DeepResearchTurn(
                user_message="give me US GDP",
                assistant_content="plan?",
                deep_research_state={},
            )
        ]
    )
    state = {
        StateVarsConfig.SHOW_DEBUG_STAGES: False,
        StateVarsConfig.DEEP_RESEARCH_SESSION: prior.model_dump(mode="json"),
    }

    content = await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs(state, "plan looks good")
    )

    assert content == "Final report."
    # The full sub-conversation is replayed: the prior turn (rebuilt into DIAL shape carrying
    # Deep Research's own state) followed by the user's verbatim latest message.
    assert captured["messages"] == [
        {"role": "user", "content": "give me US GDP"},
        {"role": "assistant", "content": "plan?", "custom_content": {"state": {}}},
        {"role": "user", "content": "plan looks good"},
    ]
    # Research started -> the finished session is dropped from state.
    assert DeepResearchSession.from_state(state) is None


def _patch_deep_research_client_raises(monkeypatch, error: Exception) -> None:
    class _RaisingCompletions:
        async def create(self, **kwargs):
            raise error

    class _RaisingClient:
        chat = type("_Chat", (), {"completions": _RaisingCompletions()})()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        deep_research_module.openai, "get_async_client", lambda api_key=None, **k: _RaisingClient()
    )


async def test_forced_start_error_is_recorded_in_state_and_surfaced_once(monkeypatch):
    """A request/stream failure on the forced-start turn must propagate to `call_tool` (which
    records the error in the tool state) rather than being surfaced by the tool itself. The Supreme
    Agent then surfaces the standard message exactly once, and the failed turn leaves no session."""
    _patch_supreme_agent_forced_dr(monkeypatch)
    _patch_deep_research_client_raises(monkeypatch, OpenAIError("deployment down"))

    choice = _RecordingChoice()
    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    inputs = _inputs(state, "give me US GDP", choice=choice)
    content = await SupremeAgentExecutor(_channel_config()).stream_response(inputs)

    assert content == DEEP_RESEARCH_ERROR_MESSAGE
    # Surfaced exactly once: the tool no longer streams the error itself, so there is no second
    # append from the Supreme Agent's ERROR handling.
    assert choice.appended == [DEEP_RESEARCH_ERROR_MESSAGE]
    # The failed turn is left untouched so the user can retry: no session is persisted.
    assert DeepResearchSession.from_state(state) is None
    # The error is recorded in the tool state (an ERROR tool message), not swallowed.
    history = inputs[ChainParametersConfig.HISTORY]
    history.dump_state(state)
    dr_states = [
        msg["custom_content"]["state"]
        for msg in state[StateVarsConfig.TOOL_MESSAGES]
        if msg.get("custom_content", {}).get("state", {}).get("type") == "DEEP_RESEARCH"
    ]
    assert dr_states, "the failed Deep Research turn must record a tool state"
    assert dr_states[0].get("error"), "the recorded tool state must carry the error"
