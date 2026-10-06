"""Routing + mediation tests for the Deep Research flow in the Supreme Agent (#575).

Deep Research is excluded from ``ChannelConfig.tool_fields`` and is reachable only via the
``deep_research`` toggle. A Deep Research turn runs in its own mediated loop
(``SupremeAgentExecutor._run_deep_research_turn``), bound only to the Deep Research tools:

- clarifications / plans are **buffered** and handed back to the agent as a tool response, never
  streamed to the user as-is;
- the agent answers from context via the resume tool and surfaces only the remainder;
- the final report is delivered to the user **verbatim** by the tool, and the turn ends without the
  agent repeating it;
- the Deep Research <-> agent exchange runs on the main history, so it is persisted into the
  cross-turn tool state and stays visible to later turns.

These tests pin that behaviour and the deterministic toggle/session routing, plus the check that
the query suits Deep Research before a new session is started (#732).
"""

import asyncio
import itertools
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_core.runnables import RunnableLambda
from openai import APIError, OpenAIError
from openai.types.chat import ChatCompletionChunk

from statgpt.app.chains import supreme_agent as supreme_agent_module
from statgpt.app.chains.deep_research import (
    DEEP_RESEARCH_ERROR_MESSAGE,
    DeepResearchFailedError,
    DeepResearchQueryChecker,
)
from statgpt.app.chains.deep_research import deep_research_tool as deep_research_module
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.supreme_agent import SupremeAgentExecutor, _DeepResearchMode
from statgpt.app.config import ChainParametersConfig, StateVarsConfig
from statgpt.app.default_prompts import supreme_agent_default_prompts
from statgpt.app.schemas import DeepResearchSession, DeepResearchTurn
from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.utils.dial_stages import DummyStage, NullChoice
from statgpt.app.utils.message_history import History
from statgpt.common.schemas.channel import ChannelConfig, SupremeAgentConfig
from statgpt.common.schemas.tools import DataQueryTool, DeepResearchTool

_ENABLED_NOTE = supreme_agent_default_prompts.deep_research_enabled_note
_DISABLED_NOTE = supreme_agent_default_prompts.deep_research_disabled_note


@pytest.fixture(autouse=True)
def query_check(monkeypatch) -> AsyncMock:
    """Stub the check that the query suits Deep Research (an LLM call). By default the query suits
    Deep Research, so a START turn starts a session; tests flip `return_value` to skip it."""
    check = AsyncMock(return_value=True)
    monkeypatch.setattr(DeepResearchQueryChecker, "suits_deep_research", check)
    return check


def _channel_config(
    *, access_claim_value: str | None = None, query_check: dict | None = None
) -> ChannelConfig:
    details: dict = {"deployment_id": "dr-app"}
    if access_claim_value is not None:
        details["access_claim_value"] = access_claim_value
    if query_check is not None:
        details["query_check"] = query_check
    return ChannelConfig(
        supreme_agent=SupremeAgentConfig(name="X", domain="d", terminology_domain="t"),
        deep_research=DeepResearchTool(
            name="deep_research",
            description="DR",
            enabled=True,
            details=details,
        ),
        data_query=DataQueryTool(name="data_query", description="DQ", enabled=True, details={}),
    )


class _RecordingChoice(NullChoice):
    """A NullChoice that records what reaches the user (content + attachments), so we can assert
    what was surfaced and that the failure message is streamed exactly once."""

    def __init__(self) -> None:
        self.appended: list[str] = []
        self.attachments: list[dict] = []

    def append_content(self, content: str) -> None:
        self.appended.append(content)

    def add_attachment(self, **kwargs) -> None:
        self.attachments.append(kwargs)

    @property
    def content(self) -> str:
        return "".join(self.appended)


class _StageRecordingChoice(_RecordingChoice):
    """A recording choice whose created stages also record their content, so we can assert Deep
    Research's clarification is surfaced to a stage (like other tools) rather than the main answer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stage_content: list[str] = []

    def create_stage(self, *args, **kwargs):
        sink = self.stage_content

        class _RecordingStage(DummyStage):
            def append_content(self, content: str) -> None:
                sink.append(content)

            def __bool__(self) -> bool:
                # A real DIAL stage is truthy; `DelayedStage.append_content` gates on it.
                return True

        return _RecordingStage()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Supreme Agent (LLM) fake ~~~~~~~~~~~~~~~~~~~~~~~~~~


def _tool_call_chunk(
    name: str, args: dict, call_id: str = "c1", content: str = ""
) -> AIMessageChunk:
    return AIMessageChunk(
        content=content,
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _text_chunk(text: str) -> AIMessageChunk:
    return AIMessageChunk(content=text)


def _patch_scripted_agent(
    monkeypatch, responses: list[AIMessageChunk], prompts: list | None = None
) -> None:
    """Drive the Supreme Agent LLM with a scripted sequence of responses, one per agent run.

    A single shared model instance is returned for every ``get_chat_model`` call so the script is
    consumed in order across the forced-start and free mediation agents. If ``prompts`` is given,
    the messages each agent run was prompted with are appended to it."""

    scripted = iter(responses)

    def _respond(prompt_value):
        if prompts is not None:
            prompts.append(prompt_value.to_messages())
        return next(scripted)

    class _SharedModel:
        def bind_tools(self, tools, **kwargs):
            return RunnableLambda(_respond)

    shared = _SharedModel()
    monkeypatch.setattr(supreme_agent_module, "get_chat_model", lambda **kwargs: shared)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Deep Research deployment fake ~~~~~~~~~~~~~~~~~~~~~~


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


def _dr_chunk(
    content: str | None = None,
    state: dict | None = None,
    attachments: list[dict] | None = None,
) -> ChatCompletionChunk:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    custom_content: dict = {}
    if state is not None:
        custom_content["state"] = state
    if attachments is not None:
        custom_content["attachments"] = attachments
    if custom_content:
        delta["custom_content"] = custom_content
    return ChatCompletionChunk.model_validate(
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "dr",
            "choices": [{"index": 0, "finish_reason": None, "delta": delta}],
        }
    )


def _clarification(text: str) -> list[ChatCompletionChunk]:
    return [_dr_chunk(content=text), _dr_chunk(state={"preparation": {"research_started": False}})]


def _report(text: str, attachments: list[dict] | None = None) -> list[ChatCompletionChunk]:
    return [
        _dr_chunk(content=text, attachments=attachments),
        _dr_chunk(state={"preparation": {"research_started": True}}),
    ]


def _patch_dr_deployment(
    monkeypatch, streams: list[list[ChatCompletionChunk]], captured: dict
) -> None:
    """Return one scripted stream per Deep Research deployment call, recording the sent messages."""

    calls = iter(streams)
    captured["messages"] = []

    class _FakeCompletions:
        async def create(self, **kwargs):
            captured["messages"].append(kwargs.get("messages"))
            return _FakeStream(next(calls))

    class _FakeClient:
        chat = type("_Chat", (), {"completions": _FakeCompletions()})()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        deep_research_module.openai, "get_async_client", lambda *a, **k: _FakeClient()
    )


def _patch_dr_deployment_raises(monkeypatch, error: Exception) -> None:
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
        deep_research_module.openai, "get_async_client", lambda *a, **k: _RaisingClient()
    )


def _patch_dr_deployment_counting(monkeypatch) -> dict:
    """A Deep Research client that only counts calls, to assert it is never invoked."""

    calls = {"count": 0}

    class _FakeCompletions:
        async def create(self, **kwargs):
            calls["count"] += 1
            return _FakeStream([])

    class _FakeClient:
        chat = type("_Chat", (), {"completions": _FakeCompletions()})()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        deep_research_module.openai, "get_async_client", lambda *a, **k: _FakeClient()
    )
    return calls


def _inputs(
    state: dict, user_text: str, *, deep_research: bool = True, choice=None, auth_context=None
) -> dict:
    return {
        ChainParametersConfig.STATE: state,
        ChainParametersConfig.CHOICE: choice if choice is not None else _RecordingChoice(),
        ChainParametersConfig.AUTH_CONTEXT: (
            auth_context if auth_context is not None else MagicMock(api_key="k", is_system=False)
        ),
        ChainParametersConfig.HISTORY: History(
            messages=[DialMessage(role=Role.USER, content=user_text)]
        ),
        ChainParametersConfig.CONFIGURATION: StatGPTConfiguration(deep_research=deep_research),
        ChainParametersConfig.ANNOTATION_INDEX_SPACE: itertools.count(),
        # Read when the general agent finishes (a turn where Deep Research was not started).
        ChainParametersConfig.PERFORMANCE_STAGE: None,
    }


def _session_state(*turns: DeepResearchTurn) -> dict:
    return {
        StateVarsConfig.SHOW_DEBUG_STAGES: False,
        StateVarsConfig.DEEP_RESEARCH_SESSION: DeepResearchSession(turns=list(turns)).model_dump(
            mode="json"
        ),
    }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_forced_start_buffers_clarification_and_surfaces_remainder(monkeypatch):
    """Forced start: the agent composes the query, Deep Research's clarification is buffered (not
    shown to the user) and handed back to the agent, which surfaces the remainder to the user."""
    _patch_scripted_agent(
        monkeypatch,
        [
            _tool_call_chunk("deep_research", {"query": "US GDP"}),  # forced first call
            _text_chunk("Which countries and what frequency would you like?"),  # surfaced remainder
        ],
    )
    captured: dict = {}
    _patch_dr_deployment(
        monkeypatch, [_clarification("Please specify time range, countries, frequency.")], captured
    )

    choice = _RecordingChoice()
    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    content = await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs(state, "research US GDP", choice=choice)
    )

    assert content == "Which countries and what frequency would you like?"
    # Only the agent's surfaced text reaches the user; the raw clarification is never streamed.
    assert choice.appended == ["Which countries and what frequency would you like?"]
    assert "Please specify" not in choice.content
    # The session persists so the next turn can resume; it carries the forced query and DR state.
    session = DeepResearchSession.from_state(state)
    assert session is not None
    assert len(session.turns) == 1
    assert session.turns[0].user_message == "US GDP"
    assert session.turns[0].deep_research_state == {"preparation": {"research_started": False}}
    # A clarification turn must not flag report delivery, so the toggle stays armed.
    assert StateVarsConfig.DEEP_RESEARCH_REPORT_DELIVERED not in state


async def test_forced_start_answers_from_context_then_delivers_report(monkeypatch):
    """The agent answers Deep Research's clarification from context (resume), then Deep Research
    returns the final report, which is delivered to the user verbatim and ends the session."""
    _patch_scripted_agent(
        monkeypatch,
        [
            _tool_call_chunk("deep_research", {"query": "US GDP 2015-2020"}),  # forced start
            _tool_call_chunk(  # answers the clarification entirely from context
                "resume_deep_research", {"message": "Countries: US. Frequency: annual."}, "c2"
            ),
        ],
    )
    captured: dict = {}
    _patch_dr_deployment(
        monkeypatch,
        [_clarification("Which countries and frequency?"), _report("# Final report\nBody.")],
        captured,
    )

    choice = _RecordingChoice()
    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    content = await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs(state, "research US GDP 2015-2020", choice=choice)
    )

    assert content == "# Final report\nBody."
    # The report is delivered verbatim; the intermediate clarification is never shown.
    assert choice.appended == ["# Final report\nBody."]
    # Research complete -> the finished session is dropped from state.
    assert DeepResearchSession.from_state(state) is None
    # ...and the delivery turn is flagged so the per-message toggle form schema disarms the toggle.
    assert state[StateVarsConfig.DEEP_RESEARCH_REPORT_DELIVERED] is True
    # The agent's second call answered from context, not the raw user text.
    assert captured["messages"][-1][-1] == {
        "role": "user",
        "content": "Countries: US. Frequency: annual.",
    }


async def test_resume_forwards_agent_composed_answer_and_completes(monkeypatch):
    """Resume turn: the agent forwards an answer it composed (mediated, not the verbatim user
    message) and Deep Research delivers the report."""
    _patch_scripted_agent(
        monkeypatch,
        [_tool_call_chunk("resume_deep_research", {"message": "The user approves the plan."})],
    )
    captured: dict = {}
    _patch_dr_deployment(monkeypatch, [_report("Final report.")], captured)

    prior = DeepResearchTurn(user_message="give me US GDP", assistant_content="plan?")
    state = _session_state(prior)

    choice = _RecordingChoice()
    content = await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs(state, "ok", choice=choice)  # terse user text; the agent composes the real message
    )

    assert content == "Final report."
    assert choice.appended == ["Final report."]
    # The prior turn is replayed and the agent's *composed* message is forwarded (not "ok").
    assert captured["messages"][-1] == [
        {"role": "user", "content": "give me US GDP"},
        {"role": "assistant", "content": "plan?", "custom_content": {"state": {}}},
        {"role": "user", "content": "The user approves the plan."},
    ]
    assert DeepResearchSession.from_state(state) is None


async def test_resume_keeps_session_focused_on_unrelated_request(monkeypatch):
    """An unrelated request mid-session: the agent keeps the session focused (replies with text,
    no resume call), the session is preserved, and Deep Research is never invoked."""
    _patch_scripted_agent(
        monkeypatch,
        [_text_chunk("A Deep Research session is in progress. Please answer it or turn it off.")],
    )
    calls = _patch_dr_deployment_counting(monkeypatch)

    prior = DeepResearchTurn(user_message="give me US GDP", assistant_content="Which region?")
    state = _session_state(prior)

    choice = _RecordingChoice()
    content = await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs(state, "actually, what's the weather?", choice=choice)
    )

    assert "Deep Research session is in progress" in content
    assert calls["count"] == 0  # no resume forwarded to the deployment
    assert DeepResearchSession.from_state(state) is not None  # session kept


async def test_toggle_off_mid_session_abandons_and_routes_normally():
    """Turning the toggle off while a session is active drops the session and routes the turn as a
    normal Supreme Agent request."""
    prior = DeepResearchTurn(user_message="give me US GDP", assistant_content="Which region?")
    state = _session_state(prior)
    inputs = _inputs(state, "show me inflation instead", deep_research=False)

    mode = await SupremeAgentExecutor(_channel_config())._resolve_deep_research_mode(inputs)

    assert mode is None  # normal turn
    assert DeepResearchSession.from_state(state) is None  # session abandoned


async def test_routing_modes_are_driven_by_toggle_and_session():
    """Routing is deterministic on the toggle + session flag, never on the message text."""
    executor = SupremeAgentExecutor(_channel_config())

    # toggle on, no session -> START
    assert (
        await executor._resolve_deep_research_mode(_inputs({}, "hi", deep_research=True))
        is _DeepResearchMode.START
    )
    # toggle on, session in progress -> RESUME
    resume_state = _session_state(DeepResearchTurn(user_message="q", assistant_content="a"))
    assert (
        await executor._resolve_deep_research_mode(_inputs(resume_state, "hi", deep_research=True))
        is _DeepResearchMode.RESUME
    )
    # toggle off, no session -> normal
    assert (
        await executor._resolve_deep_research_mode(_inputs({}, "hi", deep_research=False)) is None
    )


async def test_role_gated_caller_without_role_cannot_force_deep_research():
    """When the tool is gated on an `access_claim_value`, a caller lacking that DIAL role can never
    enter a Deep Research turn, even with the toggle forced on."""
    executor = SupremeAgentExecutor(_channel_config(access_claim_value="dr_access"))
    denied = MagicMock(api_key="k", is_system=False)
    denied.has_role = AsyncMock(return_value=False)

    mode = await executor._resolve_deep_research_mode(
        _inputs({}, "research this", deep_research=True, auth_context=denied)
    )

    assert mode is None
    denied.has_role.assert_awaited_once_with("dr_access")


async def test_role_gated_caller_with_role_starts_deep_research():
    """A caller whose DIAL roles include the required role routes normally into Deep Research."""
    executor = SupremeAgentExecutor(_channel_config(access_claim_value="dr_access"))
    granted = MagicMock(api_key="k", is_system=False)
    granted.has_role = AsyncMock(return_value=True)

    mode = await executor._resolve_deep_research_mode(
        _inputs({}, "research this", deep_research=True, auth_context=granted)
    )

    assert mode is _DeepResearchMode.START
    granted.has_role.assert_awaited_once_with("dr_access")


async def test_system_user_bypasses_role_gate_and_enters_deep_research():
    """A system user (used for evaluation, disabled in production) carries no token, so it can't
    satisfy a role gate; it is granted access instead of being denied."""
    executor = SupremeAgentExecutor(_channel_config(access_claim_value="dr_access"))
    system_user = MagicMock(api_key="k", is_system=True)
    system_user.has_role = AsyncMock(return_value=False)

    mode = await executor._resolve_deep_research_mode(
        _inputs({}, "research this", deep_research=True, auth_context=system_user)
    )

    assert mode is _DeepResearchMode.START
    system_user.has_role.assert_not_awaited()


async def test_access_role_resolves_env_var_before_gating(monkeypatch):
    """`access_claim_value` supports $env:{VAR}: the resolved role is what the caller's DIAL roles
    are checked against."""
    monkeypatch.setenv("DR_ROLE", "dr_access")
    executor = SupremeAgentExecutor(_channel_config(access_claim_value="$env:{DR_ROLE}"))
    caller = MagicMock(api_key="k", is_system=False)
    caller.has_role = AsyncMock(return_value=True)

    mode = await executor._resolve_deep_research_mode(
        _inputs({}, "research this", deep_research=True, auth_context=caller)
    )

    assert mode is _DeepResearchMode.START
    caller.has_role.assert_awaited_once_with("dr_access")


async def test_report_delivered_verbatim_with_attachments(monkeypatch):
    """The final report's content and attachments (e.g. a Canvas document) are delivered to the
    user verbatim by the tool."""
    _patch_scripted_agent(
        monkeypatch,
        [_tool_call_chunk("resume_deep_research", {"message": "approved"})],
    )
    attachment = {"type": "text/markdown", "title": "Report.md", "data": "# Report"}
    captured: dict = {}
    _patch_dr_deployment(monkeypatch, [_report("Report body.", attachments=[attachment])], captured)

    prior = DeepResearchTurn(user_message="give me US GDP", assistant_content="plan?")
    state = _session_state(prior)

    choice = _RecordingChoice()
    await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs(state, "approve", choice=choice)
    )

    assert choice.content == "Report body."
    assert len(choice.attachments) == 1
    assert choice.attachments[0]["type"] == "text/markdown"
    assert choice.attachments[0]["title"] == "Report.md"


async def test_forced_start_error_is_surfaced_once_and_session_untouched(monkeypatch):
    """A deployment failure on the forced-start turn is surfaced with the standard message exactly
    once (not double-appended) and leaves no session so the user can retry."""
    _patch_scripted_agent(monkeypatch, [_tool_call_chunk("deep_research", {"query": "q"})])
    _patch_dr_deployment_raises(monkeypatch, OpenAIError("deployment down"))

    choice = _RecordingChoice()
    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    content = await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs(state, "research US GDP", choice=choice)
    )

    assert content == DEEP_RESEARCH_ERROR_MESSAGE
    assert choice.appended == [DEEP_RESEARCH_ERROR_MESSAGE]
    assert DeepResearchSession.from_state(state) is None
    # Delivery never happened, so the toggle is left as-is (armed) for retry.
    assert StateVarsConfig.DEEP_RESEARCH_REPORT_DELIVERED not in state


_DR_REQUEST = httpx.Request("POST", "http://dial/openai/deployments/dr-app/chat/completions")
_DR_DISPLAY_MESSAGE = (
    "A required service is currently rate-limiting requests. Please try again later."
    " (error reference: 016ab904)"
)


@pytest.mark.parametrize(
    "body",
    [
        # Mid-stream error: the SDK passes the error object itself as the body.
        {"message": _DR_DISPLAY_MESSAGE, "display_message": _DR_DISPLAY_MESSAGE, "code": "500"},
        # HTTP error response: DIAL nests the error object under `error`.
        {"error": {"message": "internal", "display_message": _DR_DISPLAY_MESSAGE}},
    ],
)
async def test_deployment_error_aborts_the_turn_with_its_display_message(monkeypatch, body):
    """A deployment (OpenAI) error aborts the turn as `DeepResearchFailedError` carrying the
    deployment's `display_message`, so it is delivered as a DIAL error, not as answer content, and
    leaves no session so the user can retry."""
    _patch_scripted_agent(monkeypatch, [_tool_call_chunk("deep_research", {"query": "q"})])
    _patch_dr_deployment_raises(
        monkeypatch, APIError(_DR_DISPLAY_MESSAGE, request=_DR_REQUEST, body=body)
    )

    choice = _RecordingChoice()
    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    with pytest.raises(DeepResearchFailedError) as exc_info:
        await SupremeAgentExecutor(_channel_config()).stream_response(
            _inputs(state, "research US GDP", choice=choice)
        )

    assert exc_info.value.display_message == _DR_DISPLAY_MESSAGE
    assert choice.appended == []
    assert DeepResearchSession.from_state(state) is None


@pytest.mark.parametrize(
    "body", [None, {"message": "internal"}, {"display_message": "   "}, {"error": "boom"}]
)
async def test_deployment_error_without_display_message_uses_standard_text(monkeypatch, body):
    """Without a usable `display_message` the error carries the standard text; the internal
    `message` is never surfaced."""
    _patch_scripted_agent(monkeypatch, [_tool_call_chunk("deep_research", {"query": "q"})])
    _patch_dr_deployment_raises(monkeypatch, APIError("internal", request=_DR_REQUEST, body=body))

    with pytest.raises(DeepResearchFailedError) as exc_info:
        await SupremeAgentExecutor(_channel_config()).stream_response(
            _inputs({StateVarsConfig.SHOW_DEBUG_STAGES: False}, "research US GDP")
        )

    assert exc_info.value.display_message == DEEP_RESEARCH_ERROR_MESSAGE.strip()


async def test_deep_research_exchange_is_persisted_to_cross_turn_tool_state(monkeypatch):
    """The Deep Research <-> agent tool exchange runs on the main history, so its tool calls/responses
    are persisted into the cross-turn tool-message state and stay visible to later turns."""
    _patch_scripted_agent(
        monkeypatch,
        [
            _tool_call_chunk("deep_research", {"query": "US GDP"}),
            _text_chunk("Which region?"),
        ],
    )
    captured: dict = {}
    _patch_dr_deployment(monkeypatch, [_clarification("Which region and period?")], captured)

    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    inputs = _inputs(state, "research US GDP")
    await SupremeAgentExecutor(_channel_config()).stream_response(inputs)

    # The mediation ran on the main history, so dumping it records the DR tool exchange.
    history = inputs[ChainParametersConfig.HISTORY]
    history.dump_state(state)
    tool_messages = state[StateVarsConfig.TOOL_MESSAGES]
    # The forced-start tool call is retained...
    assert any(
        msg.get("type") == "ai"
        and any(tc["name"] == "deep_research" for tc in msg.get("tool_calls", []))
        for msg in tool_messages
    )
    # ...together with its buffered clarification response.
    assert any(msg.get("type") == "tool" for msg in tool_messages)
    # Regression: the persisted session must not embed a `custom_content` key, which the DIAL
    # chat client strips from message-shaped objects while round-tripping state.
    assert "custom_content" not in json.dumps(state[StateVarsConfig.DEEP_RESEARCH_SESSION])


async def test_mediation_loop_not_bounded_by_max_agent_iterations(monkeypatch):
    """The mediation loop ignores `max_agent_iterations`: with the cap set to 1 it still runs the
    forced start plus several context-answered resumes until Deep Research delivers the report. Under
    the old capped loop this stopped after one pass and never delivered the report. (The loop has its
    own, much larger safety cap instead; see `test_mediation_loop_safety_cap_surfaces_error`.)"""
    channel_config = ChannelConfig(
        supreme_agent=SupremeAgentConfig(
            name="X", domain="d", terminology_domain="t", max_agent_iterations=1
        ),
        deep_research=DeepResearchTool(
            name="deep_research",
            description="DR",
            enabled=True,
            details={"deployment_id": "dr-app"},
        ),
        data_query=DataQueryTool(name="data_query", description="DQ", enabled=True, details={}),
    )
    _patch_scripted_agent(
        monkeypatch,
        [
            _tool_call_chunk("deep_research", {"query": "US GDP"}),  # forced start
            _tool_call_chunk("resume_deep_research", {"message": "US, annual."}, "c2"),
            _tool_call_chunk("resume_deep_research", {"message": "2015-2020."}, "c3"),
            _tool_call_chunk("resume_deep_research", {"message": "Nominal."}, "c4"),
        ],
    )
    captured: dict = {}
    _patch_dr_deployment(
        monkeypatch,
        [
            _clarification("Which countries and frequency?"),
            _clarification("Which time range?"),
            _clarification("Nominal or real GDP?"),
            _report("# Final report\nBody."),
        ],
        captured,
    )

    choice = _RecordingChoice()
    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    content = await SupremeAgentExecutor(channel_config).stream_response(
        _inputs(state, "research US GDP", choice=choice)
    )

    assert content == "# Final report\nBody."
    assert choice.appended == ["# Final report\nBody."]  # only the report reaches the user
    assert len(captured["messages"]) == 4  # four deployment calls, well past the cap of 1
    assert DeepResearchSession.from_state(state) is None  # completed -> session dropped


async def test_mediation_loop_safety_cap_surfaces_error(monkeypatch):
    """The mediation loop is bounded by a safety cap: if Deep Research never delivers the report
    (and the agent never relays to the user), the loop stops at the cap and surfaces the standard
    error exactly once, leaving the session so the user can retry."""
    monkeypatch.setattr(supreme_agent_module, "_MAX_DEEP_RESEARCH_MEDIATION_ITERATIONS", 2)
    _patch_scripted_agent(
        monkeypatch,
        [
            _tool_call_chunk("deep_research", {"query": "US GDP"}),  # forced start
            _tool_call_chunk("resume_deep_research", {"message": "US, annual."}, "c2"),
        ],
    )
    captured: dict = {}
    _patch_dr_deployment(
        monkeypatch,
        [_clarification("Which countries?"), _clarification("Which period?")],
        captured,
    )

    choice = _RecordingChoice()
    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    content = await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs(state, "research US GDP", choice=choice)
    )

    assert content == DEEP_RESEARCH_ERROR_MESSAGE
    assert choice.appended == [DEEP_RESEARCH_ERROR_MESSAGE]  # surfaced exactly once
    assert len(captured["messages"]) == 2  # stopped at the cap, no further deployment calls
    assert DeepResearchSession.from_state(state) is not None  # session kept for retry


async def test_mediation_surfaces_agent_preamble_alongside_tool_call(monkeypatch):
    """If the agent prefixes a tool call with user-facing content, that preamble reaches the user:
    the mediation agent streams to the user-facing choice (as in the main loop), so the content is
    surfaced as it is produced, then the report follows."""
    _patch_scripted_agent(
        monkeypatch,
        [
            _tool_call_chunk(
                "resume_deep_research",
                {"message": "approved"},
                content="One moment while I continue the research.",
            ),
        ],
    )
    captured: dict = {}
    _patch_dr_deployment(monkeypatch, [_report("Final report.")], captured)

    prior = DeepResearchTurn(user_message="give me US GDP", assistant_content="plan?")
    state = _session_state(prior)

    choice = _RecordingChoice()
    content = await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs(state, "approve", choice=choice)
    )

    assert content == "Final report."
    # Preamble is surfaced first, then the report; both reach the user, in order.
    assert choice.appended == ["One moment while I continue the research.", "Final report."]


async def test_clarification_is_surfaced_to_a_stage(monkeypatch):
    """A Deep Research clarification is shown to the user in the tool-result stage (like other
    tools), not in the main answer; the agent still mediates it into the main answer."""
    _patch_scripted_agent(
        monkeypatch,
        [
            _tool_call_chunk("deep_research", {"query": "US GDP"}),
            _text_chunk("Which region?"),
        ],
    )
    captured: dict = {}
    _patch_dr_deployment(
        monkeypatch, [_clarification("Please specify countries and frequency.")], captured
    )

    choice = _StageRecordingChoice()
    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs(state, "research US GDP", choice=choice)
    )

    # The raw clarification is surfaced to a stage...
    assert "Please specify countries and frequency." in "".join(choice.stage_content)
    # ...not to the main answer, which carries only the agent's mediated question.
    assert choice.appended == ["Which region?"]
    assert "Please specify" not in choice.content


async def test_deep_research_turn_runs_fake_tool_prelude(monkeypatch):
    """The fake-tool-call prelude runs for Deep Research turns too, so its tool results are in the
    agent's context and can help answer Deep Research's clarifying questions from context."""
    _patch_scripted_agent(
        monkeypatch,
        [
            _tool_call_chunk("deep_research", {"query": "US GDP"}),
            _text_chunk("Which region?"),
        ],
    )
    captured: dict = {}
    _patch_dr_deployment(monkeypatch, [_clarification("Which region?")], captured)

    # Stub the prelude so the test needs no real tool; assert it is prepended to the history the
    # Deep Research turn runs on.
    async def _fake_prelude(self, tool_executor, inputs, show_stages):
        prelude = History.create_empty()
        prelude.add_dial_message(DialMessage(role=Role.ASSISTANT, content="FAKE_PRELUDE_MARKER"))
        return prelude

    monkeypatch.setattr(SupremeAgentExecutor, "_fake_tool_calls", _fake_prelude)

    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    inputs = _inputs(state, "research US GDP")
    await SupremeAgentExecutor(_channel_config()).stream_response(inputs)

    history = inputs[ChainParametersConfig.HISTORY]
    messages = history.get_langchain_messages(include_tool_messages=False)
    assert any(getattr(m, "content", "") == "FAKE_PRELUDE_MARKER" for m in messages)


# ~~~~~~~~~~~~~~~~ check that the query suits Deep Research (#732) ~~~~~~~~~~~~~~~~


def _system_notes(messages) -> list[str]:
    return [m.content for m in messages if isinstance(m, SystemMessage)]


async def test_query_not_suiting_deep_research_is_handled_as_normal_turn(monkeypatch, query_check):
    """A query that does not suit Deep Research (e.g. "What can you do?") is answered by the general
    agent: Deep Research is never invoked, the user is told it was not started, and the agent is told
    that Deep Research mode was disabled for the request."""
    query_check.return_value = False
    prompts: list = []
    _patch_scripted_agent(monkeypatch, [_text_chunk("I can query datasets.")], prompts)
    calls = _patch_dr_deployment_counting(monkeypatch)

    channel_config = _channel_config()
    skipped_message = channel_config.deep_research.details.query_check.skipped_message
    choice = _RecordingChoice()
    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    inputs = _inputs(state, "What can you do?", choice=choice)
    content = await SupremeAgentExecutor(channel_config).stream_response(inputs)

    assert content == "I can query datasets."
    # The notice precedes the agent's answer.
    assert choice.appended == [f"{skipped_message}\n\n", "I can query datasets."]
    assert calls["count"] == 0
    query_check.assert_awaited_once()
    # No session was started, and the toggle is left as the user set it.
    assert DeepResearchSession.from_state(state) is None
    assert StateVarsConfig.DEEP_RESEARCH_REPORT_DELIVERED not in state
    # The general agent (not the Deep Research mediation) answered, prompted with the disabled note
    # right after the user's query.
    assert prompts[0][-2].content == "What can you do?"
    assert prompts[0][-1] == SystemMessage(content=_DISABLED_NOTE)
    assert "Deep Research Mode" not in prompts[0][0].content
    # The note is persisted with the turn's tool messages, so later turns see it too.
    history = inputs[ChainParametersConfig.HISTORY]
    assert _system_notes(history.get_tool_messages()) == [_DISABLED_NOTE]


async def test_empty_skipped_message_shows_no_notice(monkeypatch, query_check):
    """With an empty `skipped_message`, only the agent's answer reaches the user."""
    query_check.return_value = False
    _patch_scripted_agent(monkeypatch, [_text_chunk("I can query datasets.")])
    _patch_dr_deployment_counting(monkeypatch)

    choice = _RecordingChoice()
    await SupremeAgentExecutor(
        _channel_config(query_check={"skipped_message": ""})
    ).stream_response(
        _inputs({StateVarsConfig.SHOW_DEBUG_STAGES: False}, "What can you do?", choice=choice)
    )

    assert choice.appended == ["I can query datasets."]


async def test_query_suiting_deep_research_starts_session_and_tells_agent(monkeypatch, query_check):
    """A query that suits Deep Research starts a session, and the agent is told that Deep Research
    mode was enabled by the user before it is forced to call the start tool."""
    prompts: list = []
    _patch_scripted_agent(
        monkeypatch,
        [_tool_call_chunk("deep_research", {"query": "US GDP"}), _text_chunk("Which region?")],
        prompts,
    )
    captured: dict = {}
    _patch_dr_deployment(monkeypatch, [_clarification("Which region?")], captured)

    state = {StateVarsConfig.SHOW_DEBUG_STAGES: False}
    inputs = _inputs(state, "research US GDP")
    await SupremeAgentExecutor(_channel_config()).stream_response(inputs)

    query_check.assert_awaited_once()
    assert len(captured["messages"]) == 1  # Deep Research was started
    assert DeepResearchSession.from_state(state) is not None
    # The forced-start run was prompted with the enabled note right after the user's query.
    assert prompts[0][-2].content == "research US GDP"
    assert prompts[0][-1] == SystemMessage(content=_ENABLED_NOTE)
    history = inputs[ChainParametersConfig.HISTORY]
    assert _system_notes(history.get_tool_messages()) == [_ENABLED_NOTE]


async def test_resume_turn_is_not_checked(monkeypatch, query_check):
    """Messages sent while a session is in progress (clarifications, plan approval) are not checked,
    and no Deep Research mode note is added."""
    query_check.return_value = False  # would skip Deep Research if it were consulted
    _patch_scripted_agent(
        monkeypatch, [_tool_call_chunk("resume_deep_research", {"message": "approved"})]
    )
    captured: dict = {}
    _patch_dr_deployment(monkeypatch, [_report("Final report.")], captured)

    prior = DeepResearchTurn(user_message="give me US GDP", assistant_content="plan?")
    inputs = _inputs(_session_state(prior), "ok")
    content = await SupremeAgentExecutor(_channel_config()).stream_response(inputs)

    assert content == "Final report."
    query_check.assert_not_awaited()
    history = inputs[ChainParametersConfig.HISTORY]
    assert _system_notes(history.get_tool_messages()) == []


async def test_disabled_query_check_starts_deep_research_unchecked(monkeypatch, query_check):
    """With the check disabled, every START turn starts Deep Research without consulting the LLM."""
    query_check.return_value = False  # would skip Deep Research if it were consulted
    _patch_scripted_agent(
        monkeypatch,
        [_tool_call_chunk("deep_research", {"query": "q"}), _text_chunk("Which region?")],
    )
    captured: dict = {}
    _patch_dr_deployment(monkeypatch, [_clarification("Which region?")], captured)

    inputs = _inputs({StateVarsConfig.SHOW_DEBUG_STAGES: False}, "What can you do?")
    await SupremeAgentExecutor(_channel_config(query_check={"enabled": False})).stream_response(
        inputs
    )

    query_check.assert_not_awaited()
    assert len(captured["messages"]) == 1
    history = inputs[ChainParametersConfig.HISTORY]
    assert _system_notes(history.get_tool_messages()) == [_ENABLED_NOTE]


async def test_query_check_sees_conversation_without_fake_tool_prelude(monkeypatch, query_check):
    """The check runs before the fake-tool-call prelude is prepended, so it judges only the
    conversation, not the prelude's (possibly long) tool results."""
    seen: list = []

    async def _check(inputs):
        history = ChainParameters.get_history(inputs)
        seen.extend(history.get_langchain_messages(include_tool_messages=False))
        return False

    query_check.side_effect = _check
    _patch_scripted_agent(monkeypatch, [_text_chunk("I can query datasets.")])

    async def _fake_prelude(self, tool_executor, inputs, show_stages):
        prelude = History.create_empty()
        prelude.add_dial_message(DialMessage(role=Role.ASSISTANT, content="FAKE_PRELUDE_MARKER"))
        return prelude

    monkeypatch.setattr(SupremeAgentExecutor, "_fake_tool_calls", _fake_prelude)

    await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs({StateVarsConfig.SHOW_DEBUG_STAGES: False}, "What can you do?")
    )

    assert [m.content for m in seen] == ["What can you do?"]


async def test_query_check_runs_alongside_fake_tool_prelude(monkeypatch, query_check):
    """The check runs concurrently with the fake-tool-call prelude, so it does not add its own
    latency to the turn. Each waits for the other to start, which works only if they overlap."""
    check_started = asyncio.Event()
    prelude_started = asyncio.Event()

    async def _check(inputs):
        check_started.set()
        await asyncio.wait_for(prelude_started.wait(), timeout=1)
        return False

    async def _fake_prelude(self, tool_executor, inputs, show_stages):
        prelude_started.set()
        await asyncio.wait_for(check_started.wait(), timeout=1)
        return History.create_empty()

    query_check.side_effect = _check
    monkeypatch.setattr(SupremeAgentExecutor, "_fake_tool_calls", _fake_prelude)
    _patch_scripted_agent(monkeypatch, [_text_chunk("I can query datasets.")])

    content = await SupremeAgentExecutor(_channel_config()).stream_response(
        _inputs({StateVarsConfig.SHOW_DEBUG_STAGES: False}, "What can you do?")
    )

    assert content == "I can query datasets."


async def test_fake_tool_prelude_failure_cancels_the_query_check(monkeypatch, query_check):
    """If the prelude fails, the check is cancelled instead of being left running past the failed
    turn, and the prelude's error is raised as is (not wrapped in an ExceptionGroup)."""
    check_cancelled = asyncio.Event()

    async def _check(inputs):
        try:
            await asyncio.Event().wait()  # never finishes on its own
        except asyncio.CancelledError:
            check_cancelled.set()
            raise

    async def _failing_prelude(self, tool_executor, inputs, show_stages):
        await asyncio.sleep(0)  # let the check start
        raise RuntimeError("prelude failed")

    query_check.side_effect = _check
    monkeypatch.setattr(SupremeAgentExecutor, "_fake_tool_calls", _failing_prelude)

    with pytest.raises(RuntimeError, match="prelude failed"):
        # Bounded, so a check that is awaited before the prelude fails the test instead of hanging.
        await asyncio.wait_for(
            SupremeAgentExecutor(_channel_config()).stream_response(
                _inputs({StateVarsConfig.SHOW_DEBUG_STAGES: False}, "What can you do?")
            ),
            timeout=1,
        )

    assert check_cancelled.is_set()
