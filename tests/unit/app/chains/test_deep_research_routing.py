"""Routing + mediation tests for the Deep Research flow in the Supreme Agent (#575).

Deep Research is excluded from ``ChannelConfig.tool_fields`` and is reachable only via the
``deep_research`` toggle. A Deep Research turn runs in its own mediated loop
(``SupremeAgentExecutor._run_deep_research_turn``), bound only to the Deep Research tools:

- clarifications / plans are **buffered** and handed back to the agent as a tool response, never
  streamed to the user as-is;
- the agent answers from context via the resume tool and surfaces only the remainder;
- the final report is delivered to the user **verbatim** by the tool, and the turn ends without the
  agent repeating it;
- the Deep Research <-> agent exchange is turn-local and must not pollute the cross-turn tool state.

These tests pin that behaviour and the deterministic toggle/session routing.
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
from statgpt.app.chains.supreme_agent import SupremeAgentExecutor, _DeepResearchMode
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


def _patch_scripted_agent(monkeypatch, responses: list[AIMessageChunk]) -> None:
    """Drive the Supreme Agent LLM with a scripted sequence of responses, one per agent run.

    A single shared model instance is returned for every ``get_chat_model`` call so the script is
    consumed in order across the forced-start and free mediation agents."""

    scripted = iter(responses)

    class _SharedModel:
        def bind_tools(self, tools, **kwargs):
            return RunnableLambda(lambda _inp: next(scripted))

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
        deep_research_module.openai, "get_async_client", lambda api_key=None, **k: _FakeClient()
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
        deep_research_module.openai, "get_async_client", lambda api_key=None, **k: _RaisingClient()
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
        deep_research_module.openai, "get_async_client", lambda api_key=None, **k: _FakeClient()
    )
    return calls


def _inputs(state: dict, user_text: str, *, deep_research: bool = True, choice=None) -> dict:
    return {
        ChainParametersConfig.STATE: state,
        ChainParametersConfig.CHOICE: choice if choice is not None else _RecordingChoice(),
        ChainParametersConfig.AUTH_CONTEXT: MagicMock(api_key="k"),
        ChainParametersConfig.HISTORY: History(
            messages=[DialMessage(role=Role.USER, content=user_text)]
        ),
        ChainParametersConfig.CONFIGURATION: StatGPTConfiguration(deep_research=deep_research),
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


def test_toggle_off_mid_session_abandons_and_routes_normally():
    """Turning the toggle off while a session is active drops the session and routes the turn as a
    normal Supreme Agent request."""
    prior = DeepResearchTurn(user_message="give me US GDP", assistant_content="Which region?")
    state = _session_state(prior)
    inputs = _inputs(state, "show me inflation instead", deep_research=False)

    mode = SupremeAgentExecutor(_channel_config())._resolve_deep_research_mode(inputs)

    assert mode is None  # normal turn
    assert DeepResearchSession.from_state(state) is None  # session abandoned


def test_routing_modes_are_driven_by_toggle_and_session():
    """Routing is deterministic on the toggle + session flag, never on the message text."""
    executor = SupremeAgentExecutor(_channel_config())

    # toggle on, no session -> START
    assert (
        executor._resolve_deep_research_mode(_inputs({}, "hi", deep_research=True))
        is _DeepResearchMode.START
    )
    # toggle on, session in progress -> RESUME
    resume_state = _session_state(DeepResearchTurn(user_message="q", assistant_content="a"))
    assert (
        executor._resolve_deep_research_mode(_inputs(resume_state, "hi", deep_research=True))
        is _DeepResearchMode.RESUME
    )
    # toggle off, no session -> normal
    assert executor._resolve_deep_research_mode(_inputs({}, "hi", deep_research=False)) is None


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


async def test_deep_research_exchange_does_not_pollute_cross_turn_tool_state(monkeypatch):
    """The Deep Research <-> agent tool exchange is turn-local: it is not persisted to the
    cross-turn tool-message state (only the separate session is)."""
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

    # The main history was forked for the mediation, so dumping it records no tool messages.
    history = inputs[ChainParametersConfig.HISTORY]
    history.dump_state(state)
    assert state[StateVarsConfig.TOOL_MESSAGES] == []
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
    """If the agent prefixes a tool call with user-facing content, that preamble is surfaced rather
    than lost: the mediation agent streams to a null choice, so unlike the main loop nothing else
    would show it to the user."""
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
