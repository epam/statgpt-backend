import asyncio
import logging
from datetime import datetime

import pytest
from aidial_sdk.chat_completion import CustomContent
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from langchain_core.messages import AIMessageChunk, ToolCall, ToolMessage
from langchain_core.runnables import Runnable

from statgpt.app.chains.main import MainChainFactory
from statgpt.app.chains.out_of_scope_checker import OutOfScopeChecker, OutOfScopeCheckerResponse
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.supreme_agent import (
    SupremeAgent,
    SupremeAgentExecutor,
    ToolCaller,
    _SupremeAgentResponse,
)
from statgpt.app.config import ChainParametersConfig, StateVarsConfig
from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.utils.message_history import History
from statgpt.common.schemas.channel import ChannelConfig, OutOfScopeConfig, SupremeAgentConfig
from statgpt.common.schemas.tools import DataQueryTool


class ChoiceLog:
    """Minimal ChoiceI stand-in recording all writes."""

    def __init__(self):
        self.log: list = []

    def create_stage(self, name: str | None = None):
        self.log.append(("create_stage", name))
        raise AssertionError("orchestrator tests never open real stages (debug stages disabled)")

    def append_content(self, content: str):
        self.log.append(("content", content))

    def add_attachment(self, *args, **kwargs):
        self.log.append(("attachment", args, kwargs))

    def set_state(self, state: dict):
        self.log.append(("state", state))


class StageLog:
    """Minimal performance-stage stand-in recording appended content."""

    def __init__(self):
        self.contents: list[str] = []

    def append_content(self, content: str) -> None:
        self.contents.append(content)


@pytest.fixture
def channel_config() -> ChannelConfig:
    return ChannelConfig(
        supreme_agent=SupremeAgentConfig(
            name="StatGPT",
            domain="official statistics",
            terminology_domain="official statistics",
        ),
        out_of_scope=OutOfScopeConfig(domain="official statistics"),
    )


@pytest.fixture
def factory(channel_config: ChannelConfig) -> MainChainFactory:
    return MainChainFactory(channel_config)


def _user_message(content: str) -> DialMessage:
    return DialMessage(role=Role.USER, content=content)


def _out_of_scope_ai_message() -> DialMessage:
    return DialMessage(
        role=Role.ASSISTANT,
        content="This request is out of scope.",
        custom_content=CustomContent(state={StateVarsConfig.OUT_OF_SCOPE: True}),
    )


def _make_inputs(choice: ChoiceLog, history: History | None = None) -> dict:
    return {
        ChainParametersConfig.STATE: {},
        ChainParametersConfig.CHOICE: choice,
        ChainParametersConfig.HISTORY: history or History([_user_message("what is GDP?")]),
        ChainParametersConfig.AUTH_CONTEXT: object(),
        ChainParametersConfig.SKIP_OUT_OF_SCOPE_CHECK: False,
    }


def _patch_check(monkeypatch, coro):
    monkeypatch.setattr(OutOfScopeChecker, "check", coro)


def _patch_check_forbidden(monkeypatch):
    async def forbidden_check(self, messages, auth_context):
        raise AssertionError("checker LLM call must be skipped on this path")

    _patch_check(monkeypatch, forbidden_check)


async def test_in_scope_commits_speculative_run(factory: MainChainFactory, monkeypatch):
    real_choice = ChoiceLog()
    inputs = _make_inputs(real_choice)
    state = inputs[ChainParametersConfig.STATE]
    history = inputs[ChainParametersConfig.HISTORY]

    agent_wrote = asyncio.Event()
    seen: dict = {}

    async def fake_agent(spec_inputs: dict) -> dict:
        # capture the speculative objects before the orchestrator commits them
        seen["state"] = ChainParameters.get_state(spec_inputs)
        seen["history"] = ChainParameters.get_history(spec_inputs)
        ChainParameters.get_choice(spec_inputs).append_content("speculative answer")
        seen["state"]["agent_ran"] = True
        seen["history"].add_dial_message(_user_message("tool output"))
        agent_wrote.set()
        # mirror the real side-effect gate: wait for the orchestrator's permission
        await spec_inputs[ChainParametersConfig.SIDE_EFFECT_GATE].wait()
        return spec_inputs

    async def fake_check(self, messages, auth_context) -> OutOfScopeCheckerResponse:
        await agent_wrote.wait()
        # the speculative output is buffered: nothing on the real choice yet
        assert real_choice.log == []
        return OutOfScopeCheckerResponse(reasoning="stats question", out_of_scope=False)

    monkeypatch.setattr(factory, "_main_chain", fake_agent)
    _patch_check(monkeypatch, fake_check)

    result = await factory._guarded_main_chain(inputs)

    assert real_choice.log == [("content", "speculative answer")]  # buffer flushed
    assert result[ChainParametersConfig.OUT_OF_SCOPE] is False
    assert result[ChainParametersConfig.OUT_OF_SCOPE_REASONING] == "stats question"
    assert result[ChainParametersConfig.CHOICE] is real_choice
    assert ChainParametersConfig.SIDE_EFFECT_GATE not in result

    # state committed in place: same dict object, speculative mutations applied
    assert result[ChainParametersConfig.STATE] is state
    assert state["agent_ran"] is True

    # the agent ran on isolated copies; the committed history is the speculative one
    assert seen["history"] is not history
    assert seen["state"] is not state
    assert result[ChainParametersConfig.HISTORY] is seen["history"]
    assert history.get_last_non_tool_message().content == "what is GDP?"


async def test_out_of_scope_cancels_agent_and_discards_buffer(
    factory: MainChainFactory, monkeypatch
):
    real_choice = ChoiceLog()
    inputs = _make_inputs(real_choice)
    state = inputs[ChainParametersConfig.STATE]
    history = inputs[ChainParametersConfig.HISTORY]

    agent_wrote = asyncio.Event()
    agent_cancelled = asyncio.Event()

    async def fake_agent(spec_inputs: dict) -> dict:
        ChainParameters.get_choice(spec_inputs).append_content("speculative answer")
        ChainParameters.get_state(spec_inputs)["agent_ran"] = True
        agent_wrote.set()
        try:
            await spec_inputs[ChainParametersConfig.SIDE_EFFECT_GATE].wait()
        except asyncio.CancelledError:
            agent_cancelled.set()
            raise
        return spec_inputs

    async def fake_check(self, messages, auth_context) -> OutOfScopeCheckerResponse:
        await agent_wrote.wait()
        return OutOfScopeCheckerResponse(reasoning="off-domain request", out_of_scope=True)

    async def fake_respond(self, respond_inputs: dict, reasoning: str) -> dict:
        ChainParameters.get_choice(respond_inputs).append_content("out-of-scope message")
        return respond_inputs

    monkeypatch.setattr(factory, "_main_chain", fake_agent)
    _patch_check(monkeypatch, fake_check)
    monkeypatch.setattr(OutOfScopeChecker, "respond_out_of_scope", fake_respond)

    result = await factory._guarded_main_chain(inputs)

    assert agent_cancelled.is_set()
    # no speculative content leaked; only the out-of-scope response is visible
    assert real_choice.log == [("content", "out-of-scope message")]
    assert result is inputs
    assert result[ChainParametersConfig.OUT_OF_SCOPE] is True
    assert result[ChainParametersConfig.OUT_OF_SCOPE_REASONING] == "off-domain request"
    # original state and history untouched by the speculative run
    assert "agent_ran" not in state
    assert history.get_tool_messages() == []


async def test_checker_failure_cancels_agent_and_reraises(factory: MainChainFactory, monkeypatch):
    real_choice = ChoiceLog()
    inputs = _make_inputs(real_choice)

    agent_cancelled = asyncio.Event()

    async def fake_agent(spec_inputs: dict) -> dict:
        try:
            await spec_inputs[ChainParametersConfig.SIDE_EFFECT_GATE].wait()
        except asyncio.CancelledError:
            agent_cancelled.set()
            raise
        return spec_inputs

    async def failing_check(self, messages, auth_context):
        raise RuntimeError("checker down")

    monkeypatch.setattr(factory, "_main_chain", fake_agent)
    _patch_check(monkeypatch, failing_check)

    with pytest.raises(RuntimeError, match="checker down"):
        await factory._guarded_main_chain(inputs)

    assert agent_cancelled.is_set()
    assert real_choice.log == []


async def test_skip_check_flag_runs_main_chain_sequentially(factory: MainChainFactory, monkeypatch):
    real_choice = ChoiceLog()
    inputs = _make_inputs(real_choice)
    inputs[ChainParametersConfig.SKIP_OUT_OF_SCOPE_CHECK] = True

    seen: dict = {}

    async def fake_agent(agent_inputs: dict) -> dict:
        seen["inputs"] = agent_inputs
        return agent_inputs

    monkeypatch.setattr(factory, "_main_chain", fake_agent)
    _patch_check_forbidden(monkeypatch)

    result = await factory._guarded_main_chain(inputs)

    assert seen["inputs"] is inputs  # no speculative copies on the skip path
    assert result[ChainParametersConfig.OUT_OF_SCOPE] is None
    assert result[ChainParametersConfig.OUT_OF_SCOPE_REASONING] == 'guardrails disabled in config'


async def test_direct_tool_calls_run_main_chain_sequentially(
    factory: MainChainFactory, monkeypatch
):
    real_choice = ChoiceLog()
    inputs = _make_inputs(real_choice)
    inputs[ChainParametersConfig.STATE][StateVarsConfig.DIRECT_TOOL_CALLS] = [{"name": "tool"}]

    seen: dict = {}

    async def fake_agent(agent_inputs: dict) -> dict:
        seen["inputs"] = agent_inputs
        return agent_inputs

    monkeypatch.setattr(factory, "_main_chain", fake_agent)
    _patch_check_forbidden(monkeypatch)

    result = await factory._guarded_main_chain(inputs)

    assert seen["inputs"] is inputs
    assert result[ChainParametersConfig.OUT_OF_SCOPE] is None
    assert (
        result[ChainParametersConfig.OUT_OF_SCOPE_REASONING]
        == "direct tool calls found - skipping guardrails"
    )


async def test_threshold_exceeded_starts_new_conversation_sequentially(
    factory: MainChainFactory, channel_config: ChannelConfig, monkeypatch
):
    threshold = channel_config.out_of_scope.start_new_conversation_messages_threshold
    messages: list[DialMessage] = []
    for _ in range(threshold + 1):
        messages.append(_out_of_scope_ai_message())
    messages.append(_user_message("still off topic"))

    real_choice = ChoiceLog()
    inputs = _make_inputs(real_choice, history=History(messages))

    seen: dict = {}

    async def fake_agent(agent_inputs: dict) -> dict:
        seen["inputs"] = agent_inputs
        return agent_inputs

    monkeypatch.setattr(factory, "_main_chain", fake_agent)
    _patch_check_forbidden(monkeypatch)

    result = await factory._guarded_main_chain(inputs)

    assert seen["inputs"] is inputs
    assert result[ChainParametersConfig.OUT_OF_SCOPE] is True
    start_message = channel_config.out_of_scope.start_new_conversation_message
    assert real_choice.log == [("content", start_message)]


async def test_cmd_out_of_scope_only_stays_sequential(factory: MainChainFactory, monkeypatch):
    real_choice = ChoiceLog()
    inputs = _make_inputs(real_choice)
    inputs[ChainParametersConfig.STATE][StateVarsConfig.CMD_OUT_OF_SCOPE_ONLY] = True

    async def fake_check(self, messages, auth_context) -> OutOfScopeCheckerResponse:
        return OutOfScopeCheckerResponse(reasoning="off-domain request", out_of_scope=True)

    async def forbidden_agent(agent_inputs: dict) -> dict:
        raise AssertionError("main chain must not run on the checker-only path")

    monkeypatch.setattr(factory, "_main_chain", forbidden_agent)
    _patch_check(monkeypatch, fake_check)

    result = await factory._guarded_main_chain(inputs)

    assert result[ChainParametersConfig.OUT_OF_SCOPE] is True
    assert result[ChainParametersConfig.OUT_OF_SCOPE_REASONING] == "off-domain request"
    assert real_choice.log == []  # verdict only: no out-of-scope response streamed


# ~~~ speculative performance-stage substitution (orchestrator level) ~~~


async def test_perf_rows_buffered_until_commit(factory: MainChainFactory, monkeypatch):
    real_choice = ChoiceLog()
    inputs = _make_inputs(real_choice)
    perf_stage = StageLog()
    inputs[ChainParametersConfig.PERFORMANCE_STAGE] = perf_stage

    agent_wrote = asyncio.Event()

    async def fake_agent(spec_inputs: dict) -> dict:
        # the agent writes perf rows to the (proxied) shared performance stage
        ChainParameters.get_performance_stage(spec_inputs).append_content("perf-row")
        agent_wrote.set()
        await spec_inputs[ChainParametersConfig.SIDE_EFFECT_GATE].wait()
        return spec_inputs

    async def fake_check(self, messages, auth_context) -> OutOfScopeCheckerResponse:
        await agent_wrote.wait()
        # perf rows are held back through the recording, not on the real stage yet
        assert perf_stage.contents == []
        return OutOfScopeCheckerResponse(reasoning="stats question", out_of_scope=False)

    monkeypatch.setattr(factory, "_main_chain", fake_agent)
    _patch_check(monkeypatch, fake_check)

    await factory._guarded_main_chain(inputs)

    assert perf_stage.contents == ["perf-row"]  # flushed on commit
    # the real perf stage was proxied, never recreated on the real choice
    assert real_choice.log == []


async def test_perf_rows_discarded_on_out_of_scope(factory: MainChainFactory, monkeypatch):
    real_choice = ChoiceLog()
    inputs = _make_inputs(real_choice)
    perf_stage = StageLog()
    inputs[ChainParametersConfig.PERFORMANCE_STAGE] = perf_stage

    agent_wrote = asyncio.Event()

    async def fake_agent(spec_inputs: dict) -> dict:
        ChainParameters.get_performance_stage(spec_inputs).append_content("perf-row")
        agent_wrote.set()
        await spec_inputs[ChainParametersConfig.SIDE_EFFECT_GATE].wait()
        return spec_inputs

    async def fake_check(self, messages, auth_context) -> OutOfScopeCheckerResponse:
        await agent_wrote.wait()
        return OutOfScopeCheckerResponse(reasoning="off-domain request", out_of_scope=True)

    async def fake_respond(self, respond_inputs: dict, reasoning: str) -> dict:
        return respond_inputs

    monkeypatch.setattr(factory, "_main_chain", fake_agent)
    _patch_check(monkeypatch, fake_check)
    monkeypatch.setattr(OutOfScopeChecker, "respond_out_of_scope", fake_respond)

    await factory._guarded_main_chain(inputs)

    assert perf_stage.contents == []  # discarded speculative run: no perf rows


# ~~~ abnormal-exit reaping and observability ~~~


async def test_agent_failure_after_commit_propagates_without_state_commit(
    factory: MainChainFactory, monkeypatch
):
    real_choice = ChoiceLog()
    inputs = _make_inputs(real_choice)
    state = inputs[ChainParametersConfig.STATE]

    agent_wrote = asyncio.Event()

    async def fake_agent(spec_inputs: dict) -> dict:
        ChainParameters.get_choice(spec_inputs).append_content("speculative answer")
        ChainParameters.get_state(spec_inputs)["agent_ran"] = True
        agent_wrote.set()
        await spec_inputs[ChainParametersConfig.SIDE_EFFECT_GATE].wait()
        raise RuntimeError("agent exploded after commit")

    async def fake_check(self, messages, auth_context) -> OutOfScopeCheckerResponse:
        await agent_wrote.wait()
        return OutOfScopeCheckerResponse(reasoning="stats question", out_of_scope=False)

    monkeypatch.setattr(factory, "_main_chain", fake_agent)
    _patch_check(monkeypatch, fake_check)

    with pytest.raises(RuntimeError, match="agent exploded after commit"):
        await factory._guarded_main_chain(inputs)

    # the in-scope verdict flushed the buffered prefix before the agent failed
    assert real_choice.log == [("content", "speculative answer")]
    # but the speculative state was never committed onto the original dict
    assert "agent_ran" not in state


async def test_oos_path_logs_speculative_agent_failure(
    factory: MainChainFactory, monkeypatch, caplog
):
    real_choice = ChoiceLog()
    inputs = _make_inputs(real_choice)

    agent_done = asyncio.Event()

    async def fake_agent(spec_inputs: dict) -> dict:
        try:
            raise RuntimeError("speculative agent bug")
        finally:
            agent_done.set()

    async def fake_check(self, messages, auth_context) -> OutOfScopeCheckerResponse:
        await agent_done.wait()
        return OutOfScopeCheckerResponse(reasoning="off-domain request", out_of_scope=True)

    async def fake_respond(self, respond_inputs: dict, reasoning: str) -> dict:
        ChainParameters.get_choice(respond_inputs).append_content("out-of-scope message")
        return respond_inputs

    monkeypatch.setattr(factory, "_main_chain", fake_agent)
    _patch_check(monkeypatch, fake_check)
    monkeypatch.setattr(OutOfScopeChecker, "respond_out_of_scope", fake_respond)

    with caplog.at_level(logging.WARNING, logger="statgpt"):
        result = await factory._guarded_main_chain(inputs)

    # the swallowed speculative failure is surfaced, not lost
    assert any("speculative agent run failed" in r.message for r in caplog.records)
    # the user still gets a clean out-of-scope response
    assert real_choice.log == [("content", "out-of-scope message")]
    assert result[ChainParametersConfig.OUT_OF_SCOPE] is True


async def test_flush_failure_reaps_agent_task(factory: MainChainFactory, monkeypatch):
    class ExplodingChoice(ChoiceLog):
        def append_content(self, content: str):
            raise RuntimeError("choice closed")

    real_choice = ExplodingChoice()
    inputs = _make_inputs(real_choice)
    state = inputs[ChainParametersConfig.STATE]

    agent_wrote = asyncio.Event()
    agent_cancelled = asyncio.Event()

    async def fake_agent(spec_inputs: dict) -> dict:
        ChainParameters.get_choice(spec_inputs).append_content("speculative answer")
        ChainParameters.get_state(spec_inputs)["agent_ran"] = True
        agent_wrote.set()
        try:
            # the gate is set before flush_to raises; the reap-cancel then lands
            # here at the gate await, which is where the run must observe it
            await spec_inputs[ChainParametersConfig.SIDE_EFFECT_GATE].wait()
        except asyncio.CancelledError:
            agent_cancelled.set()
            raise
        return spec_inputs

    async def fake_check(self, messages, auth_context) -> OutOfScopeCheckerResponse:
        await agent_wrote.wait()
        return OutOfScopeCheckerResponse(reasoning="stats question", out_of_scope=False)

    monkeypatch.setattr(factory, "_main_chain", fake_agent)
    _patch_check(monkeypatch, fake_check)

    # flush_to raises mid-replay after the gate was set; the live agent must be reaped
    with pytest.raises(RuntimeError, match="choice closed"):
        await factory._guarded_main_chain(inputs)

    assert agent_cancelled.is_set()
    assert "agent_ran" not in state  # speculative state never committed


# ~~~ create_chain composition (per-channel out_of_scope.optimistic flag) ~~~


def _chain_afuncs(chain: Runnable) -> list:
    """Async functions of the top-level steps of a RunnableSequence."""
    return [afunc for step in chain.steps if (afunc := getattr(step, "afunc", None)) is not None]


async def test_create_chain_composes_guarded_orchestrator_when_optimistic(
    channel_config: ChannelConfig,
):
    channel_config.out_of_scope.optimistic = True
    factory = MainChainFactory(channel_config)

    afuncs = _chain_afuncs(await factory.create_chain())

    assert factory._guarded_main_chain in afuncs
    # the main chain runs inside the orchestrator, not as a top-level step,
    # and there is no sequential checker step
    assert factory._main_chain not in afuncs
    assert all(
        getattr(afunc, "__func__", None) is not OutOfScopeChecker.stream_response
        for afunc in afuncs
    )


async def test_create_chain_composes_sequential_checker_when_not_optimistic(
    factory: MainChainFactory,
):
    # channel_config.out_of_scope.optimistic defaults to False
    afuncs = _chain_afuncs(await factory.create_chain())

    assert factory._guarded_main_chain not in afuncs
    assert factory._main_chain in afuncs
    assert any(
        getattr(afunc, "__func__", None) is OutOfScopeChecker.stream_response for afunc in afuncs
    )


async def test_create_chain_composes_sequential_checker_when_out_of_scope_absent(
    channel_config: ChannelConfig,
):
    channel_config.out_of_scope = None
    factory = MainChainFactory(channel_config)

    afuncs = _chain_afuncs(await factory.create_chain())

    assert factory._guarded_main_chain not in afuncs
    assert factory._main_chain in afuncs


# ~~~ SupremeAgentExecutor side-effect gate (real stream_response loop) ~~~


class ScriptedSupremeAgent:
    """LLM boundary stub: each ``run`` call returns the next scripted turn."""

    def __init__(self, turns: list[_SupremeAgentResponse]):
        self._turns = list(turns)
        self.turns_started = 0

    async def run(self, history, configuration) -> _SupremeAgentResponse:
        self.turns_started += 1
        return self._turns.pop(0)


class RecordingToolCaller:
    """Tool boundary stub recording every dispatched tool call."""

    def __init__(self):
        self.tools: list = []
        self.calls: list[str] = []

    async def call_tool(self, tool_call, inputs, show_stage=True, prefix='') -> ToolMessage:
        self.calls.append(tool_call["name"])
        return ToolMessage(content="tool result", tool_call_id=tool_call["id"])


def _tool_call_turn() -> _SupremeAgentResponse:
    now = datetime.now()
    resp = AIMessageChunk(
        content="",
        tool_calls=[ToolCall(name="Data_Query", args={}, id="call-1", type="tool_call")],
    )
    return _SupremeAgentResponse(start_time=now, first_token_time=None, resp=resp, finished=True)


def _final_turn() -> _SupremeAgentResponse:
    now = datetime.now()
    resp = AIMessageChunk(content="final answer")
    return _SupremeAgentResponse(start_time=now, first_token_time=now, resp=resp, finished=True)


@pytest.fixture
def agent_channel_config(channel_config: ChannelConfig) -> ChannelConfig:
    # stream_response requires a configured data_query tool
    return channel_config.model_copy(
        update={"data_query": DataQueryTool(name="Data_Query", description="Query data.")}
    )


def _make_executor(
    monkeypatch, channel_config: ChannelConfig, turns: list[_SupremeAgentResponse]
) -> tuple[SupremeAgentExecutor, RecordingToolCaller, ScriptedSupremeAgent]:
    tool_caller = RecordingToolCaller()
    agent = ScriptedSupremeAgent(turns)
    monkeypatch.setattr(ToolCaller, "from_config", lambda config: tool_caller)
    monkeypatch.setattr(SupremeAgent, "create", lambda choice, auth_context, config, tools: agent)
    return SupremeAgentExecutor(channel_config), tool_caller, agent


def _agent_inputs(gate: asyncio.Event | None = None) -> dict:
    inputs = {
        ChainParametersConfig.STATE: {},
        ChainParametersConfig.CHOICE: ChoiceLog(),
        ChainParametersConfig.HISTORY: History([_user_message("what is GDP?")]),
        ChainParametersConfig.AUTH_CONTEXT: object(),
        ChainParametersConfig.CONFIGURATION: StatGPTConfiguration(),
        ChainParametersConfig.PERFORMANCE_STAGE: StageLog(),
        ChainParametersConfig.START_OF_REQUEST: datetime.now(),
    }
    if gate is not None:
        inputs[ChainParametersConfig.SIDE_EFFECT_GATE] = gate
    return inputs


async def _settle() -> None:
    """Let all currently runnable tasks progress until they block."""
    for _ in range(20):
        await asyncio.sleep(0)


async def test_agent_tool_dispatch_waits_for_gate(agent_channel_config: ChannelConfig, monkeypatch):
    executor, tool_caller, agent = _make_executor(
        monkeypatch, agent_channel_config, [_tool_call_turn(), _final_turn()]
    )
    gate = asyncio.Event()

    task = asyncio.create_task(executor.stream_response(_agent_inputs(gate)))
    await _settle()

    # turn 1 finished (checker slower than the agent), yet the gate holds
    assert agent.turns_started == 1
    assert tool_caller.calls == []

    gate.set()
    result = await task

    assert tool_caller.calls == ["Data_Query"]
    assert agent.turns_started == 2
    assert result == "final answer"


async def test_agent_cancelled_while_gated_never_dispatches_tools(
    agent_channel_config: ChannelConfig, monkeypatch
):
    executor, tool_caller, agent = _make_executor(
        monkeypatch, agent_channel_config, [_tool_call_turn(), _final_turn()]
    )
    # out-of-scope verdict: the gate is never set, the task gets cancelled
    task = asyncio.create_task(executor.stream_response(_agent_inputs(asyncio.Event())))
    await _settle()

    assert agent.turns_started == 1
    assert tool_caller.calls == []

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tool_caller.calls == []


async def test_agent_without_gate_runs_ungated(agent_channel_config: ChannelConfig, monkeypatch):
    executor, tool_caller, agent = _make_executor(
        monkeypatch, agent_channel_config, [_tool_call_turn(), _final_turn()]
    )

    result = await executor.stream_response(_agent_inputs())

    assert tool_caller.calls == ["Data_Query"]
    assert agent.turns_started == 2
    assert result == "final answer"
