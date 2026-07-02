import asyncio

import pytest
from aidial_sdk.chat_completion import CustomContent
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role

from statgpt.app.chains.main import MainChainFactory
from statgpt.app.chains.out_of_scope_checker import OutOfScopeChecker, OutOfScopeCheckerResponse
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.config import ChainParametersConfig, StateVarsConfig
from statgpt.app.utils.message_history import History
from statgpt.common.schemas.channel import ChannelConfig, OutOfScopeConfig, SupremeAgentConfig


class RecordingChoice:
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


def _make_inputs(choice: RecordingChoice, history: History | None = None) -> dict:
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
    real_choice = RecordingChoice()
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
        # mirror the real tool-dispatch gate: wait for the verdict
        await spec_inputs[ChainParametersConfig.OOS_VERDICT_EVENT].wait()
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
    assert ChainParametersConfig.OOS_VERDICT_EVENT not in result

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
    real_choice = RecordingChoice()
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
            await spec_inputs[ChainParametersConfig.OOS_VERDICT_EVENT].wait()
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
    real_choice = RecordingChoice()
    inputs = _make_inputs(real_choice)

    agent_cancelled = asyncio.Event()

    async def fake_agent(spec_inputs: dict) -> dict:
        try:
            await spec_inputs[ChainParametersConfig.OOS_VERDICT_EVENT].wait()
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
    real_choice = RecordingChoice()
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
    real_choice = RecordingChoice()
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

    real_choice = RecordingChoice()
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
    real_choice = RecordingChoice()
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
