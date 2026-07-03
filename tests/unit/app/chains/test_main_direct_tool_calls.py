"""Unit tests for MainChainFactory._direct_tool_calls_chain dispatch."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import ToolMessage

from statgpt.app.chains.main import MainChainFactory
from statgpt.app.config import StateVarsConfig
from statgpt.app.config.chain_parameters import ChainParametersConfig
from statgpt.app.settings.dial_app import dial_app_settings


def _dial_tool_call(id_: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments="{}"))


def _inputs_with_tool_calls(tool_calls: list[SimpleNamespace]) -> tuple[dict, Mock]:
    history = Mock()
    history.get_last_non_tool_message.return_value = SimpleNamespace(tool_calls=tool_calls)
    inputs = {ChainParametersConfig.STATE: {}, ChainParametersConfig.HISTORY: history}
    return inputs, history


async def test_no_tool_calls_skips_dispatch(monkeypatch):
    monkeypatch.setattr(dial_app_settings, "enable_direct_tool_calls", True)
    inputs, history = _inputs_with_tool_calls([])

    factory = MainChainFactory(channel_config=Mock())
    result = await factory._direct_tool_calls_chain(inputs)

    assert result is inputs
    assert inputs[ChainParametersConfig.STATE][StateVarsConfig.DIRECT_TOOL_CALLS] == []
    history.add_tool_message.assert_not_called()


async def test_dispatches_concurrently_and_preserves_order(monkeypatch):
    monkeypatch.setattr(dial_app_settings, "enable_direct_tool_calls", True)

    second_started = asyncio.Event()

    async def call_tool(tool_call, inputs, show_stage=True, prefix=''):
        if tool_call["id"] == "call-1":
            # Resolves only after the second call has started: sequential
            # dispatch would hang here (bounded by the wait_for timeouts).
            await asyncio.wait_for(second_started.wait(), timeout=1)
        else:
            second_started.set()
        return ToolMessage(content=f"result {tool_call['id']}", tool_call_id=tool_call["id"])

    monkeypatch.setattr(
        "statgpt.app.chains.main.ToolCaller",
        SimpleNamespace(from_config=lambda config: SimpleNamespace(call_tool=call_tool)),
    )
    inputs, history = _inputs_with_tool_calls(
        [_dial_tool_call("call-1", "tool_a"), _dial_tool_call("call-2", "tool_b")]
    )

    factory = MainChainFactory(channel_config=Mock())
    result = await asyncio.wait_for(factory._direct_tool_calls_chain(inputs), timeout=1)

    assert result is inputs
    state = inputs[ChainParametersConfig.STATE]
    assert [tc["id"] for tc in state[StateVarsConfig.DIRECT_TOOL_CALLS]] == ["call-1", "call-2"]
    # results are appended to history in the original tool-call order
    added_ids = [call.args[0].tool_call_id for call in history.add_tool_message.call_args_list]
    assert added_ids == ["call-1", "call-2"]
