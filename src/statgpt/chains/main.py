import typing as t

from langchain_core.runnables import Runnable, RunnablePassthrough

from common.config import logger
from common.schemas import ChannelConfig
from common.schemas.dial import Message as DialMessage
from statgpt.chains.out_of_scope_checker import OutOfScopeChecker
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.supreme_agent import SupremeAgentExecutor, ToolCaller
from statgpt.config import StateVarsConfig
from statgpt.config.chain_parameters import ChainParametersConfig
from statgpt.settings.dial_app import dial_app_settings
from statgpt.utils.message_history import History, dial_tool_call_to_langchain_tool_call


class MainChainFactory:
    def __init__(self, channel_config: ChannelConfig):
        self._channel_config = channel_config

    @staticmethod
    async def _init_history(inputs: dict) -> History:
        request = ChainParameters.get_request(inputs)
        state = ChainParameters.get_state(inputs)
        data_service = ChainParameters.get_data_service(inputs)

        # NOTE: we introduced custom Message model that uses pydantic v2,
        # since aidial_sdk uses pydantic v1 models.
        # the interface should be the same, but this is source of potential bugs.
        dial_messages: list[DialMessage] = t.cast(list[DialMessage], request.messages)

        return await History.from_dial_with_interceptors(
            messages=dial_messages, state=state, data_service=data_service
        )

    async def create_chain(self) -> Runnable:
        out_of_scope_checker = OutOfScopeChecker(self._channel_config)
        out_of_scope_chain = await out_of_scope_checker.create_chain()

        return (
            RunnablePassthrough.assign(**{ChainParametersConfig.HISTORY: self._init_history})
            | self._direct_tool_calls_chain
            | out_of_scope_chain
            | self._main_chain
            | self._update_state
        )

    async def _direct_tool_calls_chain(self, inputs: dict) -> dict:
        state = ChainParameters.get_state(inputs)

        if not dial_app_settings.enable_direct_tool_calls:
            state[StateVarsConfig.DIRECT_TOOL_CALLS] = []
            return inputs

        history = ChainParameters.get_history(inputs)
        last_msg = history.get_last_non_tool_message()

        tool_calls_received = last_msg.tool_calls

        if not tool_calls_received:
            state[StateVarsConfig.DIRECT_TOOL_CALLS] = []
            return inputs  # This is a common request, so we skip direct tool calls chain

        # parse tool calls to langchain format
        tool_calls_parsed = []
        for dial_tool_call in tool_calls_received:
            lc_tool_call = dial_tool_call_to_langchain_tool_call(dial_tool_call)
            tool_calls_parsed.append(lc_tool_call)

        state[StateVarsConfig.DIRECT_TOOL_CALLS] = tool_calls_parsed

        tool_executor = ToolCaller.from_config(self._channel_config)
        for tool_call in tool_calls_parsed:
            tool_msg = await tool_executor.call_tool(tool_call, inputs, show_stage=False)
            history.add_tool_message(tool_msg)

        return inputs

    async def _main_chain(self, inputs: dict) -> dict:
        state = ChainParameters.get_state(inputs)

        skip_reason: str = ''

        if state.get(StateVarsConfig.DIRECT_TOOL_CALLS, []):
            skip_reason = "Direct tool calls found"
        elif state.get(StateVarsConfig.CMD_OUT_OF_SCOPE_ONLY, False):
            skip_reason = "CMD_OUT_OF_SCOPE_ONLY is set to True"
        elif ChainParameters.is_out_of_scope(inputs):
            skip_reason = "User message is out of scope"

        if skip_reason:
            logger.info(f"skipping the main chain, reason: {skip_reason}")
            return inputs

        supreme_agent = SupremeAgentExecutor(self._channel_config)
        supreme_agent_chain = await supreme_agent.create_chain()
        return await supreme_agent_chain.ainvoke(inputs)

    @staticmethod
    async def _update_state(inputs: dict) -> dict:
        state = ChainParameters.get_state(inputs)
        history = ChainParameters.get_history(inputs)
        history.dump_state(state)

        state[StateVarsConfig.OUT_OF_SCOPE] = ChainParameters.is_out_of_scope(inputs)
        state[StateVarsConfig.OUT_OF_SCOPE_REASONING] = ChainParameters.get_out_of_scope_reasoning(
            inputs
        )
        return inputs
