from langchain_core.runnables import Runnable, RunnablePassthrough

from statgpt.app.chains.out_of_scope_checker import OutOfScopeChecker
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.supreme_agent import SupremeAgentExecutor, ToolCaller
from statgpt.app.config.chain_parameters import ChainParametersConfig
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.app.utils.message_history import History, dial_tool_call_to_langchain_tool_call
from statgpt.common.config import logger
from statgpt.common.schemas import ChannelConfig


class MainChainFactory:
    def __init__(self, channel_config: ChannelConfig):
        self._channel_config = channel_config

    @staticmethod
    async def _init_history(inputs: dict) -> History:
        request = ChainParameters.get_request(inputs)
        state = ChainParameters.get_state(inputs)
        data_service = ChainParameters.get_data_service(inputs)

        return await History.from_dial_with_interceptors(
            messages=request.messages, state=state, data_service=data_service
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
            state.direct_tool_calls = []
            return inputs

        history = ChainParameters.get_history(inputs)
        last_msg = history.get_last_non_tool_message()

        tool_calls_received = last_msg.tool_calls

        if not tool_calls_received:
            state.direct_tool_calls = []
            return inputs  # This is a common request, so we skip direct tool calls chain

        # parse tool calls to langchain format
        tool_calls_parsed = []
        for dial_tool_call in tool_calls_received:
            lc_tool_call = dial_tool_call_to_langchain_tool_call(dial_tool_call)
            tool_calls_parsed.append(lc_tool_call)

        state.direct_tool_calls = tool_calls_parsed

        tool_executor = ToolCaller.from_config(self._channel_config)
        for tool_call in tool_calls_parsed:
            tool_msg = await tool_executor.call_tool(tool_call, inputs, show_stage=False)
            history.add_tool_message(tool_msg)

        return inputs

    async def _main_chain(self, inputs: dict) -> dict:
        state = ChainParameters.get_state(inputs)

        skip_reason: str = ''

        if state.direct_tool_calls:
            skip_reason = "Direct tool calls found"
        elif state.cmd_out_of_scope_only:
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

        state.out_of_scope = ChainParameters.is_out_of_scope(inputs)
        state.out_of_scope_reasoning = ChainParameters.get_out_of_scope_reasoning(inputs)
        return inputs
