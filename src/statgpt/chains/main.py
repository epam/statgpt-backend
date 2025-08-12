import typing as t

from langchain_core.runnables import Runnable, RunnablePassthrough

from common.config import logger
from common.schemas import ChannelConfig
from common.schemas.dial import Message as DialMessage
from statgpt.chains.out_of_scope_checker import OutOfScopeChecker
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.supreme_agent import SupremeAgentExecutor, ToolCaller
from statgpt.config import DialAppConfig, StateVarsConfig
from statgpt.config.chain_parameters import ChainParametersConfig
from statgpt.schemas import ToolArtifact
from statgpt.utils.message_history import History, dial_tool_call_to_langchain_tool_call


class MainChainFactory:
    def __init__(self, channel_config: ChannelConfig):
        self._channel_config = channel_config

    @staticmethod
    def _init_history(inputs: dict) -> History:
        request = ChainParameters.get_request(inputs)
        state = ChainParameters.get_state(inputs)

        # NOTE: we introduced custom Message model that uses pydantic v2,
        # since aidial_sdk uses pydantic v1 models.
        # the interface should be the same, but this is source of potential bugs.
        dial_messages: list[DialMessage] = t.cast(list[DialMessage], request.messages)

        return History.from_dial_with_commands_interceptor(messages=dial_messages, state=state)

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

        if not DialAppConfig.ENABLE_DIRECT_TOOL_CALLS:
            state[StateVarsConfig.DIRECT_TOOL_CALLS] = None
            return inputs

        history = ChainParameters.get_history(inputs)
        last_msg = history.get_last_non_tool_message()

        if not last_msg.tool_calls:
            state[StateVarsConfig.DIRECT_TOOL_CALLS] = False
            return inputs  # This is a common request, so we skip this chain

        # This is a mode for direct tool calls, so we skip normal processing
        state[StateVarsConfig.DIRECT_TOOL_CALLS] = True
        inputs[ChainParametersConfig.SKIP_OUT_OF_SCOPE_CHECK] = True

        tool_executor = ToolCaller.from_config(self._channel_config)
        for dial_tool_call in last_msg.tool_calls:
            lc_tool_call = dial_tool_call_to_langchain_tool_call(dial_tool_call)
            tool_msg = await tool_executor.call_tool(lc_tool_call, inputs, show_stage=False)
            history.add_tool_message(tool_msg)

        return inputs

    async def _main_chain(self, inputs: dict) -> dict:
        state = ChainParameters.get_state(inputs)

        skip_reason: str = ''

        if state.get(StateVarsConfig.DIRECT_TOOL_CALLS, False):
            skip_reason = "Direct tool calls found"
        elif state.get(StateVarsConfig.CMD_OUT_OF_SCOPE_ONLY, False):
            skip_reason = "CMD_OUT_OF_SCOPE_ONLY is set to True"
        elif ChainParameters.is_out_of_scope(inputs):
            skip_reason = "User message is out of scope"

        if skip_reason:
            logger.info(f"{skip_reason} - skipping the main chain")
            return inputs

        supreme_agent = SupremeAgentExecutor(self._channel_config)
        supreme_agent_chain = await supreme_agent.create_chain()
        return await supreme_agent_chain.ainvoke(inputs)

    @staticmethod
    async def _update_state(inputs: dict) -> dict:
        state = ChainParameters.get_state(inputs)
        history = ChainParameters.get_history(inputs)

        tool_messages = []
        for msg in history.get_tool_messages():
            msg_dump: dict = msg.model_dump(mode='json', exclude={'artifact'}, exclude_none=True)

            if (artifact := getattr(msg, 'artifact', None)) is not None:
                artifact: ToolArtifact
                if msg_dump.get('custom_content') is None:
                    msg_dump['custom_content'] = {}

                msg_dump['custom_content']['state'] = artifact.state.model_dump(mode='json')

            tool_messages.append(msg_dump)

        state[StateVarsConfig.TOOL_MESSAGES] = tool_messages
        state[StateVarsConfig.OUT_OF_SCOPE] = ChainParameters.is_out_of_scope(inputs)
        state[StateVarsConfig.OUT_OF_SCOPE_REASONING] = ChainParameters.get_out_of_scope_reasoning(
            inputs
        )
        return inputs
