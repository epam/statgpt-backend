import asyncio

from aidial_sdk.exceptions import InvalidRequestError
from langchain_core.runnables import Runnable, RunnableLambda

from statgpt.app.chains.data_query.eval_attachments_displayer import (
    DataQueryEvalAttachmentsDisplayer,
)
from statgpt.app.chains.out_of_scope_checker import OutOfScopeChecker
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.supreme_agent import SupremeAgentExecutor, ToolCaller
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.app.utils.message_history import (
    InvalidToolCallError,
    dial_tool_call_to_langchain_tool_call,
)
from statgpt.common.config import logger
from statgpt.common.schemas import ChannelConfig


class MainChainFactory:
    def __init__(self, channel_config: ChannelConfig):
        self._channel_config = channel_config

    async def create_chain(self) -> Runnable:
        out_of_scope_checker = OutOfScopeChecker(self._channel_config)
        out_of_scope_chain = await out_of_scope_checker.create_chain()

        return (
            RunnableLambda(self._direct_tool_calls_chain)
            | out_of_scope_chain
            | RunnableLambda(self._main_chain)
            | RunnableLambda(self._update_state)
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
        try:
            tool_calls_parsed = [
                dial_tool_call_to_langchain_tool_call(tool_call)
                for tool_call in tool_calls_received
            ]
        except InvalidToolCallError as e:
            # The caller sent these tool calls in the current request,
            # so this is a bad request rather than a corrupted history.
            raise InvalidRequestError(str(e)) from e

        state[StateVarsConfig.DIRECT_TOOL_CALLS] = tool_calls_parsed

        tool_executor = ToolCaller.from_config(self._channel_config)
        # Dispatch concurrently with bare gather to mirror the agent path
        # (supreme_agent); gather preserves the original tool-call order for
        # history. Accepted trade-off: on first failure sibling tool calls keep
        # running detached instead of being cancelled - same as the agent path.
        tool_messages = await asyncio.gather(
            *(
                tool_executor.call_tool(tool_call, inputs, show_stage=False)
                for tool_call in tool_calls_parsed
            )
        )
        data_query_artifacts: dict[str, DataQueryArtifact] = {}
        for tool_msg in tool_messages:
            history.add_tool_message(tool_msg)

            artifact = tool_msg.artifact
            if artifact and isinstance(artifact, DataQueryArtifact):
                data_query_artifacts[tool_msg.tool_call_id] = artifact

        if data_query_artifacts:
            # Eval attachments only: a direct caller renders the data itself from the tool
            # message, so the data attachments (table, plotly, CSV, python) would be noise.
            eval_displayer = DataQueryEvalAttachmentsDisplayer(
                choice=ChainParameters.get_choice(inputs),
                auth_context=ChainParameters.get_auth_context(inputs),
                enabled=ChainParameters.get_configuration(inputs).enable_debug_attachments,
            )
            await eval_displayer.display(data_query_artifacts)

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
