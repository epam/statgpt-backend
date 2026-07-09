import asyncio
import copy

from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from statgpt.app.chains.out_of_scope_checker import OutOfScopeChecker
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.supreme_agent import SupremeAgentExecutor, ToolCaller
from statgpt.app.config import StateVarsConfig
from statgpt.app.config.chain_parameters import ChainParametersConfig
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.app.utils.buffered_choice import BufferedChoice
from statgpt.app.utils.dial_stages import optional_timed_stage
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
        if dial_app_settings.optimistic_guardrails:
            return (
                RunnablePassthrough.assign(**{ChainParametersConfig.HISTORY: self._init_history})
                | self._direct_tool_calls_chain
                | self._guarded_main_chain
                | self._update_state
            )

        out_of_scope_checker = OutOfScopeChecker(self._channel_config)
        out_of_scope_chain = await out_of_scope_checker.create_chain()

        return (
            RunnablePassthrough.assign(**{ChainParametersConfig.HISTORY: self._init_history})
            | RunnableLambda(self._direct_tool_calls_chain)
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
        tool_calls_parsed = []
        for dial_tool_call in tool_calls_received:
            lc_tool_call = dial_tool_call_to_langchain_tool_call(dial_tool_call)
            tool_calls_parsed.append(lc_tool_call)

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
        for tool_msg in tool_messages:
            history.add_tool_message(tool_msg)

        return inputs

    async def _guarded_main_chain(self, inputs: dict) -> dict:
        """Run the out-of-scope check concurrently with a speculative agent run.

        The agent runs against a buffered choice, a copied history, and a
        deepcopied state; its real tool dispatch is gated on the verdict event.
        In scope: flush the buffer, commit history/state, await the agent —
        visible output matches the sequential flow, minus one LLM round-trip of
        latency. Out of scope: cancel the agent, discard the buffer, and stream
        the out-of-scope response exactly as the sequential flow does.
        """
        checker = OutOfScopeChecker(self._channel_config)
        if (resolved := checker.resolve_skip(inputs)) is not None:
            return await self._main_chain(resolved)

        state = ChainParameters.get_state(inputs)
        if state.get(StateVarsConfig.CMD_OUT_OF_SCOPE_ONLY, False):
            # Checker-only debug path: nothing to speculate on, stay sequential.
            return await checker.stream_response(inputs)

        history = ChainParameters.get_history(inputs)
        choice = ChainParameters.get_choice(inputs)
        auth_context = ChainParameters.get_auth_context(inputs)

        verdict_event = asyncio.Event()
        buffered = BufferedChoice()
        spec_inputs = {
            **inputs,
            ChainParametersConfig.CHOICE: buffered,
            ChainParametersConfig.HISTORY: history.copy(),
            ChainParametersConfig.STATE: copy.deepcopy(state),
            ChainParametersConfig.OOS_VERDICT_EVENT: verdict_event,
            # speculative run assumes the request is in scope
            ChainParametersConfig.OUT_OF_SCOPE: False,
            ChainParametersConfig.OUT_OF_SCOPE_REASONING: None,
        }
        messages = history.get_langchain_messages(include_tool_messages=False)

        checker_task = asyncio.create_task(checker.check(messages, auth_context))
        agent_task = asyncio.create_task(self._main_chain(spec_inputs))

        try:
            show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False)
            with optional_timed_stage(
                choice, "[DEBUG] Guardrails: Relevancy", enabled=show_debug_stages
            ) as stage:
                response = await checker_task
                checker.append_verdict_to_stage(stage, response)
        except BaseException:
            # Checker failed: kill the speculative run and surface the checker
            # error, matching the sequential flow.
            agent_task.cancel()
            await asyncio.gather(agent_task, return_exceptions=True)
            raise

        inputs[ChainParametersConfig.OUT_OF_SCOPE] = response.out_of_scope
        inputs[ChainParametersConfig.OUT_OF_SCOPE_REASONING] = response.reasoning

        if not response.out_of_scope:
            verdict_event.set()  # unblock the agent's real tool dispatch
            buffered.flush_to(choice)
            result = await agent_task  # propagate agent errors normally
            # Commit the speculative state by mutating the original dict in
            # place: channel_completion holds a reference to it for set_dial_state.
            state.clear()
            state.update(result[ChainParametersConfig.STATE])
            result[ChainParametersConfig.STATE] = state
            result[ChainParametersConfig.CHOICE] = choice
            result[ChainParametersConfig.OUT_OF_SCOPE] = response.out_of_scope
            result[ChainParametersConfig.OUT_OF_SCOPE_REASONING] = response.reasoning
            result.pop(ChainParametersConfig.OOS_VERDICT_EVENT, None)
            return result  # committed speculative inputs (incl. history)

        agent_task.cancel()
        await asyncio.gather(agent_task, return_exceptions=True)
        buffered.discard()
        return await checker.respond_out_of_scope(inputs, response.reasoning)

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
