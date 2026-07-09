import asyncio
import copy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from statgpt.app.chains.out_of_scope_checker import OutOfScopeChecker
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.supreme_agent import SupremeAgentExecutor, ToolCaller
from statgpt.app.config import StateVarsConfig
from statgpt.app.config.chain_parameters import ChainParametersConfig
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.app.utils.message_history import History, dial_tool_call_to_langchain_tool_call
from statgpt.app.utils.recording_choice import RecordingChoice
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
        out_of_scope = self._channel_config.out_of_scope
        if out_of_scope is not None and out_of_scope.optimistic:
            return (
                RunnablePassthrough.assign(**{ChainParametersConfig.HISTORY: self._init_history})
                | RunnableLambda(self._direct_tool_calls_chain)
                | RunnableLambda(self._guarded_main_chain)
                | RunnableLambda(self._update_state)
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

        The agent runs against a recording choice, a copied history, and a
        deepcopied state; its irreversible side effects (real tool dispatch) are
        gated on a permission event. In scope: flush the recording, commit
        history/state, await the agent — visible output matches the sequential
        flow, minus one LLM round-trip of latency. Out of scope: cancel the
        agent, discard the recording, and stream the out-of-scope response
        exactly as the sequential flow does.
        """
        checker = OutOfScopeChecker(self._channel_config)
        if (resolved := checker.try_short_circuit(inputs)) is not None:
            return await self._main_chain(resolved)

        state = ChainParameters.get_state(inputs)
        if state.get(StateVarsConfig.CMD_OUT_OF_SCOPE_ONLY, False):
            # Checker-only debug path: nothing to speculate on, stay sequential.
            return await checker.stream_response(inputs)

        history = ChainParameters.get_history(inputs)
        choice = ChainParameters.get_choice(inputs)

        gate = asyncio.Event()
        recording = RecordingChoice()
        # Isolation invariant: every real-choice-derived object placed on
        # spec_inputs must be substituted (choice/history/state below) or routed
        # through the recording (adopt_stage below). Anything left pointing at the
        # real choice would let speculative writes escape before the in-scope
        # verdict commits.
        spec_inputs = {
            **inputs,
            ChainParametersConfig.CHOICE: recording,
            ChainParametersConfig.HISTORY: history.copy(),
            ChainParametersConfig.STATE: copy.deepcopy(state),
            ChainParametersConfig.SIDE_EFFECT_GATE: gate,
            # speculative run assumes the request is in scope
            ChainParametersConfig.OUT_OF_SCOPE: False,
            ChainParametersConfig.OUT_OF_SCOPE_REASONING: None,
        }
        # The performance stage lives on the real choice; route it through the
        # recording so speculative rows are held back until the run is committed.
        # A falsy stage (debug off) is left as-is — _log_performance early-returns.
        perf_stage = inputs.get(ChainParametersConfig.PERFORMANCE_STAGE)
        if perf_stage:
            spec_inputs[ChainParametersConfig.PERFORMANCE_STAGE] = recording.adopt_stage(perf_stage)

        checker_task = asyncio.create_task(checker.check_with_stage(inputs))
        agent_task = asyncio.create_task(self._main_chain(spec_inputs))

        # The context manager guarantees that, however this block exits (commit,
        # out-of-scope, checker failure, cancellation, or flush_to raising
        # mid-replay), neither task outlives the call and no un-flushed
        # speculative output escapes.
        async with self._speculative_cleanup((checker_task, agent_task), recording):
            response = await checker_task

            if not response.out_of_scope:
                logger.info("optimistic guardrails: in-scope verdict, committing speculative run")
                gate.set()  # permit the agent's real tool dispatch
                recording.flush_to(choice)
                result = await agent_task  # propagate agent errors normally
                # Commit the speculative state by mutating the original dict in
                # place: channel_completion holds a reference to it for set_dial_state.
                state.clear()
                state.update(result[ChainParametersConfig.STATE])
                result[ChainParametersConfig.STATE] = state
                result[ChainParametersConfig.CHOICE] = choice
                result[ChainParametersConfig.OUT_OF_SCOPE] = response.out_of_scope
                result[ChainParametersConfig.OUT_OF_SCOPE_REASONING] = response.reasoning
                result.pop(ChainParametersConfig.SIDE_EFFECT_GATE, None)
                return result  # committed speculative inputs (incl. history)

            # Out of scope: stop the speculative run now (before the out-of-scope
            # LLM response) and surface any genuine agent failure, which would
            # otherwise be invisible on out-of-scope traffic.
            logger.info("optimistic guardrails: out-of-scope verdict, discarding speculative run")
            agent_task.cancel()
            outcome = (await asyncio.gather(agent_task, return_exceptions=True))[0]
            if isinstance(outcome, BaseException) and not isinstance(
                outcome, asyncio.CancelledError
            ):
                logger.warning(
                    "speculative agent run failed before the out-of-scope abort",
                    exc_info=outcome,
                )
            # inputs carries the OUT_OF_SCOPE keys written by check_with_stage.
            return await checker.respond_out_of_scope(inputs, response.reasoning)

    @staticmethod
    @asynccontextmanager
    async def _speculative_cleanup(
        tasks: tuple[asyncio.Task, ...], recording: RecordingChoice
    ) -> AsyncIterator[None]:
        """Ensure no speculative work outlives the block.

        On exit — normal or exceptional — cancel and reap any still-running task,
        then drop any un-flushed buffer. All three are no-ops on the committed
        path (tasks already awaited, recording already in pass-through).
        """
        try:
            yield
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            recording.discard()

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
