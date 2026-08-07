import asyncio
import copy
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, NamedTuple

from aidial_sdk.chat_completion import FunctionCall
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from aidial_sdk.chat_completion import ToolCall as DialToolCall
from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage, ToolCall, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnablePassthrough

from statgpt.app.chains.data_query.data_query_artifacts_displayer import DataQueryArtifactDisplayer
from statgpt.app.chains.deep_research import ResumeDeepResearchTool
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import StatGptTool
from statgpt.app.config import ChainParametersConfig, StateVarsConfig
from statgpt.app.default_prompts import supreme_agent_default_prompts
from statgpt.app.schemas import (
    DEEP_RESEARCH_ERROR_MESSAGE,
    DeepResearchSession,
    FailedToolArtifact,
    FailedToolMessageState,
    ToolArtifact,
    ToolMessageState,
    ToolResponseStatus,
)
from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.app.utils.dial_stages import (
    ChoiceI,
    optional_delayed_timed_stage,
    optional_timed_stage,
)
from statgpt.app.utils.message_history import History
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.schemas import ChannelConfig, FakeCall, ToolTypes
from statgpt.common.utils import InvalidLLMStreamResponse
from statgpt.common.utils.llm_call_duration_context import get_llm_call_duration_manager
from statgpt.common.utils.markdown import format_as_markdown_list
from statgpt.common.utils.models import get_chat_model

FAKE_TOOL_STAGE_PREFIX = '[FAKE TOOL] '


class ToolCaller:
    def __init__(self, tools: list[StatGptTool]):
        self._tools = {tool.name: tool for tool in tools}

    @classmethod
    def from_config(cls, channel_config: ChannelConfig) -> 'ToolCaller':
        tools = cls.get_tools_from_config(channel_config)
        return cls(tools)

    @property
    def tools(self) -> list[StatGptTool]:
        return list(self._tools.values())

    @staticmethod
    def get_tools_from_config(channel_config: ChannelConfig) -> list[StatGptTool]:
        # `agent_tools` excludes `mcp_only` tools, which are surfaced exclusively via the
        # MCP server (e.g. UI-initiated tool calls) and must not be exposed to the LLM.
        return [
            StatGptTool.from_config(tool_cfg, channel_config)
            for tool_cfg in channel_config.agent_tools
        ]

    @staticmethod
    def _format_stage_name(name: str, args: dict) -> str:
        if not args:
            return name
        result = {}
        # replace all lists with ", ".join(list)
        for k, v in args.items():
            if isinstance(v, list):
                result[k] = ", ".join(map(str, v))
            else:
                result[k] = str(v)
        try:
            return name.format(**result)
        except KeyError as e:
            logger.warning(f"Error formatting stage name: {e}")
            return name

    async def call_tool(
        self, tool_call: ToolCall, inputs: dict, show_stage: bool = True, prefix: str = ''
    ) -> ToolMessage:
        tool_call = deepcopy(tool_call)  # to prevent edits to incoming tool call object

        tool = self._tools[tool_call['name']]

        choice = ChainParameters.get_choice(inputs)

        formatted_stage_name = self._format_stage_name(tool.stage_name, tool_call['args'])
        formatted_result_stage_name = self._format_stage_name(
            tool.result_stage_name, tool_call['args']
        )
        tool_call_name = f"{prefix}{formatted_stage_name}"
        tool_result_name = f"{prefix}{formatted_result_stage_name}"

        with optional_timed_stage(choice=choice, name=tool_call_name, enabled=show_stage):
            logger.debug(f"Calling tool: {tool.name}")
            # NOTE: deepcopy raises errors - something with pydantic.
            # shallow copy seems to be enough here.
            inputs = copy.copy(inputs)

            tool_call['args']['inputs'] = inputs  # inject tool argument

            with optional_delayed_timed_stage(
                choice=choice, name=tool_result_name, enabled=show_stage
            ) as tool_result_stage:
                inputs[ChainParametersConfig.TARGET] = tool_result_stage
                # logger.info(f"{tool_call=}")
                state = ChainParameters.get_state(inputs)
                skip_tools_execution = (
                    state.get(StateVarsConfig.CMD_SKIP_TOOLS_EXECUTION, False)
                    and prefix != FAKE_TOOL_STAGE_PREFIX
                )
                if skip_tools_execution:
                    return ToolMessage(
                        content="tool execution skipped due to configuration",
                        tool_call_id=tool_call["id"],
                        status=ToolResponseStatus.SUCCESS.value,
                        artifact=ToolArtifact(state=ToolMessageState(type=tool.tool_type)),
                    )
                try:
                    tool_msg: ToolMessage = await tool.ainvoke(tool_call)
                except Exception as e:
                    logger.exception(f"Error calling tool {tool.name}:\n{e}")
                    return ToolMessage(
                        content=f"{tool.tool_type} tool failed to execute. error: {repr(e)}",
                        tool_call_id=tool_call['id'],
                        artifact=FailedToolArtifact(
                            state=FailedToolMessageState(type=tool.tool_type, error=repr(e))
                        ),
                        status=ToolResponseStatus.ERROR.value,
                    )

        logger.info(f"Tool message: {tool_msg!r}")
        return tool_msg


class _SupremeAgentResponse(NamedTuple):
    start_time: datetime
    first_token_time: datetime | None
    resp: AIMessageChunk
    finished: bool


class SupremeAgent:
    def __init__(self, choice: ChoiceI, chain: Runnable):
        self._choice = choice
        self._chain = chain

    @classmethod
    def create(
        cls,
        choice: ChoiceI,
        auth_context: AuthContext,
        channel_config: ChannelConfig,
        tools: list[StatGptTool],
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> 'SupremeAgent':
        chain = cls._create_chain(
            auth_context, channel_config, tools, tool_choice, parallel_tool_calls
        )
        return cls(choice, chain)

    async def run(
        self, history: History, configuration: StatGPTConfiguration
    ) -> _SupremeAgentResponse:
        chunk: AIMessageChunk
        resp: AIMessageChunk | None = None

        inputs = {
            'chat_history': history.get_langchain_messages(include_tool_messages=True),
            'today_date': configuration.get_current_date(),
        }
        first_token_time = None
        start_time = datetime.now()
        try:
            async for chunk in self._chain.astream(inputs):
                if chunk.content:
                    if not isinstance(chunk.content, str):
                        continue
                    if first_token_time is None:
                        first_token_time = datetime.now()
                    self._choice.append_content(chunk.content)

                if resp is None:
                    resp = chunk
                else:
                    resp = resp + chunk  # type: ignore

            finished = True
        except InvalidLLMStreamResponse as e:
            logger.warning(f"Error in the response from Supreme Agent: {e}")
            if resp is None:
                raise

            user_msg = f"<!-- delete_chars({len(resp.content) + 2}) -->"
            self._choice.append_content(f"\n\n{user_msg}\n\n")

            llm_msg = (
                "\n\n[ERROR] The response contained too many consecutive spaces and was terminated."
            )
            resp.content = str(resp.content) + llm_msg
            finished = False

        assert resp is not None, "No response chunks received from LLM"
        return _SupremeAgentResponse(start_time, first_token_time, resp, finished)

    @classmethod
    def _create_system_prompt(cls, channel_config: ChannelConfig) -> str:
        template = supreme_agent_default_prompts.system_prompt

        if channel_config.supreme_agent.additional_context:
            template += "\n\n" + supreme_agent_default_prompts.additional_context_wrapper_section

        return template

    @classmethod
    def _create_chain(
        cls,
        auth_context: AuthContext,
        channel_config: ChannelConfig,
        tools: list[StatGptTool],
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> Runnable:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    cls._create_system_prompt(channel_config)
                ),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
            ]
        ).partial(
            chat_bot_name=channel_config.supreme_agent.name,
            chat_bot_domain=channel_config.supreme_agent.domain,
            chat_bot_terminology_domain=channel_config.supreme_agent.terminology_domain,
            chat_bot_language_instructions=format_as_markdown_list(
                channel_config.supreme_agent.language_instructions, list_type="ordered"
            ),
            additional_context=channel_config.supreme_agent.additional_context,
            general_section=(
                channel_config.supreme_agent.general_section
                or supreme_agent_default_prompts.default_general_section
            ),
            tool_usage_section=(
                channel_config.supreme_agent.tool_usage_section
                or supreme_agent_default_prompts.default_tool_usage_section
            ),
            no_calculations_section=(
                channel_config.supreme_agent.no_calculations_section
                or supreme_agent_default_prompts.default_no_calculations_section
            ),
        )
        model = get_chat_model(
            api_key=auth_context.api_key,
            model_config=channel_config.supreme_agent.llm_model_config,
        )
        bind_kwargs: dict[str, Any] = {'strict': True}
        if tool_choice is not None:
            bind_kwargs['tool_choice'] = tool_choice
        if parallel_tool_calls is not None:
            bind_kwargs['parallel_tool_calls'] = parallel_tool_calls
        model_with_tools = model.bind_tools(tools, **bind_kwargs)

        return prompt_template | model_with_tools


class SupremeAgentExecutor:
    def __init__(self, channel_config: ChannelConfig):
        self._channel_config = channel_config

    async def stream_response(self, inputs: dict) -> str:
        auth_context = ChainParameters.get_auth_context(inputs)
        choice = ChainParameters.get_choice(inputs)
        history = ChainParameters.get_history(inputs)
        state = ChainParameters.get_state(inputs)
        configuration = ChainParameters.get_configuration(inputs)
        debug = state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False)

        tool_executor = ToolCaller.from_config(self._channel_config)

        fake_history = await self._fake_tool_calls(tool_executor, inputs, show_stages=debug)
        history.prepend(fake_history)

        # ~~~ Deep Research routing ~~~
        # Deep Research is deliberately excluded from `ChannelConfig.tool_fields`, so it never
        # appears in `tool_executor.tools`, the MCP server, or the out-of-scope checker. It is
        # reachable only via the `deep_research` toggle ("button"): we build the (start) tool on
        # demand from its config, and resume is derived from it in turn.
        deep_research_tool: StatGptTool | None = None
        if self._channel_config.is_deep_research_available:
            dr_config = self._channel_config.deep_research
            assert dr_config is not None  # guaranteed by is_deep_research_available
            deep_research_tool = StatGptTool.from_config(dr_config, self._channel_config)
        # A session lives in state only while a run is in progress (the tool drops it once the
        # report is delivered), so its presence is the in-progress signal.
        session_in_progress = DeepResearchSession.from_state(state) is not None

        agent_tools = list(tool_executor.tools)

        # The `deep_research` toggle (advertised to the frontend only where the tool is enabled)
        # is the single switch for Deep Research: it gates both starting a new session and
        # resuming an in-progress one. When on, an active session is resumed and otherwise the
        # start tool is forced as the only bound tool; when off, Deep Research is fully
        # unavailable and any in-progress session is abandoned.
        tool_choice: str | None = None
        parallel_tool_calls: bool | None = None
        if deep_research_tool is not None:
            if configuration.deep_research:
                if session_in_progress:
                    # Active session + toggle on: forward the user's next message to Deep Research
                    # verbatim (no interpretation) and surface its reply as-is. Deterministic
                    # routing driven by the session flag, so a resume never depends on the LLM
                    # re-composing the message.
                    return await self._resume_deep_research(inputs, deep_research_tool)
                # Toggle on, fresh request: force the start tool as the only bound tool, and
                # disallow parallel calls so the model cannot emit more than one Deep Research
                # call in a single response.
                agent_tools = [deep_research_tool]
                tool_choice = deep_research_tool.name
                parallel_tool_calls = False
            elif session_in_progress:
                # Toggle turned off mid-session: abandon the run so it can't silently resume,
                # or resurrect on a later turn if the user re-enables the toggle in this chat.
                DeepResearchSession.drop_from_state(state)

        supreme_agent = SupremeAgent.create(
            choice,
            auth_context,
            self._channel_config,
            agent_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        data_query_artifacts: dict[str, DataQueryArtifact] = {}
        assert self._channel_config.data_query is not None, "data_query must be configured"
        data_displayer = DataQueryArtifactDisplayer(
            choice=choice,
            config=self._channel_config.data_query.details.attachments,
            chat_config=ChainParameters.get_configuration(inputs),
            max_cells=self._channel_config.data_query.details.tool_response_max_cells,
            auth_context=auth_context,
        )

        state = ChainParameters.get_state(inputs)
        skip_tools_execution = state.get(StateVarsConfig.CMD_SKIP_TOOLS_EXECUTION, False)

        for i in range(self._channel_config.supreme_agent.max_agent_iterations):
            name = f"[DEBUG] Supreme Agent run {i + 1}"
            with optional_timed_stage(choice=choice, name=name, enabled=debug):
                response = await supreme_agent.run(history, configuration)
                logger.info(f"Response: {response.resp!r}")

            tool_data_query_artifacts: list[DataQueryArtifact] = []

            if tool_calls := response.resp.tool_calls:
                history.add_chunk_as_tool_message(response.resp)

                res: list[ToolMessage] = await asyncio.gather(
                    *(tool_executor.call_tool(tool_call, inputs) for tool_call in tool_calls)
                )

                deep_research_msg: ToolMessage | None = None
                for tool_msg in res:
                    history.add_tool_message(tool_msg)

                    artifact = tool_msg.artifact
                    # Deep Research is force-selected as the sole bound tool with
                    # `parallel_tool_calls=False`, so at most one Deep Research message appears here.
                    if artifact and artifact.type == ToolTypes.DEEP_RESEARCH:
                        deep_research_msg = tool_msg

                    if artifact and isinstance(artifact, DataQueryArtifact):
                        tool_data_query_artifacts.append(artifact)
                        data_query_artifacts[tool_msg.tool_call_id] = artifact

                if tool_data_query_artifacts:
                    # await data_displayer.display(tool_data_query_artifacts)
                    content = await data_displayer.get_system_message_content(
                        tool_data_query_artifacts
                    )
                    history.add_tool_message(SystemMessage(content=content))

                if skip_tools_execution:
                    logger.info(
                        "Further supreme agent runs are skipped due to tools execution skip"
                    )
                    return ""

                if deep_research_msg is not None:
                    # Deep Research owns this turn: it streamed its clarifying question / final
                    # report straight into the conversation, or failed. Either way, end the loop —
                    # under the forced tool_choice, continuing would just re-invoke Deep Research
                    # and, on a persistent failure, spin until max_agent_iterations.
                    if deep_research_msg.status == ToolResponseStatus.ERROR.value:
                        choice.append_content(DEEP_RESEARCH_ERROR_MESSAGE)
                        return DEEP_RESEARCH_ERROR_MESSAGE
                    dr_content = deep_research_msg.content
                    return dr_content if isinstance(dr_content, str) else str(dr_content)
            elif response.finished:
                if response.first_token_time is not None:
                    self._log_performance(inputs, response.start_time, response.first_token_time)

                if data_query_artifacts:
                    await data_displayer.display(data_query_artifacts)

                # Handle response content - can be str or list
                resp_content = response.resp.content
                if isinstance(resp_content, str):
                    return resp_content
                else:
                    # Convert list content to string
                    return str(resp_content)
            else:
                # Not finished due to error and no tool calls
                # Let's add the chunk to history and try again (if retries are left)
                history.add_chunk_as_tool_message(response.resp)

        warning_msg = '\n\n[WARNING] Maximum number of agent loop iterations reached. Please enter "continue" to proceed.'
        choice.append_content(warning_msg)
        return warning_msg

    async def _resume_deep_research(self, inputs: dict, start_tool: StatGptTool) -> str:
        """Forward the user's latest message to the active Deep Research session, verbatim.

        Bypasses the LLM so the message reaches Deep Research exactly as the user wrote it — no
        paraphrasing, no answering from context (that mediation is the follow-up, #575). The tool
        streams Deep Research's reply straight to the user and updates the session in `state`."""
        history = ChainParameters.get_history(inputs)
        choice = ChainParameters.get_choice(inputs)

        resume_tool = ResumeDeepResearchTool.build(start_tool._tool_config, self._channel_config)
        user_message = history.last_user_message_text
        tool_call: ToolCall = {
            'name': resume_tool.name,
            'args': {'message': user_message},
            'id': str(uuid.uuid4()),
            'type': 'tool_call',
        }
        tool_msg = await ToolCaller([resume_tool]).call_tool(tool_call, inputs, show_stage=False)
        if tool_msg.status == ToolResponseStatus.ERROR.value:
            choice.append_content(DEEP_RESEARCH_ERROR_MESSAGE)
            return DEEP_RESEARCH_ERROR_MESSAGE
        content = tool_msg.content
        return content if isinstance(content, str) else str(content)

    async def create_chain(self) -> Runnable:
        return RunnablePassthrough.assign(general_response=self.stream_response)

    async def _fake_tool_calls(
        self, tool_executor: ToolCaller, inputs: dict, show_stages: bool
    ) -> History:
        fake_history = History.create_empty()
        tool_calls = []
        tasks = []

        for tool_cfg in self._channel_config.agent_tools:
            if (fake_call := tool_cfg.details.fake_call) is not None:
                tool_call = self._get_tool_call_from_cfg(tool_cfg.name, fake_call)
                tool_calls.append(tool_call)
                last_msg = History.dial_to_langchain_message(tool_call)
                assert isinstance(last_msg, AIMessage), "Fake tool call must convert to AIMessage"
                tasks.append(
                    tool_executor.call_tool(
                        last_msg.tool_calls[0],
                        inputs,
                        show_stage=show_stages,
                        prefix=FAKE_TOOL_STAGE_PREFIX,
                    )
                )

        if tasks:
            tool_messages = await asyncio.gather(*tasks)
            for tool_call, tool_msg in zip(tool_calls, tool_messages):
                fake_history.add_dial_message(tool_call)
                fake_history.add_tool_message_as_dial_message(tool_msg)

        return fake_history

    @staticmethod
    def _get_tool_call_from_cfg(tool_name, fake_call: FakeCall) -> DialMessage:
        return DialMessage(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                DialToolCall(
                    index=None,
                    id=fake_call.tool_call_id,
                    type="function",
                    function=FunctionCall(name=tool_name, arguments=fake_call.args),
                )
            ],
        )

    @staticmethod
    def _log_performance(
        inputs: dict, start_time_of_last_request: datetime, first_token_time: datetime
    ) -> None:
        performance_stage = ChainParameters.get_performance_stage(inputs)
        if not performance_stage:
            return
        start_time = ChainParameters.get_start_of_request(inputs)

        performance_stage.append_content(
            "| Name | Start Time | End Time | Duration (s) | Description |\n"
            "|------|------------|----------|--------------|-------------|\n"
        )
        performance_stage.append_content(
            "| Time to first token, overall "
            f"| {start_time.strftime('%H:%M:%S')} "
            f"| {first_token_time.strftime('%H:%M:%S')} "
            f"| {(first_token_time - start_time).total_seconds():.2f} "
            "| Time from receipt of request to first token of agent's last response |\n"
        )
        performance_stage.append_content(
            "| Time to first token, final response only"
            f"| {start_time_of_last_request.strftime('%H:%M:%S')} "
            f"| {first_token_time.strftime('%H:%M:%S')} "
            f"| {(first_token_time - start_time_of_last_request).total_seconds():.2f} "
            "| Time since the agent's last request and the first token received in response to it |\n"
        )
        duration_manager = get_llm_call_duration_manager()
        if duration_manager is not None:
            for item in duration_manager.get_durations():
                performance_stage.append_content(
                    f"| LLM call duration: {item.deployment} "
                    "| | "
                    f"| {item.duration_s:.3f} "
                    f"| Cumulative duration for {item.deployment} |\n"
                )
            performance_stage.append_content(
                "| Total LLM calls duration "
                "| | "
                f"| {duration_manager.total_duration_s:.3f} "
                "| Sum of all LLM call durations |\n"
            )
