import asyncio
import copy
from datetime import datetime

from aidial_sdk.chat_completion import Choice, Role
from langchain_core.messages import AIMessage, AIMessageChunk, ToolCall, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnablePassthrough

from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.schemas import ChannelConfig, FakeCall
from common.schemas.dial import FunctionCall
from common.schemas.dial import Message as DialMessage
from common.schemas.dial import ToolCall as DialToolCall
from common.utils.markdown import format_as_markdown_list
from common.utils.models import get_chat_model
from statgpt.chains.data_query.data_query_artifacts_displayer import DataQueryArtifactDisplayer
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.tools import StatGptTool
from statgpt.config import ChainParametersConfig
from statgpt.default_prompts.prompts import SupremeAgentPrompts
from statgpt.schemas import FailedToolArtifact, FailedToolMessageState
from statgpt.schemas.tool_artifact import DataQueryArtifact
from statgpt.utils.dial_tools import optional_stage, timed_stage
from statgpt.utils.message_history import History


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
        return [
            StatGptTool.from_config(tool_cfg, channel_config) for tool_cfg in channel_config.tools
        ]

    async def call_tool(
        self, tool_call: ToolCall, inputs: dict, show_stage: bool = True
    ) -> ToolMessage:
        tool = self._tools[tool_call['name']]

        choice = ChainParameters.get_choice(inputs)

        stage_generator = timed_stage(choice=choice, name=tool.stage_name)
        with optional_stage(stage_generator, enabled=show_stage) as stage:
            # logger.info(f"{tool_call=}")
            # stage.append_content(f"Calling tool with args: {tool_call['args']}  \n\n")

            # NOTE: deepcopy raises errors - something with pydantic.
            # shallow copy seems to be enough here.
            inputs = copy.copy(inputs)
            inputs[ChainParametersConfig.TARGET] = stage

            tool_call['args']['inputs'] = inputs  # inject tool argument
            # logger.info(f"{tool_call=}")
            try:
                tool_msg: ToolMessage = await tool.ainvoke(tool_call)
            except Exception as e:
                logger.exception(f"Error calling tool {tool.name}:\n{e}")
                return ToolMessage(
                    content=f"The tool {tool.name} failed to execute.",
                    tool_call_id=tool_call['id'],
                    artifact=FailedToolArtifact(
                        state=FailedToolMessageState(type=tool.tool_type, error=repr(e))
                    ),
                    status='error',
                )

        logger.info(f"Tool message: {tool_msg!r}")
        return tool_msg


class SupremeAgent:
    def __init__(self, choice: Choice, chain: Runnable):
        self._choice = choice
        self._chain = chain

    @classmethod
    def create(
        cls,
        choice: Choice,
        auth_context: AuthContext,
        channel_config: ChannelConfig,
        tools: list[StatGptTool],
    ) -> 'SupremeAgent':
        chain = cls._create_chain(auth_context, channel_config, tools)
        return cls(choice, chain)

    async def run(self, history: History) -> AIMessageChunk:
        chunk: AIMessageChunk
        resp: AIMessageChunk | None = None

        inputs = {
            'chat_history': history.get_langchain_messages(),
            'datetime_now': datetime.now().isoformat(),
        }
        async for chunk in self._chain.astream(inputs):
            if resp is None:
                resp = chunk
            else:
                resp = resp + chunk  # type: ignore

            if chunk.content:
                self._choice.append_content(chunk.content)

        return resp

    @classmethod
    def _create_chain(
        cls, auth_context: AuthContext, channel_config: ChannelConfig, tools: list[StatGptTool]
    ) -> Runnable:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SupremeAgentPrompts.SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
            ]
        ).partial(
            chat_bot_name=channel_config.supreme_agent.name,
            chat_bot_domain=channel_config.supreme_agent.domain,
            chat_bot_terminology_domain=channel_config.supreme_agent.terminology_domain,
            chat_bot_language_instructions=format_as_markdown_list(
                channel_config.supreme_agent.language_instructions, list_type="ordered"
            ),
        )
        model = get_chat_model(
            api_key=auth_context.api_key,
            model=channel_config.supreme_agent.llm_model.deployment_name,
            temperature=channel_config.supreme_agent.llm_model.temperature,
            seed=channel_config.supreme_agent.llm_model.seed,
        )
        model_with_tools = model.bind_tools(tools, strict=True)

        return prompt_template | model_with_tools


class SupremeAgentExecutor:
    def __init__(self, channel_config: ChannelConfig):
        self._channel_config = channel_config

    async def stream_response(self, inputs: dict) -> str:
        auth_context = ChainParameters.get_auth_context(inputs)
        choice = ChainParameters.get_choice(inputs)
        history = ChainParameters.get_history(inputs)

        tool_executor = ToolCaller.from_config(self._channel_config)
        supreme_agent = SupremeAgent.create(
            choice, auth_context, self._channel_config, tool_executor.tools
        )

        fake_history = await self._fake_tool_calls(tool_executor, inputs)
        history.prepend(fake_history)

        data_query_artifacts: list[DataQueryArtifact] = []

        for _ in range(self._channel_config.supreme_agent.max_agent_iterations):
            resp = await supreme_agent.run(history)
            logger.info(f"Response: {resp!r}")

            if tool_calls := resp.tool_calls:
                history.add_chunk_as_tool_message(resp)

                res: list[ToolMessage] = await asyncio.gather(
                    *(tool_executor.call_tool(tool_call, inputs) for tool_call in tool_calls)
                )

                for tool_msg in res:
                    history.add_tool_message(tool_msg)

                    if (artifact := tool_msg.artifact) and isinstance(artifact, DataQueryArtifact):
                        data_query_artifacts.append(artifact)

            else:
                if data_query_artifacts:
                    data_displayer = DataQueryArtifactDisplayer(
                        choice, self._channel_config.data_query.details.attachments, auth_context
                    )
                    await data_displayer.display(data_query_artifacts)

                return resp.content

        warning_msg = '\n\n[WARNING] Maximum number of tool calls reached. Please enter "continue" to proceed.'
        choice.append_content(warning_msg)
        return warning_msg

    async def create_chain(self) -> Runnable:
        return RunnablePassthrough.assign(general_response=self.stream_response)

    async def _fake_tool_calls(self, tool_executor: ToolCaller, inputs: dict) -> History:
        fake_history = History.create_empty()

        for tool_cfg in self._channel_config.tools:
            if (fake_call := tool_cfg.details.fake_call) is not None:
                fake_history.add_dial_message(
                    self._get_tool_call_from_cfg(tool_cfg.name, fake_call)
                )
                last_msg: AIMessage = fake_history.get_langchain_messages()[-1]
                tool_msg = await tool_executor.call_tool(
                    last_msg.tool_calls[0], inputs, show_stage=False
                )
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
