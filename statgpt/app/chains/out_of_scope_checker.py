import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.config import ChainParametersConfig, StateVarsConfig
from statgpt.app.default_prompts import guardrails_default_prompts
from statgpt.app.utils.dial_stages import ChoiceI, optional_timed_stage
from statgpt.app.utils.message_history import History
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import ChannelConfig
from statgpt.common.utils.markdown import format_as_markdown_list
from statgpt.common.utils.models import get_chat_model


class OutOfScopeCheckerResponse(BaseModel):
    reasoning: str = Field(
        description="Short and concise reasoning for the out of scope decision."
        "Not more than 20 words."
        "If your decision is 'out-of-scope', you MUST reference specific criteria from the instruction. "
        "Don't provide any statements like 'This request is out of scope', just provide the reasoning."
    )
    out_of_scope: bool = Field(description="Whether the user's message is out of scope")


class OutOfScopeChecker:
    def __init__(self, channel_config: ChannelConfig):
        self._channel_config = channel_config

    def _get_tool_description(self) -> str:
        return json.dumps(
            {tool.name: tool.out_of_scope_description for tool in self._channel_config.tools},
            ensure_ascii=False,
        )

    @staticmethod
    def _count_out_of_scope_msgs(history: History) -> int:
        count: int = 0
        for msg in history.get_ai_messages():
            if msg.custom_content and msg.custom_content.state:
                if msg.custom_content.state.get(StateVarsConfig.OUT_OF_SCOPE, False) is True:
                    count += 1
        return count

    @staticmethod
    def _start_new_conversation(
        inputs: dict,
        choice: ChoiceI,
        out_of_scope_msgs_count: int,
        start_new_conversation_messages_threshold: int,
        start_new_conversation_message: str,
    ) -> dict:
        choice.append_content(start_new_conversation_message)
        inputs[ChainParametersConfig.OUT_OF_SCOPE] = True
        inputs[ChainParametersConfig.OUT_OF_SCOPE_REASONING] = (
            f"User has {out_of_scope_msgs_count} out-of-scope messages in the conversation history, "
            f"exceeding the threshold of {start_new_conversation_messages_threshold}."
        )
        return inputs

    def _build_checker_params(self, messages: Sequence[BaseMessage]) -> dict[str, Any]:
        """Build the prompt parameters shared by the checker and response prompts.

        Chat-agnostic: takes the messages to classify directly instead of reading
        them from the chat history, so it can be reused outside the chat chain.
        """
        if self._channel_config.out_of_scope is None:
            raise ValueError("out_of_scope must be configured for the channel")

        language_instructions = format_as_markdown_list(
            self._channel_config.supreme_agent.language_instructions, list_type="ordered"
        )

        params: dict[str, Any] = dict(
            chat_history=messages,
            domain_description=self._channel_config.out_of_scope.domain,
            tools_description=self._get_tool_description(),
            chat_bot_language_instructions=language_instructions,
        )

        blacklist = []
        if self._channel_config.out_of_scope.use_general_topics_blacklist:
            blacklist += guardrails_default_prompts.general_topics_blacklist
        if self._channel_config.out_of_scope.custom_blacklist:
            blacklist += self._channel_config.out_of_scope.custom_blacklist
        params["blacklist"] = (
            (
                "# The following topics and questions are strictly OUT OF SCOPE:  \n"
                + format_as_markdown_list(blacklist, list_type="ordered")
            )
            if blacklist
            else ""
        )
        return params

    async def classify(
        self, messages: Sequence[BaseMessage], auth_context: AuthContext
    ) -> OutOfScopeCheckerResponse:
        """Classify whether the given messages are out of scope for the channel.

        This is the chat-agnostic core of the guardrail: just the LLM relevancy
        decision, with no chat-history bookkeeping, streaming, or state. Reused by
        both the chat chain (``_stream_response``) and the MCP input guardrail.
        Requires the channel to have an ``out_of_scope`` configuration.
        """
        if self._channel_config.out_of_scope is None:
            raise ValueError("out_of_scope must be configured for the channel")

        params = self._build_checker_params(messages)
        checker_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    guardrails_default_prompts.checker_prompt
                ),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        ).partial(**params)

        model = get_chat_model(
            api_key=auth_context.api_key,
            model_config=self._channel_config.out_of_scope.llm_model_config,
        )

        checker_chain = checker_prompt | model.with_structured_output(
            OutOfScopeCheckerResponse, method="json_schema"
        )

        response: OutOfScopeCheckerResponse = await checker_chain.ainvoke({})  # type: ignore[assignment]
        return response

    def _build_response_chain(
        self, messages: Sequence[BaseMessage], reasoning: str, auth_context: AuthContext
    ) -> Runnable:
        """Build the chain that generates the user-facing out-of-scope message.

        Shared by the chat chain (which streams it to the DIAL choice) and the MCP
        input guardrail (which invokes it for the full message). The prompt is fully
        bound via ``.partial``, so the chain can be invoked with ``{}``.
        """
        if self._channel_config.out_of_scope is None:
            raise ValueError("out_of_scope must be configured for the channel")

        params = self._build_checker_params(messages)
        params["out_of_scope_reasoning"] = reasoning
        response_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    guardrails_default_prompts.response_prompt
                ),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        ).partial(**params)

        model = get_chat_model(
            api_key=auth_context.api_key,
            model_config=self._channel_config.out_of_scope.llm_model_config,
        )
        return response_prompt | model

    async def generate_response(
        self, messages: Sequence[BaseMessage], reasoning: str, auth_context: AuthContext
    ) -> str:
        """Generate the user-facing out-of-scope message (non-streaming).

        Reused by the MCP input guardrail, which has no DIAL choice to stream into.
        """
        chain = self._build_response_chain(messages, reasoning, auth_context)
        result = await chain.ainvoke({})
        return result.content if isinstance(result.content, str) else str(result.content)

    async def _stream_response(self, inputs: dict) -> dict:
        state = ChainParameters.get_state(inputs)
        oos_only = state.get(StateVarsConfig.CMD_OUT_OF_SCOPE_ONLY, False)
        tool_calls = state.get(StateVarsConfig.DIRECT_TOOL_CALLS, [])

        skip = ChainParameters.skip_out_of_scope_check(inputs)
        if skip or self._channel_config.out_of_scope is None:
            inputs[ChainParametersConfig.OUT_OF_SCOPE] = None
            inputs[ChainParametersConfig.OUT_OF_SCOPE_REASONING] = 'guardrails disabled in config'
            return inputs

        if tool_calls:
            inputs[ChainParametersConfig.OUT_OF_SCOPE] = None
            inputs[ChainParametersConfig.OUT_OF_SCOPE_REASONING] = (
                "direct tool calls found - skipping guardrails"
            )
            return inputs

        auth_context = ChainParameters.get_auth_context(inputs)
        choice = ChainParameters.get_choice(inputs)
        history = ChainParameters.get_history(inputs)

        start_new_conversation_messages_threshold = (
            self._channel_config.out_of_scope.start_new_conversation_messages_threshold
        )
        if start_new_conversation_messages_threshold != -1:
            out_of_scope_msgs_count = self._count_out_of_scope_msgs(history)
            if out_of_scope_msgs_count > start_new_conversation_messages_threshold:
                return self._start_new_conversation(
                    inputs,
                    choice,
                    out_of_scope_msgs_count,
                    start_new_conversation_messages_threshold,
                    self._channel_config.out_of_scope.start_new_conversation_message,
                )

        messages = history.get_langchain_messages(include_tool_messages=False)

        show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False)
        with optional_timed_stage(
            choice, "[DEBUG] Guardrails: Relevancy", enabled=show_debug_stages
        ) as stage:
            response = await self.classify(messages, auth_context)
            if stage:
                if response.out_of_scope:
                    stage.append_content(
                        f"Request is out of scope, reasoning: {response.reasoning}"
                    )
                else:
                    stage.append_content(f"Request is in scope, reasoning: {response.reasoning}")

        inputs[ChainParametersConfig.OUT_OF_SCOPE] = response.out_of_scope
        inputs[ChainParametersConfig.OUT_OF_SCOPE_REASONING] = response.reasoning

        if oos_only or not response.out_of_scope:
            return inputs

        # provide message to user

        if start_new_conversation_messages_threshold != -1:
            out_of_scope_msgs_count = self._count_out_of_scope_msgs(history) + 1
            if out_of_scope_msgs_count > start_new_conversation_messages_threshold:
                return self._start_new_conversation(
                    inputs,
                    choice,
                    out_of_scope_msgs_count,
                    start_new_conversation_messages_threshold,
                    self._channel_config.out_of_scope.start_new_conversation_message,
                )

        # tell user that the request is out of scope

        response_chain = self._build_response_chain(messages, response.reasoning, auth_context)

        async for chunk in response_chain.astream(inputs):
            if isinstance(chunk.content, str):
                choice.append_content(chunk.content)

        return inputs

    async def create_chain(self) -> Runnable:
        return RunnableLambda(self._stream_response)
