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
from statgpt.app.utils.dial_stages import ChoiceI, ContentStageI, optional_timed_stage
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
            {tool.name: tool.out_of_scope_description for tool in self._channel_config.agent_tools},
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

    def build_checker_chain(
        self, messages: Sequence[BaseMessage], auth_context: AuthContext
    ) -> Runnable:
        """Build the chain that classifies whether messages are out of scope.

        Chat-agnostic core of the guardrail: just the LLM relevancy decision, with
        no chat-history bookkeeping, streaming, or state. The prompt is fully bound
        via ``.partial``, so the chain can be invoked with ``{}``. Reused by both
        the chat chain (``stream_response``) and the MCP input guardrail. Requires
        the channel to have an ``out_of_scope`` configuration.
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

        return checker_prompt | model.with_structured_output(
            OutOfScopeCheckerResponse, method="json_schema"
        )

    def build_response_chain(
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

    def try_short_circuit(self, inputs: dict) -> dict | None:
        """Resolve the short-circuit paths that bypass the LLM relevancy check.

        Mutates ``inputs`` and may stream to the choice (the start-new-conversation
        message on the threshold path). Returns the completed ``inputs`` when the
        check is skipped (guardrails disabled, direct tool calls, or the
        out-of-scope messages threshold already exceeded), ``None`` when the LLM
        check should run.
        """
        state = ChainParameters.get_state(inputs)
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

        return None

    async def check(
        self, messages: Sequence[BaseMessage], auth_context: AuthContext
    ) -> OutOfScopeCheckerResponse:
        """Run the LLM relevancy check on the given messages."""
        checker_chain = self.build_checker_chain(messages, auth_context)
        return await checker_chain.ainvoke({})

    @staticmethod
    def append_verdict_to_stage(stage: ContentStageI, response: OutOfScopeCheckerResponse) -> None:
        if not stage:
            return
        scope = "out of scope" if response.out_of_scope else "in scope"
        stage.append_content(f"Request is {scope}, reasoning: {response.reasoning}")

    async def check_with_stage(self, inputs: dict) -> OutOfScopeCheckerResponse:
        """Run the LLM relevancy check inside the ``[DEBUG] Guardrails: Relevancy``
        stage and record the verdict on ``inputs``.

        Shared by the sequential (``stream_response``) and concurrent
        (``MainChainFactory._guarded_main_chain``) compositions so the visible
        verdict stage and the ``OUT_OF_SCOPE`` keys are produced identically.
        """
        state = ChainParameters.get_state(inputs)
        auth_context = ChainParameters.get_auth_context(inputs)
        choice = ChainParameters.get_choice(inputs)
        history = ChainParameters.get_history(inputs)

        messages = history.get_langchain_messages(include_tool_messages=False)

        show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False)
        with optional_timed_stage(
            choice, "[DEBUG] Guardrails: Relevancy", enabled=show_debug_stages
        ) as stage:
            response = await self.check(messages, auth_context)
            self.append_verdict_to_stage(stage, response)

        inputs[ChainParametersConfig.OUT_OF_SCOPE] = response.out_of_scope
        inputs[ChainParametersConfig.OUT_OF_SCOPE_REASONING] = response.reasoning
        return response

    async def respond_out_of_scope(self, inputs: dict, reasoning: str) -> dict:
        """Stream the user-facing out-of-scope message to the choice.

        Expects the ``OUT_OF_SCOPE`` keys to be already set on ``inputs`` (they
        are overridden only when the start-new-conversation threshold trips).
        """
        if self._channel_config.out_of_scope is None:
            raise ValueError("out_of_scope must be configured for the channel")

        auth_context = ChainParameters.get_auth_context(inputs)
        choice = ChainParameters.get_choice(inputs)
        history = ChainParameters.get_history(inputs)

        start_new_conversation_messages_threshold = (
            self._channel_config.out_of_scope.start_new_conversation_messages_threshold
        )
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

        messages = history.get_langchain_messages(include_tool_messages=False)
        response_chain = self.build_response_chain(messages, reasoning, auth_context)

        async for chunk in response_chain.astream(inputs):
            if isinstance(chunk.content, str):
                choice.append_content(chunk.content)

        return inputs

    async def stream_response(self, inputs: dict) -> dict:
        """Sequential composition of the guardrail: skip resolution, LLM check,
        out-of-scope response. Used when optimistic guardrails are disabled and
        for the ``CMD_OUT_OF_SCOPE_ONLY`` debug path."""
        if (resolved := self.try_short_circuit(inputs)) is not None:
            return resolved

        response = await self.check_with_stage(inputs)

        state = ChainParameters.get_state(inputs)
        oos_only = state.get(StateVarsConfig.CMD_OUT_OF_SCOPE_ONLY, False)
        if oos_only or not response.out_of_scope:
            return inputs

        return await self.respond_out_of_scope(inputs, response.reasoning)

    async def create_chain(self) -> Runnable:
        return RunnableLambda(self.stream_response)
