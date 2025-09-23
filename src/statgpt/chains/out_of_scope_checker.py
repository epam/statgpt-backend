import json

from aidial_sdk.chat_completion import Choice
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from common.schemas import ChannelConfig
from common.utils.markdown import format_as_markdown_list
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters
from statgpt.config import ChainParametersConfig, StateVarsConfig
from statgpt.default_prompts import NotSupportedScenariosPrompts
from statgpt.utils.dial_stages import optional_timed_stage
from statgpt.utils.message_history import History


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

    @staticmethod
    def _get_checker_prompt() -> str:
        return NotSupportedScenariosPrompts.CHECKER_PROMPT

    @staticmethod
    def _get_response_prompt() -> str:
        return NotSupportedScenariosPrompts.RESPONSE_PROMPT

    def _get_tool_description(self) -> str:
        return json.dumps(
            {tool.name: tool.out_of_scope_description for tool in self._channel_config.tools}
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
        choice: Choice,
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

    async def _stream_response(self, inputs: dict) -> dict:
        state = ChainParameters.get_state(inputs)

        skip = ChainParameters.skip_out_of_scope_check(inputs)
        if skip or self._channel_config.out_of_scope is None:
            inputs[ChainParametersConfig.OUT_OF_SCOPE] = None
            inputs[ChainParametersConfig.OUT_OF_SCOPE_REASONING] = None
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

        language_instructions = format_as_markdown_list(
            self._channel_config.supreme_agent.language_instructions, list_type="ordered"
        )

        params = dict(
            chat_history=history.get_langchain_messages(include_tool_messages=False),
            domain_description=self._channel_config.out_of_scope.domain,
            tools_description=self._get_tool_description(),
            chat_bot_language_instructions=language_instructions,
        )

        topics_blacklist = []
        if self._channel_config.out_of_scope.use_general_topics_blacklist:
            topics_blacklist += NotSupportedScenariosPrompts.GENERAL_TOPICS_BLACKLIST
        if self._channel_config.out_of_scope.custom_instructions:
            topics_blacklist += self._channel_config.out_of_scope.custom_instructions
        if topics_blacklist:
            params["custom_instructions"] = (
                "# The following topics and questions are strictly OUT OF SCOPE:  \n"
                + format_as_markdown_list(topics_blacklist, list_type="ordered")
            )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self._get_checker_prompt()),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        ).partial(**params)

        model = get_chat_model(
            api_key=auth_context.api_key,
            model_config=self._channel_config.supreme_agent.llm_model_config,
        )

        chain = prompt_template | model.with_structured_output(
            OutOfScopeCheckerResponse, method="json_schema"
        )

        show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False)
        with optional_timed_stage(
            choice, "[DEBUG] Guardrails: Relevancy", enabled=show_debug_stages
        ) as stage:
            response: OutOfScopeCheckerResponse = await chain.ainvoke({})
            if response.out_of_scope:
                stage.append_content(f"Request is out of scope, reasoning: {response.reasoning}")
            else:
                stage.append_content(f"Request is in scope, reasoning: {response.reasoning}")

        inputs[ChainParametersConfig.OUT_OF_SCOPE] = response.out_of_scope
        inputs[ChainParametersConfig.OUT_OF_SCOPE_REASONING] = response.reasoning

        if not response.out_of_scope or state.get(StateVarsConfig.CMD_OUT_OF_SCOPE_ONLY, False):
            return inputs

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

        params["out_of_scope_reasoning"] = response.reasoning
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self._get_response_prompt()),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        ).partial(**params)

        chain = prompt_template | model

        async for chunk in chain.astream(inputs):
            choice.append_content(chunk.content)

        return inputs

    async def create_chain(self) -> Runnable:
        return RunnableLambda(self._stream_response)
