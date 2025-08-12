import json

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
from statgpt.utils.dial_tools import optional_stage, timed_stage


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

        language_instructions = format_as_markdown_list(
            self._channel_config.supreme_agent.language_instructions, list_type="ordered"
        )

        params = dict(
            chat_history=history.get_langchain_messages(include_tool_messages=False),
            domain_description=self._channel_config.out_of_scope.domain,
            tools_description=self._get_tool_description(),
            chat_bot_language_instructions=language_instructions,
        )

        if self._channel_config.out_of_scope.custom_instructions:
            params["custom_instructions"] = (
                "# The following topics and questions are strictly OUT OF SCOPE:  \n"
                + format_as_markdown_list(
                    self._channel_config.out_of_scope.custom_instructions, list_type="ordered"
                )
            )
        else:
            params["custom_instructions"] = ""

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self._get_checker_prompt()),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        ).partial(**params)

        model = get_chat_model(
            api_key=auth_context.api_key,
            model=self._channel_config.supreme_agent.llm_model.deployment_name,
            temperature=self._channel_config.supreme_agent.llm_model.temperature,
            seed=self._channel_config.supreme_agent.llm_model.seed,
        )

        chain = prompt_template | model.with_structured_output(
            OutOfScopeCheckerResponse, method="json_schema"
        )

        show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES)
        stage_generator = timed_stage(choice=choice, name="[DEBUG] Guardrails: Relevancy")
        with optional_stage(stage_generator, enabled=show_debug_stages) as stage:
            response: OutOfScopeCheckerResponse = await chain.ainvoke({})
            if response.out_of_scope:
                stage.append_content(f"Request is out of scope, reasoning: {response.reasoning}")
            else:
                stage.append_content(f"Request is in scope, reasoning: {response.reasoning}")

        inputs[ChainParametersConfig.OUT_OF_SCOPE] = response.out_of_scope
        inputs[ChainParametersConfig.OUT_OF_SCOPE_REASONING] = response.reasoning

        if not response.out_of_scope:
            return inputs

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
        return RunnableLambda.assign(self._stream_response)
