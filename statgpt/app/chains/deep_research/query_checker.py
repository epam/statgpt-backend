import asyncio
from collections.abc import Sequence

from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.config import StateVarsConfig
from statgpt.app.default_prompts import deep_research_default_prompts
from statgpt.app.utils.dial_stages import optional_timed_stage
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import logger
from statgpt.common.schemas import DeepResearchQueryCheckConfig
from statgpt.common.utils.markdown import format_as_markdown_list
from statgpt.common.utils.models import get_chat_model


class DeepResearchQueryCheckResponse(BaseModel):
    reasoning: str = Field(
        description="Short and concise reasoning for the decision. Not more than 20 words."
    )
    suits_deep_research: bool = Field(
        description="Whether the user's latest message suits Deep Research."
    )


class DeepResearchQueryChecker:
    """Decides whether the user's query suits Deep Research, before a new session is started.

    A cheap LLM call that keeps queries which are not research questions (e.g. "What can you do?")
    from launching a slow, expensive Deep Research run while Deep Research mode is on."""

    def __init__(self, config: DeepResearchQueryCheckConfig):
        self._config = config

    def _excluded_topics_section(self) -> str:
        topics = self._config.excluded_topics
        if topics is None:
            topics = deep_research_default_prompts.default_excluded_topics
        if not topics:
            return ""
        return (
            "# Topics that do not suit Deep Research\n\n"
            "The following messages **do not suit** Deep Research:\n"
            + format_as_markdown_list(topics, list_type="ordered")
        )

    def build_chain(self, messages: Sequence[BaseMessage], auth_context: AuthContext) -> Runnable:
        """Build the chain that classifies whether the latest message suits Deep Research.

        The prompt is fully bound via ``.partial``, so the chain can be invoked with ``{}``."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    deep_research_default_prompts.query_check_prompt
                ),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        ).partial(chat_history=messages, excluded_topics=self._excluded_topics_section())

        model = get_chat_model(
            api_key=auth_context.api_key, model_config=self._config.llm_model_config
        )
        return prompt | model.with_structured_output(
            DeepResearchQueryCheckResponse, method="json_schema"
        )

    async def suits_deep_research(self, inputs: dict) -> bool:
        """Whether the query of this turn suits Deep Research, judged from the conversation.

        Fails open: the check only spares an unnecessary run, so if any part of it fails or it
        does not finish within `timeout_seconds`, the user's explicit choice of Deep Research is
        honored."""
        try:
            return await self._suits_deep_research(inputs)
        except Exception:
            logger.exception("Deep Research query check failed, starting Deep Research")
            return True

    async def _suits_deep_research(self, inputs: dict) -> bool:
        auth_context = ChainParameters.get_auth_context(inputs)
        choice = ChainParameters.get_choice(inputs)
        state = ChainParameters.get_state(inputs)
        messages = ChainParameters.get_history(inputs).get_langchain_messages(
            include_tool_messages=False
        )

        show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False)
        with optional_timed_stage(
            choice, "[DEBUG] Deep Research: Query Check", enabled=show_debug_stages
        ) as stage:
            try:
                # The deadline bounds the whole check, including the model client's retries and
                # their backoff, so a slow or unavailable model cannot hold the turn up.
                async with asyncio.timeout(self._config.timeout_seconds):
                    response = await self.build_chain(messages, auth_context).ainvoke({})
            except TimeoutError:
                logger.warning(
                    "Deep Research query check did not finish within %ss, starting Deep Research",
                    self._config.timeout_seconds,
                )
                if stage:
                    stage.append_content("The check timed out, so Deep Research is started.")
                return True
            except Exception:
                if stage:
                    stage.append_content("The check failed, so Deep Research is started.")
                # Logged and failed open by `suits_deep_research`, which also covers the rest of
                # the check.
                raise

            if stage:
                verdict = "suits" if response.suits_deep_research else "does not suit"
                stage.append_content(
                    f"The query {verdict} Deep Research, reasoning: {response.reasoning}"
                )
        return response.suits_deep_research
