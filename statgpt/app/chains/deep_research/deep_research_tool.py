import time
from typing import Annotated, Any, Self

from mcp.types import ToolAnnotations
from pydantic import Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import GuardrailInput, StatGptTool, ToolArgs
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas import (
    DEEP_RESEARCH_ERROR_MESSAGE,
    DeepResearchSession,
    ToolArtifact,
    ToolMessageState,
)
from statgpt.app.utils import OpenAiToDialStreamer, openai
from statgpt.common.config import multiline_logger as logger
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas import DeepResearchTool as DeepResearchToolConfig
from statgpt.common.schemas import ToolTypes
from statgpt.common.schemas.llm_call_duration import LLMCallDurationItem
from statgpt.common.utils.llm_call_duration_context import get_llm_call_duration_manager

RESUME_DEEP_RESEARCH_TOOL_NAME = "resume_deep_research"
RESUME_DEEP_RESEARCH_TOOL_DESCRIPTION = (
    "Resume the active Deep Research session by forwarding the user's latest message"
    " (their answer to a clarifying question or their approval of the plan)."
)


class DeepResearchArgs(ToolArgs):
    query: Annotated[str, GuardrailInput] = Field(
        description="The natural language question to research in depth."
    )


class ResumeDeepResearchArgs(ToolArgs):
    message: Annotated[str, GuardrailInput] = Field(
        description=(
            "The user's latest message, forwarded verbatim to the active Deep Research session."
        )
    )


class DeepResearchTool(StatGptTool[DeepResearchToolConfig], tool_type=ToolTypes.DEEP_RESEARCH):
    """Starts a Deep Research session from a research question."""

    @classmethod
    def get_mcp_annotations(cls) -> ToolAnnotations:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)

    @classmethod
    def get_args_schema(cls, tool_config: DeepResearchToolConfig) -> type[DeepResearchArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return DeepResearchArgs

    @staticmethod
    def _load_session(state: dict) -> DeepResearchSession:
        """Load the active session from state, or start a fresh one.

        A finished session is dropped from state (see `_drop_session`), so any session found
        here is an in-progress run to resume; its absence starts a new investigation (the
        finished report stays in the chat history)."""
        session = DeepResearchSession.from_state(state)
        return session if session is not None else DeepResearchSession()

    @staticmethod
    def _build_request_messages(
        system_prompt: str | None, session: DeepResearchSession, user_message: str
    ) -> list[dict[str, Any]]:
        """The DIAL messages sent to Deep Research: the replayed sub-conversation plus the new
        user input. Deep Research resumes from the `custom_content.state` it stored on the last
        assistant message."""
        messages: list[dict[str, Any]] = []
        # Deep Research reads its own system prompt from its app properties and skips the
        # system role on input; we still forward the configured prompt for parity.
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.extend(session.messages)
        messages.append({'role': 'user', 'content': user_message})
        return messages

    @staticmethod
    def _research_started(dr_state: dict[str, Any] | None) -> bool:
        """Whether Deep Research has started the investigation, per its persisted state.

        CONTRACT: Deep Research sets ``preparation.research_started`` to ``True`` only once the
        plan is approved AND it has fully streamed the final report within this same response.
        We rely on that to treat the session as finished; if the deployment ever emits
        ``research_started`` before the report is delivered, the session would be closed
        prematurely and the report never resumed."""
        preparation = (dr_state or {}).get('preparation') or {}
        return bool(preparation.get('research_started'))

    @staticmethod
    def _append_turn(
        session: DeepResearchSession,
        user_message: str,
        assistant_content: str,
        dr_state: dict[str, Any],
    ) -> None:
        """Record a user/assistant exchange (assistant carrying Deep Research's state) so the
        sub-conversation can be replayed on the next call/turn."""
        session.messages.append({'role': 'user', 'content': user_message})
        session.messages.append(
            {
                'role': 'assistant',
                'content': assistant_content,
                'custom_content': {'state': dr_state},
            }
        )

    @staticmethod
    def _save_session(state: dict, session: DeepResearchSession) -> None:
        state[StateVarsConfig.DEEP_RESEARCH_SESSION] = session.model_dump(mode='json')

    @staticmethod
    def _drop_session(state: dict) -> None:
        # Drop a finished session rather than persisting the (potentially large) report and
        # accumulated state on every later turn — the report already lives in the chat history.
        DeepResearchSession.drop_from_state(state)

    @staticmethod
    async def _run_turn(
        tool_config: DeepResearchToolConfig,
        inputs: dict,
        user_message: str,
    ) -> str:
        """Run one Deep Research turn: call the deployment, stream its output verbatim to the
        user, persist the session, and drop it once research has started (final report delivered).

        Shared by the start (`deep_research`) and resume (`resume_deep_research`) tools; the only
        difference between them is which message they forward and when the agent invokes them."""
        auth_context = ChainParameters.get_auth_context(inputs)
        choice = ChainParameters.get_choice(inputs)
        state = ChainParameters.get_state(inputs)

        details = tool_config.details
        deployment_id = details.get_deployment_id()

        session = DeepResearchTool._load_session(state)
        messages = DeepResearchTool._build_request_messages(
            details.system_prompt, session, user_message
        )

        show_debug_stages = (
            state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False) or details.always_show_stages
        )

        create_kwargs: dict[str, Any] = dict(
            model=deployment_id, stream=True, messages=messages
        )
        client = openai.get_async_client(api_key=auth_context.api_key)
        time_start = time.monotonic()
        failed = False
        async with client:
            # Stream verbatim: whatever Deep Research returns (clarifying questions, plan for
            # approval, or the final report) is surfaced to the user as-is, token by token.
            dial_streamer = OpenAiToDialStreamer(
                choice,
                choice,
                deployment=deployment_id,
                stream_content=True,
                show_debug_stages=show_debug_stages,
                stages_config=details.stages_config,
            )
            with dial_streamer:
                try:
                    stream = await client.chat.completions.create(**create_kwargs)
                    async for chunk in stream:
                        dial_streamer.send_chunk(chunk)
                except Exception as e:
                    logger.exception(e)
                    failed = True

            content = dial_streamer.content
            dr_state = dial_streamer.state

        duration_s = time.monotonic() - time_start
        if (duration_manager := get_llm_call_duration_manager()) is not None:
            duration_manager.add_duration(
                LLMCallDurationItem(deployment=deployment_id, duration_s=duration_s)
            )

        if failed:
            # Deep Research owns this turn and is force-selected, so surface a friendly message
            # and leave the session untouched so the user can retry.
            choice.append_content(DEEP_RESEARCH_ERROR_MESSAGE)
            return DEEP_RESEARCH_ERROR_MESSAGE

        DeepResearchTool._append_turn(session, user_message, content, dr_state or {})
        if DeepResearchTool._research_started(dr_state):
            DeepResearchTool._drop_session(state)
        else:
            DeepResearchTool._save_session(state, session)
        return content

    async def _arun(self, inputs: dict, query: str, **kwargs) -> tuple[str, ToolArtifact]:
        content = await self._run_turn(self._tool_config, inputs, query)
        return content, ToolArtifact(state=ToolMessageState(type=self.tool_type))


class ResumeDeepResearchTool(
    StatGptTool[DeepResearchToolConfig], tool_type=ToolTypes.RESUME_DEEP_RESEARCH
):
    """Resumes an in-progress Deep Research session with the user's next message.

    Not admin-configurable: built on demand from the `deep_research` tool config and invoked
    only while a session is in progress. Kept separate from the start tool so each has a clean,
    single-purpose contract."""

    @classmethod
    def get_mcp_annotations(cls) -> ToolAnnotations:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)

    @classmethod
    def get_args_schema(cls, tool_config: DeepResearchToolConfig) -> type[ResumeDeepResearchArgs]:
        return ResumeDeepResearchArgs

    @classmethod
    def build(cls, tool_config: DeepResearchToolConfig, channel_config: ChannelConfig) -> Self:
        """Construct the resume tool from the (start) Deep Research tool config. The name and
        description are fixed rather than admin-configured, since this tool is an internal
        continuation mechanism."""
        return cls(
            tool_config=tool_config,
            channel_config=channel_config,
            name=RESUME_DEEP_RESEARCH_TOOL_NAME,
            description=RESUME_DEEP_RESEARCH_TOOL_DESCRIPTION,
            args_schema=ResumeDeepResearchArgs,
        )

    async def _arun(self, inputs: dict, message: str, **kwargs) -> tuple[str, ToolArtifact]:
        content = await DeepResearchTool._run_turn(self._tool_config, inputs, message)
        return content, ToolArtifact(state=ToolMessageState(type=self.tool_type))
