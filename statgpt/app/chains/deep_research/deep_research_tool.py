import time
from typing import Annotated, Any, NamedTuple, Self

from mcp.types import ToolAnnotations
from pydantic import Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import GuardrailInput, StatGptTool, ToolArgs
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas import (
    DeepResearchArtifact,
    DeepResearchSession,
    DeepResearchToolMessageState,
    DeepResearchTurn,
)
from statgpt.app.utils import OpenAiToDialStreamer, openai
from statgpt.app.utils.dial_stages import ChoiceI
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas import DeepResearchTool as DeepResearchToolConfig
from statgpt.common.schemas import ToolTypes
from statgpt.common.schemas.llm_call_duration import LLMCallDurationItem
from statgpt.common.utils.llm_call_duration_context import get_llm_call_duration_manager


class DeepResearchArgs(ToolArgs):
    query: Annotated[str, GuardrailInput] = Field(
        description="The natural language question to research in depth."
    )


class ResumeDeepResearchArgs(ToolArgs):
    # No `GuardrailInput`: this message is composed by the Supreme Agent while mediating the session,
    # not free-text typed by the user. (Deep Research is never exposed via the MCP server, the only
    # place the input guardrail runs, so the marker was inert here regardless.)
    message: str = Field(
        description=(
            "The message to send to the active Deep Research session: the user's answers to its"
            " clarifying questions (answered from context where possible) or their approval of the"
            " plan."
        )
    )


class DeepResearchTurnResult(NamedTuple):
    """Outcome of a single Deep Research turn.

    ``report_delivered`` is True only when the deployment delivered its final report (research
    complete). The report is already streamed to the user verbatim, so the Supreme Agent must end
    the turn without repeating it. Otherwise ``content`` is a clarification / plan-for-approval that
    the Supreme Agent mediates."""

    content: str
    report_delivered: bool


def _build_deep_research_artifact(
    tool_type: ToolTypes, result: DeepResearchTurnResult
) -> DeepResearchArtifact:
    return DeepResearchArtifact(
        state=DeepResearchToolMessageState(type=tool_type, report_delivered=result.report_delivered)
    )


class DeepResearchRunner:
    """Runs a single Deep Research turn and owns the persisted session.

    Holds the business logic shared by the start (`deep_research`) and resume
    (`resume_deep_research`) tools: it calls the deployment, hands a clarification / plan back to the
    Supreme Agent (buffered, not shown to the user) or delivers the final report verbatim, and
    persists or drops the `DeepResearchSession`. Both tools construct a runner and call `run`; the
    only difference between them is which message they forward and when the agent invokes them."""

    def __init__(self, tool_config: DeepResearchToolConfig):
        self._tool_config = tool_config

    @staticmethod
    def _load_session(state: dict[str, Any]) -> DeepResearchSession:
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
        assistant message, so the DIAL shape is rebuilt here from the persisted turns."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        for turn in session.turns:
            messages.append({'role': 'user', 'content': turn.user_message})
            messages.append(
                {
                    'role': 'assistant',
                    'content': turn.assistant_content,
                    'custom_content': {'state': turn.deep_research_state},
                }
            )
        messages.append({'role': 'user', 'content': user_message})
        return messages

    @staticmethod
    def _research_started(deep_research_state: dict[str, Any] | None) -> bool:
        """Whether Deep Research has started the investigation, per its persisted state.

        CONTRACT: Deep Research sets ``preparation.research_started`` to ``True`` only once the
        plan is approved AND it has fully streamed the final report within this same response.
        We rely on that to treat the session as finished; if the deployment ever emits
        ``research_started`` before the report is delivered, the session would be closed
        prematurely and the report never resumed."""
        preparation = (deep_research_state or {}).get('preparation') or {}
        return bool(preparation.get('research_started'))

    @staticmethod
    def _append_turn(
        session: DeepResearchSession,
        user_message: str,
        assistant_content: str,
        deep_research_state: dict[str, Any],
    ) -> None:
        """Record a user/assistant exchange (with Deep Research's own state) so the
        sub-conversation can be replayed on the next call/turn."""
        session.turns.append(
            DeepResearchTurn(
                user_message=user_message,
                assistant_content=assistant_content,
                deep_research_state=deep_research_state,
            )
        )

    @staticmethod
    def _save_session(state: dict[str, Any], session: DeepResearchSession) -> None:
        state[StateVarsConfig.DEEP_RESEARCH_SESSION] = session.model_dump(mode='json')

    @staticmethod
    def _drop_session(state: dict[str, Any]) -> None:
        # Drop a finished session rather than persisting the (potentially large) report and
        # accumulated state on every later turn — the report already lives in the chat history.
        DeepResearchSession.drop_from_state(state)

    @staticmethod
    def _deliver_report(choice: ChoiceI, content: str, attachments: list[dict[str, Any]]) -> None:
        """Stream Deep Research's final report to the user verbatim.

        Delivered straight from the deployment output (content plus any attachments, e.g. a Canvas
        document) so no tokens are re-spent and nothing is paraphrased. Content was buffered during
        the run (see `run`), so it is appended here in one shot."""
        if content:
            choice.append_content(content)
        for attachment in attachments:
            choice.add_attachment(
                type=attachment.get('type'),
                title=attachment.get('title'),
                data=attachment.get('data'),
                url=attachment.get('url'),
                reference_url=attachment.get('reference_url'),
                reference_type=attachment.get('reference_type'),
            )

    async def run(self, inputs: dict, user_message: str) -> DeepResearchTurnResult:
        """Run one Deep Research turn: call the deployment, then either hand a clarification / plan
        back to the Supreme Agent (buffered, not shown to the user) or deliver the final report to
        the user verbatim once research has started. Persist the session between clarifications and
        drop it once the report is delivered.

        Request/stream failures are not surfaced here: they propagate so `ToolCaller.call_tool`
        records the error in the tool state (an ERROR tool message) and the Supreme Agent surfaces
        the standard failure message once. The session is left untouched so the user can retry."""
        auth_context = ChainParameters.get_auth_context(inputs)
        choice = ChainParameters.get_choice(inputs)
        target = ChainParameters.get_target(inputs)
        state = ChainParameters.get_state(inputs)

        details = self._tool_config.details
        deployment_id = details.get_deployment_id()

        session = self._load_session(state)
        messages = self._build_request_messages(details.system_prompt, session, user_message)

        show_debug_stages = (
            state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False) or details.always_show_stages
        )

        create_kwargs: dict[str, Any] = dict(model=deployment_id, stream=True, messages=messages)
        client = openai.get_async_client(api_key=auth_context.api_key)
        time_start = time.monotonic()
        try:
            async with client:
                # Buffer content (`stream_content=False`) so it can be routed once the run ends:
                # a clarification / plan is shown in the tool-result stage (like other tools) and
                # handed back to the Supreme Agent to mediate, while the final report is delivered to
                # the user's main answer verbatim. The two are indistinguishable until the deployment
                # signals `research_started` (only after the report has streamed), so we cannot stream
                # straight to a single target. Progress stages still stream live throughout.
                dial_streamer = OpenAiToDialStreamer(
                    choice,
                    choice,
                    deployment=deployment_id,
                    stream_content=False,
                    show_debug_stages=show_debug_stages,
                    stages_config=details.stages_config,
                )
                with dial_streamer:
                    stream = await client.chat.completions.create(**create_kwargs)
                    async for chunk in stream:
                        dial_streamer.send_chunk(chunk)
                content = dial_streamer.content
                deep_research_state = dial_streamer.state

            if self._research_started(deep_research_state):
                # Final report delivered: deliver it to the user's main answer verbatim, then drop
                # the session instead of recording this turn.
                self._deliver_report(choice, content, dial_streamer.attachments)
                self._drop_session(state)
                return DeepResearchTurnResult(content=content, report_delivered=True)

            # Clarification / plan-for-approval: show it in the tool-result stage (like other tools)
            # and hand the content back to the Supreme Agent, keeping the session so the next call can
            # resume from Deep Research's own state.
            target.append_content(content)
            self._append_turn(session, user_message, content, deep_research_state or {})
            self._save_session(state, session)
            return DeepResearchTurnResult(content=content, report_delivered=False)
        finally:
            duration_s = time.monotonic() - time_start
            if (duration_manager := get_llm_call_duration_manager()) is not None:
                duration_manager.add_duration(
                    LLMCallDurationItem(deployment=deployment_id, duration_s=duration_s)
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

    async def _arun(self, inputs: dict, query: str, **kwargs) -> tuple[str, DeepResearchArtifact]:
        result = await DeepResearchRunner(self._tool_config).run(inputs, query)
        return result.content, _build_deep_research_artifact(self.tool_type, result)


class ResumeDeepResearchTool(
    StatGptTool[DeepResearchToolConfig], tool_type=ToolTypes.RESUME_DEEP_RESEARCH
):
    """Resumes an in-progress Deep Research session with the user's next message.

    Built on demand from the `deep_research` tool config (its name and description come from
    `details.resume_tool_name` / `details.resume_tool_description`) and invoked only while a
    session is in progress. Kept separate from the start tool so each has a clean, single-purpose
    contract."""

    @classmethod
    def get_mcp_annotations(cls) -> ToolAnnotations:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)

    @classmethod
    def get_args_schema(cls, tool_config: DeepResearchToolConfig) -> type[ResumeDeepResearchArgs]:
        return ResumeDeepResearchArgs

    @classmethod
    def build(cls, tool_config: DeepResearchToolConfig, channel_config: ChannelConfig) -> Self:
        """Construct the resume tool from the (start) Deep Research tool config. The name and
        description are taken from the Deep Research tool config (`details.resume_tool_name` /
        `details.resume_tool_description`) so they stay admin-configurable."""
        details = tool_config.details
        return cls(
            tool_config=tool_config,
            channel_config=channel_config,
            name=details.resume_tool_name,
            description=details.resume_tool_description,
            args_schema=ResumeDeepResearchArgs,
        )

    async def _arun(self, inputs: dict, message: str, **kwargs) -> tuple[str, DeepResearchArtifact]:
        result = await DeepResearchRunner(self._tool_config).run(inputs, message)
        return result.content, _build_deep_research_artifact(self.tool_type, result)
