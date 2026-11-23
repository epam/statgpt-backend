import time
from typing import Annotated, Any, NamedTuple, Self

from openai import APIError
from pydantic import Field

from statgpt.app.chains.deep_research.errors import DeepResearchFailedError
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import GuardrailInput, StatGptTool, ToolArgs
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas import (
    DeepResearchArtifact,
    DeepResearchSession,
    DeepResearchToolMessageState,
    DeepResearchTurn,
)
from statgpt.app.utils import CustomContentRewriter, OpenAiToDialStreamer, openai
from statgpt.app.utils.dial_stages import ChoiceI
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas import DeepResearchTool as DeepResearchToolConfig
from statgpt.common.schemas import ResumeStagesConfig, ToolTypes
from statgpt.common.schemas.llm_call_duration import LLMCallDurationItem
from statgpt.common.utils.llm_call_duration_context import get_llm_call_duration_manager

_ORIGINAL_QUESTION_KEY = 'original_question'
_QUERY_KEY = 'query'


class DeepResearchArgs(ToolArgs):
    query: Annotated[str, GuardrailInput] = Field(
        description=(
            "The natural language question to research in depth. Make it self-contained: Deep"
            " Research does not see the chat, so resolve references to the earlier conversation"
            " into concrete details (e.g. the exact title of a publication mentioned earlier)."
        )
    )


class ResumeDeepResearchArgs(ToolArgs):
    # No `GuardrailInput`: the Supreme Agent composes this message, and Deep Research is never
    # exposed over MCP, where the guardrail runs.
    message: str = Field(
        description=(
            "The message to send to the active Deep Research session: the user's answers to its"
            " clarifying questions (answered from context where possible), their approval of the"
            " plan, or their requested changes to the query or plan. Make it self-contained: Deep"
            " Research does not see the chat, so resolve references to the earlier conversation"
            " into concrete details (e.g. the exact title of a publication mentioned earlier)."
        )
    )


class DeepResearchTurnResult(NamedTuple):
    """Outcome of a single Deep Research turn.

    With ``report_delivered``, ``content`` is the report plus the agent-only citations note and
    ``report`` the report as the user got it; otherwise ``content`` is a clarification or plan."""

    content: str
    report_delivered: bool
    report: str = ''


def _get_display_message(e: APIError) -> str | None:
    """The user-facing `display_message` of a DIAL error, if present.

    Mid-stream errors carry the error object as the body; HTTP error responses nest it under
    ``error``."""
    body = e.body if isinstance(e.body, dict) else {}
    error = body.get('error', body)
    display_message = error.get('display_message') if isinstance(error, dict) else None
    if isinstance(display_message, str) and (display_message := display_message.strip()):
        return display_message
    return None


def _build_deep_research_artifact(
    tool_type: ToolTypes, result: DeepResearchTurnResult
) -> DeepResearchArtifact:
    return DeepResearchArtifact(
        state=DeepResearchToolMessageState(
            type=tool_type, report_delivered=result.report_delivered
        ),
        report=result.report,
    )


class DeepResearchRunner:
    """Runs a single Deep Research turn and owns the persisted session.

    Shared by the start and resume tools, which differ only in the message they forward."""

    def __init__(self, tool_config: DeepResearchToolConfig):
        self._tool_config = tool_config

    @staticmethod
    def _load_session(state: dict[str, Any]) -> DeepResearchSession:
        """Load the in-progress session from state, or start a fresh one."""
        session = DeepResearchSession.from_state(state)
        return session if session is not None else DeepResearchSession()

    @staticmethod
    def _build_request_messages(
        system_prompt: str | None, session: DeepResearchSession, user_message: str
    ) -> list[dict[str, Any]]:
        """The DIAL messages sent to Deep Research: the replayed turns plus the new user input."""
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
        """Whether Deep Research's state sets `preparation.research_started`, which marks the
        response as the final report.

        See openspec/specs/deep-research: "Final-report contract"."""
        preparation = (deep_research_state or {}).get('preparation') or {}
        return bool(preparation.get('research_started'))

    @staticmethod
    def _append_turn(
        session: DeepResearchSession,
        user_message: str,
        assistant_content: str,
        deep_research_state: dict[str, Any],
    ) -> None:
        """Record a user/assistant exchange with Deep Research's own state."""
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
        # The report already lives in the chat history; don't resend it with every turn's state.
        DeepResearchSession.drop_from_state(state)

    @staticmethod
    def _deliver_report(choice: ChoiceI, content: str, attachments: list[dict[str, Any]]) -> None:
        """Append the buffered final report (content and attachments) to the user's answer."""
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
        """Run one Deep Research turn: hand a clarification / plan back to the Supreme Agent and
        save the session, or deliver the final report and drop the session.

        Deployment failures (`openai.APIError`) are raised as `DeepResearchFailedError`; other
        failures propagate to `ToolCaller.call_tool`."""
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

        # Keep `stream=True`: a blocking response loses the annotation indexes the streamer
        # renumbers (see openspec/specs/deep-research: "Requests are streamed").
        create_kwargs: dict[str, Any] = dict(model=deployment_id, stream=True, messages=messages)
        client = openai.get_async_client(
            api_key=auth_context.api_key, access_token=auth_context.dial_access_token
        )
        time_start = time.monotonic()
        dial_streamer: OpenAiToDialStreamer | None = None
        try:
            # Buffered: a clarification and the report can be told apart only once the run ends.
            dial_streamer = OpenAiToDialStreamer(
                choice,
                choice,
                deployment=deployment_id,
                stream_content=False,
                show_debug_stages=show_debug_stages,
                stages_config=details.stages_config,
                annotation_index_space=ChainParameters.get_annotation_index_space(inputs),
                rewriter=CustomContentRewriter(details.rewrite_rules),
            )
            with dial_streamer:
                stream = await client.chat.completions.create(**create_kwargs)
                async for chunk in stream:
                    dial_streamer.send_chunk(chunk)
            content = dial_streamer.content
            deep_research_state = dial_streamer.state

            if self._research_started(deep_research_state):
                self._deliver_report(choice, content, dial_streamer.attachments)
                self._drop_session(state)
                # Set only after delivery succeeds, so a failure leaves the toggle armed.
                state[StateVarsConfig.DEEP_RESEARCH_REPORT_DELIVERED] = True
                # The note is built from the annotations relayed now, so it is persisted with the
                # tool response; a report delivered before the note was introduced has none.
                return DeepResearchTurnResult(
                    content=content + dial_streamer.citations_note,
                    report_delivered=True,
                    report=content,
                )

            # Clarification / plan-for-approval. Saved before the content opens the result stage,
            # so the stage is named after the query Deep Research may have rewritten in this call.
            self._append_turn(session, user_message, content, deep_research_state or {})
            self._save_session(state, session)
            target.append_content(content)
            return DeepResearchTurnResult(content=content, report_delivered=False)
        except APIError as e:
            if dial_streamer is not None and dial_streamer.state:
                self._append_turn(session, user_message, dial_streamer.content, dial_streamer.state)
                self._save_session(state, session)
            raise DeepResearchFailedError(_get_display_message(e)) from e
        finally:
            duration_s = time.monotonic() - time_start
            if (duration_manager := get_llm_call_duration_manager()) is not None:
                duration_manager.add_duration(
                    LLMCallDurationItem(deployment=deployment_id, duration_s=duration_s)
                )


class DeepResearchTool(StatGptTool[DeepResearchToolConfig], tool_type=ToolTypes.DEEP_RESEARCH):
    """Starts a Deep Research session from a research question."""

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

    Built on demand from the `deep_research` tool config."""

    @classmethod
    def get_args_schema(cls, tool_config: DeepResearchToolConfig) -> type[ResumeDeepResearchArgs]:
        return ResumeDeepResearchArgs

    @property
    def _resume_stages_config(self) -> ResumeStagesConfig:
        return self._tool_config.details.resume_stages_config

    @property
    def stage_name(self) -> str:
        return self._resume_stages_config.tool_call_name or self.default_stage_name

    @property
    def result_stage_name(self) -> str:
        return self._resume_stages_config.tool_result_name or self.default_result_stage_name

    def _render_stage_name(self, name: str, args: dict, inputs: dict) -> str:
        """Expose the session's original question and latest query, read from the state, to
        stage-name templates."""
        session = DeepResearchSession.from_state(ChainParameters.get_state(inputs))
        original_question = session.original_question if session else None
        # The latest query falls back to the original question, so both are missing together.
        latest_query = session.latest_query if session else None
        if original_question is None or latest_query is None:
            if any(f"{{{key}}}" in name for key in (_ORIGINAL_QUESTION_KEY, _QUERY_KEY)):
                return self._resume_stages_config.missing_question_name
            return super()._render_stage_name(name, args, inputs)
        session_values = {_ORIGINAL_QUESTION_KEY: original_question, _QUERY_KEY: latest_query}
        return super()._render_stage_name(name, args | session_values, inputs)

    @classmethod
    def build(cls, tool_config: DeepResearchToolConfig, channel_config: ChannelConfig) -> Self:
        """Construct the resume tool from the (start) Deep Research tool config."""
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
