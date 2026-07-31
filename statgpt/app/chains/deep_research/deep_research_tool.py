import time
from typing import Annotated, Any, ClassVar

from mcp.types import ToolAnnotations
from pydantic import Field

from statgpt.app.chains.deep_research.mediation import (
    build_transcript,
    format_auto_answers,
    summarize_conversation,
    triage_questions,
)
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import GuardrailInput, StatGptTool, ToolArgs
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas import (
    DEEP_RESEARCH_ERROR_MESSAGE,
    DeepResearchSession,
    DeepResearchStatus,
    ToolArtifact,
    ToolMessageState,
)
from statgpt.app.utils import OpenAiToDialStreamer, openai
from statgpt.app.utils.dial_stages import ChoiceI
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.schemas import DeepResearchTool as DeepResearchToolConfig
from statgpt.common.schemas import LLMModelConfig, ToolTypes
from statgpt.common.schemas.llm_call_duration import LLMCallDurationItem
from statgpt.common.schemas.tool_details import DeepResearchDetails
from statgpt.common.utils.llm_call_duration_context import get_llm_call_duration_manager
from statgpt.common.utils.markdown import format_as_markdown_list


class DeepResearchArgs(ToolArgs):
    query: Annotated[str, GuardrailInput] = Field(
        description=(
            "The natural language input for Deep Research. For a new investigation this is the "
            "research question. While a Deep Research session is in progress this is the user's "
            "answer to the outstanding clarifying question(s) or their approval of the plan, "
            "enriched with any relevant details already known from the conversation."
        )
    )


class DeepResearchTool(StatGptTool[DeepResearchToolConfig], tool_type=ToolTypes.DEEP_RESEARCH):
    # Upper bound on Deep Research calls made within a single turn while the agent silently
    # auto-answers clarifying questions from context. Prevents an endless answer/re-ask loop; on
    # the final round any outstanding questions are handed to the user instead of auto-answered.
    MAX_CLARIFICATION_ROUNDS: ClassVar[int] = 4

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

        A completed session is treated as absent so a new query begins a new
        investigation (the finished report stays in the chat history)."""
        session = DeepResearchSession.from_state(state)
        if session is not None and session.is_in_progress:
            return session
        return DeepResearchSession()

    @staticmethod
    def _build_request_messages(
        system_prompt: str | None, session: DeepResearchSession, query: str
    ) -> list[dict[str, Any]]:
        """The DIAL messages sent to Deep Research: the replayed sub-conversation
        plus the new user input. Deep Research resumes from the `custom_content.state`
        it stored on the last assistant message."""
        messages: list[dict[str, Any]] = []
        # Deep Research reads its own system prompt from its app properties and skips the
        # system role on input; we still forward the configured prompt for parity.
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.extend(session.messages)
        messages.append({'role': 'user', 'content': query})
        return messages

    @staticmethod
    def _parse_preparation(dr_state: dict[str, Any] | None) -> tuple[bool, list[str]]:
        """Read Deep Research's persisted state and return ``(research_started, questions)``.

        CONTRACT: Deep Research sets ``preparation.research_started`` to ``True`` only once the
        plan is approved AND it has fully streamed the final report within this same response.
        We rely on that to treat the session as finished; if the deployment ever emits
        ``research_started`` before the report is delivered, the session would be closed
        prematurely and the report never resumed."""
        preparation = (dr_state or {}).get('preparation') or {}
        research_started = bool(preparation.get('research_started'))
        clarification = preparation.get('clarification') or {}
        questions = clarification.get('questions') or []
        return research_started, questions

    async def _arun(self, inputs: dict, query: str, **kwargs) -> tuple[str, ToolArtifact]:
        auth_context = ChainParameters.get_auth_context(inputs)
        choice = ChainParameters.get_choice(inputs)
        state = ChainParameters.get_state(inputs)
        history = ChainParameters.get_history(inputs)

        details = self._tool_config.details
        deployment_id = details.get_deployment_id()
        model_config = self._channel_config.supreme_agent.llm_model_config

        session = self._load_session(state)
        first_call = not session.messages

        # Merge any answers the agent held back from a previous turn (questions the user was not
        # asked) with the user's new reply, so Deep Research receives every answer together.
        user_content = query
        if session.pending_auto_answers:
            user_content = "\n\n".join([*session.pending_auto_answers, query])
            session.pending_auto_answers = []

        # Summarize the conversation once, at the start, and forward it to Deep Research as context.
        if first_call:
            if session.context_summary is None:
                transcript = build_transcript(
                    history.get_langchain_messages(include_tool_messages=False)
                )
                session.context_summary = await summarize_conversation(
                    api_key=auth_context.api_key, model_config=model_config, transcript=transcript
                )
            if session.context_summary:
                user_content = (
                    "<conversation_summary>\n"
                    f"{session.context_summary}\n"
                    "</conversation_summary>\n\n"
                    f"{user_content}"
                )

        show_debug_stages = (
            state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False) or details.always_show_stages
        )
        client = openai.get_async_client(api_key=auth_context.api_key)

        time_start = time.monotonic()
        async with client:
            result = await self._mediated_research_loop(
                client=client,
                choice=choice,
                state=state,
                session=session,
                details=details,
                deployment_id=deployment_id,
                show_debug_stages=show_debug_stages,
                first_user_content=user_content,
                auth_context=auth_context,
                model_config=model_config,
            )

        duration_s = time.monotonic() - time_start
        if (duration_manager := get_llm_call_duration_manager()) is not None:
            duration_manager.add_duration(
                LLMCallDurationItem(deployment=deployment_id, duration_s=duration_s)
            )

        artifact = ToolArtifact(state=ToolMessageState(type=self.tool_type))
        return result, artifact

    async def _mediated_research_loop(
        self,
        *,
        client,
        choice: ChoiceI,
        state: dict,
        session: DeepResearchSession,
        details: DeepResearchDetails,
        deployment_id: str,
        show_debug_stages: bool,
        first_user_content: str,
        auth_context: AuthContext,
        model_config: LLMModelConfig,
    ) -> str:
        """Drive one turn of the clarification flow.

        Calls Deep Research, and while it only asks clarifying questions the agent can answer from
        context, replies on the user's behalf and calls again — up to ``MAX_CLARIFICATION_ROUNDS``.
        Stops (and returns the user-facing text) as soon as Deep Research delivers its report, or
        as soon as a question needs the user, whose remaining questions are shown verbatim."""
        user_content = first_user_content

        for round_idx in range(self.MAX_CLARIFICATION_ROUNDS):
            messages = self._build_request_messages(details.system_prompt, session, user_content)
            content, dr_state, attachments, ok, streamed_live = await self._call_deep_research(
                client=client,
                choice=choice,
                deployment_id=deployment_id,
                details=details,
                show_debug_stages=show_debug_stages,
                messages=messages,
            )

            if not ok:
                # Request/network/mid-stream failure. Deep Research owns this turn and is force-
                # selected, so surface a friendly message and leave the session untouched.
                choice.append_content(DEEP_RESEARCH_ERROR_MESSAGE)
                return DEEP_RESEARCH_ERROR_MESSAGE

            if dr_state is None:
                logger.warning(
                    "Deep Research returned no custom_content.state; surfacing its content as-is "
                    "and leaving the session unchanged."
                )
                self._surface_to_user(choice, content, attachments)
                return content

            self._append_turn(session, user_content, content, dr_state)
            research_started, questions = self._parse_preparation(dr_state)

            if research_started:
                # Plan approved and the final report delivered: finish the session. The report was
                # streamed live if research_started surfaced mid-stream; otherwise flush it now.
                if not streamed_live:
                    self._surface_to_user(choice, content, attachments)
                self._drop_session(state)
                return content

            if not questions:
                # Deep Research responded without structured questions (e.g. awaiting plan
                # approval): surface its message and wait for the user.
                session.outstanding_questions = []
                self._save_session(state, session)
                self._surface_to_user(choice, content, attachments)
                return content

            answered, pending = await triage_questions(
                api_key=auth_context.api_key,
                model_config=model_config,
                context=session.context_summary or "",
                questions=questions,
            )

            is_last_round = round_idx == self.MAX_CLARIFICATION_ROUNDS - 1
            if answered and not pending and not is_last_round:
                # Every question answered from context: reply to Deep Research and continue
                # silently, without involving the user.
                user_content = "\n\n".join(format_auto_answers(answered))
                continue

            if not pending:
                # All questions were answerable but the round cap is reached: ask the user
                # directly rather than looping further.
                pending = questions
                answered = {}

            # Hold the auto-answers for next turn and ask the user the rest, verbatim.
            session.outstanding_questions = pending
            session.pending_auto_answers = format_auto_answers(answered)
            self._save_session(state, session)
            message = self._format_clarification_message(pending)
            choice.append_content(message)
            return message

        # The loop always returns above; this satisfies the type checker.
        return DEEP_RESEARCH_ERROR_MESSAGE

    @staticmethod
    async def _call_deep_research(
        *,
        client,
        choice: ChoiceI,
        deployment_id: str,
        details: DeepResearchDetails,
        show_debug_stages: bool,
        messages: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]], bool, bool]:
        """One Deep Research call.

        Content starts buffered (``stream_content=False``) so the caller can filter clarifications
        (surface only the questions the user must answer) or consume them silently while the agent
        auto-answers. The moment Deep Research signals that research has started, we switch to live
        streaming so the final report is shown as-is, token by token. Progress stages always stream
        live. Returns ``(content, state, attachments, ok, streamed_live)`` with ``ok=False`` on any
        failure; ``streamed_live`` is True once the report was streamed to the user."""
        dial_streamer = OpenAiToDialStreamer(
            choice,
            choice,
            deployment=deployment_id,
            stream_content=False,
            show_debug_stages=show_debug_stages,
            stages_config=details.stages_config,
        )
        streamed_live = False
        with dial_streamer:
            try:
                stream = await client.chat.completions.create(
                    model=deployment_id, stream=True, messages=messages
                )
                async for chunk in stream:
                    dial_streamer.send_chunk(chunk)
                    if not streamed_live and DeepResearchTool._research_started(
                        dial_streamer.state
                    ):
                        # Research has begun: stream the report as-is from here on, flushing any
                        # buffered lead-in content in one go.
                        dial_streamer.enable_content_streaming()
                        streamed_live = True
            except Exception as e:
                logger.exception(e)
                return "", None, [], False, False

        return (
            dial_streamer.content,
            dial_streamer.state,
            dial_streamer.attachments,
            True,
            streamed_live,
        )

    @staticmethod
    def _research_started(dr_state: dict[str, Any] | None) -> bool:
        research_started, _ = DeepResearchTool._parse_preparation(dr_state)
        return research_started

    @staticmethod
    def _surface_to_user(choice: ChoiceI, content: str, attachments: list[dict[str, Any]]) -> None:
        """Append buffered Deep Research output (content + attachments) to the visible answer."""
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

    @staticmethod
    def _format_clarification_message(questions: list[str]) -> str:
        listed = format_as_markdown_list(questions, list_type="ordered")
        return (
            "Deep Research needs a bit more information before it can continue. "
            "Please answer the following:\n\n" + listed
        )

    @staticmethod
    def _append_turn(
        session: DeepResearchSession,
        user_content: str,
        assistant_content: str,
        dr_state: dict[str, Any],
    ) -> None:
        """Record a user/assistant exchange (assistant carrying Deep Research's state) so the
        sub-conversation can be replayed on the next call/turn."""
        session.messages.append({'role': 'user', 'content': user_content})
        session.messages.append(
            {
                'role': 'assistant',
                'content': assistant_content,
                'custom_content': {'state': dr_state},
            }
        )
        session.status = DeepResearchStatus.IN_PROGRESS

    @staticmethod
    def _save_session(state: dict, session: DeepResearchSession) -> None:
        state[StateVarsConfig.DEEP_RESEARCH_SESSION] = session.model_dump(mode='json')

    @staticmethod
    def _drop_session(state: dict) -> None:
        # Drop a finished session rather than persisting the (potentially large) report and
        # accumulated state on every later turn — the report already lives in the chat history.
        state.pop(StateVarsConfig.DEEP_RESEARCH_SESSION, None)
