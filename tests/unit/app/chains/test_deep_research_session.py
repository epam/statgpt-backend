from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from statgpt.app.chains.deep_research.deep_research_tool import DeepResearchTool
from statgpt.app.chains.deep_research.mediation import (
    build_transcript,
    format_auto_answers,
    triage_questions,
)
from statgpt.app.chains.supreme_agent import SupremeAgentExecutor
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas import DeepResearchSession, DeepResearchStatus, ToolResponseStatus
from statgpt.app.utils import OpenAiToDialStreamer


def _tool_call(name: str, id_: str) -> dict:
    return {"name": name, "id": id_, "args": {}, "type": "tool_call"}


class TestParsePreparation:
    def test_clarification_questions(self):
        dr_state = {
            "preparation": {
                "research_started": False,
                "clarification": {"questions": ["When?", "Where?"]},
            }
        }
        started, questions = DeepResearchTool._parse_preparation(dr_state)
        assert started is False
        assert questions == ["When?", "Where?"]

    def test_research_started(self):
        dr_state = {"preparation": {"research_started": True, "clarification": None}}
        started, questions = DeepResearchTool._parse_preparation(dr_state)
        assert started is True
        assert questions == []

    def test_missing_or_empty(self):
        assert DeepResearchTool._parse_preparation(None) == (False, [])
        assert DeepResearchTool._parse_preparation({}) == (False, [])


class TestFromState:
    def test_absent_returns_none(self):
        assert DeepResearchSession.from_state({}) is None

    def test_present_is_validated(self):
        stored = DeepResearchSession(messages=[{"role": "user", "content": "q"}]).model_dump(
            mode="json"
        )
        session = DeepResearchSession.from_state({StateVarsConfig.DEEP_RESEARCH_SESSION: stored})
        assert session is not None
        assert session.messages == [{"role": "user", "content": "q"}]


class TestLoadSession:
    def test_no_stored_session_starts_fresh(self):
        session = DeepResearchTool._load_session({})
        assert session.messages == []
        assert session.is_in_progress

    def test_in_progress_session_is_resumed(self):
        stored = DeepResearchSession(
            status=DeepResearchStatus.IN_PROGRESS,
            messages=[{"role": "user", "content": "q"}],
        ).model_dump(mode="json")
        session = DeepResearchTool._load_session({StateVarsConfig.DEEP_RESEARCH_SESSION: stored})
        assert session.messages == [{"role": "user", "content": "q"}]

    def test_completed_session_treated_as_fresh(self):
        stored = DeepResearchSession(
            status=DeepResearchStatus.COMPLETED,
            messages=[{"role": "user", "content": "q"}],
        ).model_dump(mode="json")
        session = DeepResearchTool._load_session({StateVarsConfig.DEEP_RESEARCH_SESSION: stored})
        assert session.messages == []


class TestSessionMutationHelpers:
    def test_append_turn_records_user_and_assistant_with_state(self):
        session = DeepResearchSession()
        dr_state = {"preparation": {"research_started": False}}
        DeepResearchTool._append_turn(session, "my query", "the question", dr_state)

        assert session.status == DeepResearchStatus.IN_PROGRESS
        assert session.messages[0] == {"role": "user", "content": "my query"}
        assert session.messages[1]["role"] == "assistant"
        assert session.messages[1]["content"] == "the question"
        assert session.messages[1]["custom_content"]["state"] == dr_state

    def test_save_session_serializes_new_fields(self):
        state: dict = {}
        session = DeepResearchSession(
            outstanding_questions=["Where?"],
            pending_auto_answers=["Q: When?\nA: 2024"],
            context_summary="the summary",
        )
        DeepResearchTool._save_session(state, session)

        stored = DeepResearchSession.model_validate(state[StateVarsConfig.DEEP_RESEARCH_SESSION])
        assert stored.outstanding_questions == ["Where?"]
        assert stored.pending_auto_answers == ["Q: When?\nA: 2024"]
        assert stored.context_summary == "the summary"

    def test_drop_session_removes_it(self):
        state: dict = {
            StateVarsConfig.DEEP_RESEARCH_SESSION: DeepResearchSession().model_dump(mode="json")
        }
        DeepResearchTool._drop_session(state)
        assert StateVarsConfig.DEEP_RESEARCH_SESSION not in state

    def test_drop_session_is_noop_when_absent(self):
        state: dict = {}
        DeepResearchTool._drop_session(state)
        assert state == {}


class TestSessionFieldDefaults:
    def test_new_fields_default_empty(self):
        session = DeepResearchSession()
        assert session.pending_auto_answers == []
        assert session.context_summary is None


class TestBuildTranscript:
    def test_keeps_only_user_and_assistant_text(self):
        messages = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="hello"),
            AIMessage(content="", tool_calls=[_tool_call("t", "1")]),
            ToolMessage(content="tool output", tool_call_id="1"),
            AIMessage(content="hi there"),
        ]
        assert build_transcript(messages) == "User: hello\n\nAssistant: hi there"

    def test_skips_blank_messages(self):
        assert build_transcript([HumanMessage(content="   ")]) == ""


class TestFormatAutoAnswers:
    def test_formats_question_answer_blocks(self):
        assert format_auto_answers({"When?": "2024", "Where?": "EU"}) == [
            "Q: When?\nA: 2024",
            "Q: Where?\nA: EU",
        ]

    def test_empty(self):
        assert format_auto_answers({}) == []


class TestTriageQuestionsTrivialBranches:
    async def test_no_questions_returns_empty(self):
        answered, pending = await triage_questions(
            api_key="k", model_config=Mock(), context="ctx", questions=[]
        )
        assert answered == {}
        assert pending == []

    async def test_blank_context_marks_all_pending(self):
        answered, pending = await triage_questions(
            api_key="k", model_config=Mock(), context="   ", questions=["a", "b"]
        )
        assert answered == {}
        # Verbatim, original wording preserved.
        assert pending == ["a", "b"]


class TestResearchStarted:
    def test_true_when_flag_set(self):
        assert (
            DeepResearchTool._research_started({"preparation": {"research_started": True}}) is True
        )

    def test_false_for_clarification_or_missing(self):
        assert DeepResearchTool._research_started(None) is False
        assert (
            DeepResearchTool._research_started(
                {"preparation": {"clarification": {"questions": ["q"]}}}
            )
            is False
        )


class TestStreamerContentGate:
    def _streamer(self, target):
        return OpenAiToDialStreamer(
            target,
            Mock(),
            deployment="d",
            show_debug_stages=False,
            stages_config=Mock(),
            stream_content=False,
        )

    def test_enable_flushes_buffered_content_and_is_idempotent(self):
        target = Mock()
        streamer = self._streamer(target)
        streamer._content = "buffered report"
        streamer._attachments = [{"type": "text/markdown", "title": "t", "data": "d"}]

        streamer.enable_content_streaming()

        target.append_content.assert_called_once_with("buffered report")
        target.add_attachment.assert_called_once()
        # Second call must not re-flush.
        streamer.enable_content_streaming()
        target.append_content.assert_called_once()

    def test_no_flush_when_buffer_empty(self):
        target = Mock()
        streamer = self._streamer(target)
        streamer.enable_content_streaming()
        target.append_content.assert_not_called()


class TestBuildRequestMessages:
    def test_fresh_session_prepends_system_prompt(self):
        messages = DeepResearchTool._build_request_messages("SYS", DeepResearchSession(), "hello")
        assert messages == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
        ]

    def test_continuation_replays_prior_conversation(self):
        prior = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "custom_content": {"state": {}}},
        ]
        session = DeepResearchSession(messages=list(prior))
        messages = DeepResearchTool._build_request_messages(None, session, "answer")
        assert messages == prior + [{"role": "user", "content": "answer"}]


class TestGuardDeepResearchCalls:
    def test_no_deep_research_tool_keeps_all(self):
        calls = [_tool_call("x", "1")]
        kept, rejected = SupremeAgentExecutor._guard_deep_research_calls(calls, None)
        assert kept == calls
        assert rejected == []

    def test_single_call_kept(self):
        dr = SimpleNamespace(name="deep_research")
        calls = [_tool_call("deep_research", "1"), _tool_call("other", "2")]
        kept, rejected = SupremeAgentExecutor._guard_deep_research_calls(calls, dr)
        assert kept == calls
        assert rejected == []

    def test_duplicate_calls_rejected(self):
        dr = SimpleNamespace(name="deep_research")
        calls = [
            _tool_call("deep_research", "1"),
            _tool_call("deep_research", "2"),
            _tool_call("other", "3"),
        ]
        kept, rejected = SupremeAgentExecutor._guard_deep_research_calls(calls, dr)
        assert [c["id"] for c in kept] == ["1", "3"]
        assert len(rejected) == 1
        assert rejected[0].tool_call_id == "2"
        assert rejected[0].status == ToolResponseStatus.ERROR.value


class TestSupremeAgentSessionHelpers:
    def test_load_returns_none_when_absent(self):
        assert SupremeAgentExecutor._load_deep_research_session({}) is None

    def test_exit_removes_active_session(self):
        state = {
            StateVarsConfig.DEEP_RESEARCH_SESSION: DeepResearchSession().model_dump(mode="json")
        }
        SupremeAgentExecutor._exit_deep_research_session(state)
        assert StateVarsConfig.DEEP_RESEARCH_SESSION not in state

    def test_exit_is_noop_when_absent(self):
        state: dict = {}
        SupremeAgentExecutor._exit_deep_research_session(state)
        assert state == {}
