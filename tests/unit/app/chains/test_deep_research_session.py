import json
from unittest.mock import Mock

from statgpt.app.chains.deep_research.deep_research_tool import DeepResearchRunner
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas import DeepResearchSession, DeepResearchTurn
from statgpt.app.utils import OpenAiToDialStreamer


class TestBuildRequestMessages:
    def test_fresh_session_prepends_system_prompt(self):
        messages = DeepResearchRunner._build_request_messages("SYS", DeepResearchSession(), "hello")
        assert messages == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
        ]

    def test_fresh_session_without_system_prompt(self):
        messages = DeepResearchRunner._build_request_messages(None, DeepResearchSession(), "hello")
        assert messages == [{"role": "user", "content": "hello"}]

    def test_continuation_replays_prior_turns(self):
        session = DeepResearchSession(
            turns=[
                DeepResearchTurn(
                    user_message="q1",
                    assistant_content="a1",
                    deep_research_state={"preparation": {"research_started": False}},
                )
            ]
        )
        messages = DeepResearchRunner._build_request_messages(None, session, "answer")
        assert messages == [
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": "a1",
                "custom_content": {"state": {"preparation": {"research_started": False}}},
            },
            {"role": "user", "content": "answer"},
        ]


class TestResearchStarted:
    def test_not_started_during_clarification(self):
        dr_state = {
            "preparation": {
                "research_started": False,
                "clarification": {"questions": ["When?", "Where?"]},
            }
        }
        assert DeepResearchRunner._research_started(dr_state) is False

    def test_started(self):
        dr_state = {"preparation": {"research_started": True, "clarification": None}}
        assert DeepResearchRunner._research_started(dr_state) is True

    def test_missing_or_empty(self):
        assert DeepResearchRunner._research_started(None) is False
        assert DeepResearchRunner._research_started({}) is False


class TestFromState:
    def test_absent_returns_none(self):
        assert DeepResearchSession.from_state({}) is None

    def test_present_is_validated(self):
        stored = DeepResearchSession(
            turns=[DeepResearchTurn(user_message="q", assistant_content="a")]
        ).model_dump(mode="json")
        session = DeepResearchSession.from_state({StateVarsConfig.DEEP_RESEARCH_SESSION: stored})
        assert session is not None
        assert len(session.turns) == 1
        assert session.turns[0].user_message == "q"
        assert session.turns[0].assistant_content == "a"


class TestDropFromState:
    def test_removes_session(self):
        state = {
            StateVarsConfig.DEEP_RESEARCH_SESSION: DeepResearchSession().model_dump(mode="json")
        }
        DeepResearchSession.drop_from_state(state)
        assert StateVarsConfig.DEEP_RESEARCH_SESSION not in state

    def test_is_noop_when_absent(self):
        state: dict = {}
        DeepResearchSession.drop_from_state(state)
        assert state == {}


class TestLoadSession:
    def test_no_stored_session_starts_fresh(self):
        session = DeepResearchRunner._load_session({})
        assert session.turns == []

    def test_stored_session_is_resumed(self):
        stored = DeepResearchSession(
            turns=[DeepResearchTurn(user_message="q", assistant_content="a")]
        ).model_dump(mode="json")
        session = DeepResearchRunner._load_session({StateVarsConfig.DEEP_RESEARCH_SESSION: stored})
        assert len(session.turns) == 1
        assert session.turns[0].user_message == "q"


class TestSessionMutationHelpers:
    def test_append_turn_records_turn_with_state(self):
        session = DeepResearchSession()
        dr_state = {"preparation": {"research_started": False}}
        DeepResearchRunner._append_turn(session, "my query", "the question", dr_state)

        assert len(session.turns) == 1
        turn = session.turns[0]
        assert turn.user_message == "my query"
        assert turn.assistant_content == "the question"
        assert turn.deep_research_state == dr_state

    def test_save_session_serializes_to_state(self):
        state: dict = {}
        session = DeepResearchSession(
            turns=[DeepResearchTurn(user_message="q", assistant_content="a")]
        )
        DeepResearchRunner._save_session(state, session)

        stored = DeepResearchSession.model_validate(state[StateVarsConfig.DEEP_RESEARCH_SESSION])
        assert len(stored.turns) == 1
        assert stored.turns[0].user_message == "q"
        # Regression: the persisted turn must not embed a `custom_content` key, which the DIAL
        # chat client strips from message-shaped objects while round-tripping state.
        assert "custom_content" not in json.dumps(state[StateVarsConfig.DEEP_RESEARCH_SESSION])

    def test_drop_session_removes_it(self):
        state: dict = {
            StateVarsConfig.DEEP_RESEARCH_SESSION: DeepResearchSession().model_dump(mode="json")
        }
        DeepResearchRunner._drop_session(state)
        assert StateVarsConfig.DEEP_RESEARCH_SESSION not in state

    def test_drop_session_is_noop_when_absent(self):
        state: dict = {}
        DeepResearchRunner._drop_session(state)
        assert state == {}


class TestStreamerStateCapture:
    def _streamer(self) -> OpenAiToDialStreamer:
        return OpenAiToDialStreamer(
            Mock(),
            Mock(),
            deployment="d",
            show_debug_stages=False,
            stages_config=Mock(),
            stream_content=True,
        )

    def test_captures_custom_content_state(self):
        streamer = self._streamer()
        assert streamer.state is None
        streamer._process_custom_content({"state": {"preparation": {"research_started": True}}})
        assert streamer.state == {"preparation": {"research_started": True}}

    def test_absent_state_leaves_none(self):
        streamer = self._streamer()
        streamer._process_custom_content({})
        assert streamer.state is None


class _RecordingChoice:
    def __init__(self) -> None:
        self.content = ""
        self.attachments: list[dict] = []

    def append_content(self, content: str) -> None:
        self.content += content

    def add_attachment(self, **kwargs) -> None:
        self.attachments.append(kwargs)


class TestDeliverReport:
    def test_delivers_content_and_attachments_verbatim(self):
        choice = _RecordingChoice()
        attachment = {"type": "text/markdown", "title": "Report.md", "data": "# Report"}
        DeepResearchRunner._deliver_report(choice, "Final report body.", [attachment])

        assert choice.content == "Final report body."
        assert choice.attachments == [
            {
                "type": "text/markdown",
                "title": "Report.md",
                "data": "# Report",
                "url": None,
                "reference_url": None,
                "reference_type": None,
            }
        ]

    def test_no_attachments_appends_only_content(self):
        choice = _RecordingChoice()
        DeepResearchRunner._deliver_report(choice, "Just text.", [])

        assert choice.content == "Just text."
        assert choice.attachments == []
