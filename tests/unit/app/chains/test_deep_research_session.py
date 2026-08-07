from unittest.mock import Mock

from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role

from statgpt.app.chains.deep_research.deep_research_tool import DeepResearchRunner
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas import DeepResearchSession
from statgpt.app.utils import OpenAiToDialStreamer
from statgpt.app.utils.message_history import History


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
        stored = DeepResearchSession(messages=[{"role": "user", "content": "q"}]).model_dump(
            mode="json"
        )
        session = DeepResearchSession.from_state({StateVarsConfig.DEEP_RESEARCH_SESSION: stored})
        assert session is not None
        assert session.messages == [{"role": "user", "content": "q"}]


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
        assert session.messages == []

    def test_stored_session_is_resumed(self):
        stored = DeepResearchSession(
            messages=[{"role": "user", "content": "q"}],
        ).model_dump(mode="json")
        session = DeepResearchRunner._load_session({StateVarsConfig.DEEP_RESEARCH_SESSION: stored})
        assert session.messages == [{"role": "user", "content": "q"}]


class TestSessionMutationHelpers:
    def test_append_turn_records_user_and_assistant_with_state(self):
        session = DeepResearchSession()
        dr_state = {"preparation": {"research_started": False}}
        DeepResearchRunner._append_turn(session, "my query", "the question", dr_state)

        assert session.messages[0] == {"role": "user", "content": "my query"}
        assert session.messages[1]["role"] == "assistant"
        assert session.messages[1]["content"] == "the question"
        assert session.messages[1]["custom_content"]["state"] == dr_state

    def test_save_session_serializes_to_state(self):
        state: dict = {}
        session = DeepResearchSession(messages=[{"role": "user", "content": "q"}])
        DeepResearchRunner._save_session(state, session)

        stored = DeepResearchSession.model_validate(state[StateVarsConfig.DEEP_RESEARCH_SESSION])
        assert stored.messages == [{"role": "user", "content": "q"}]

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

    def test_continuation_replays_prior_conversation(self):
        prior = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "custom_content": {"state": {}}},
        ]
        session = DeepResearchSession(messages=list(prior))
        messages = DeepResearchRunner._build_request_messages(None, session, "answer")
        assert messages == prior + [{"role": "user", "content": "answer"}]


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


class TestLastUserMessageText:
    def test_returns_plain_string_content(self):
        history = History(messages=[DialMessage(role=Role.USER, content="hello there")])
        assert history.last_user_message_text == "hello there"
