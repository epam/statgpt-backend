import json
from unittest.mock import Mock

from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role

from statgpt.app.chains.deep_research.deep_research_tool import DeepResearchRunner
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas import DeepResearchSession, DeepResearchTurn
from statgpt.app.utils import OpenAiToDialStreamer
from statgpt.app.utils.message_history import History


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
                    dr_state={"preparation": {"research_started": False}},
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
        assert turn.dr_state == dr_state

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


class TestLastUserMessageText:
    def test_returns_plain_string_content(self):
        history = History(messages=[DialMessage(role=Role.USER, content="hello there")])
        assert history.last_user_message_text == "hello there"


class _CollectingTarget:
    """Minimal `append_content` sink that records what the streamer displays."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def append_content(self, content: str) -> None:
        self.chunks.append(content)

    @property
    def displayed(self) -> str:
        return "".join(self.chunks)


class TestStreamerLeadingQueryStrip:
    def _streamer(self, target, strip_leading_query: bool) -> OpenAiToDialStreamer:
        return OpenAiToDialStreamer(
            target,
            Mock(),
            deployment="d",
            show_debug_stages=False,
            stages_config=Mock(),
            stream_content=True,
            strip_leading_query=strip_leading_query,
        )

    def _feed(self, streamer: OpenAiToDialStreamer, *chunks: str) -> None:
        for chunk in chunks:
            streamer._process_content(chunk)
        streamer._flush_leading_buffer()

    def test_strips_leading_query_echo_across_chunk_boundary(self):
        target = _CollectingTarget()
        streamer = self._streamer(target, strip_leading_query=True)
        # The `Query:` line is split across chunks to prove the display waits for a full line.
        self._feed(streamer, 'Query: "Retriev', 'e US GDP."\n\nPlan:\n', '1. do it\n')

        assert target.displayed == "Plan:\n1. do it\n"
        # `content` stays verbatim so Deep Research session replay is unaffected.
        assert streamer.content == 'Query: "Retrieve US GDP."\n\nPlan:\n1. do it\n'

    def test_keeps_content_without_query_echo(self):
        target = _CollectingTarget()
        streamer = self._streamer(target, strip_leading_query=True)
        self._feed(streamer, "Here is the plan\n\n1. step")

        assert target.displayed == "Here is the plan\n\n1. step"
        assert streamer.content == "Here is the plan\n\n1. step"

    def test_keeps_leading_query_line_without_quotes(self):
        target = _CollectingTarget()
        streamer = self._streamer(target, strip_leading_query=True)
        # A natural-language line that merely starts with "Query:" (no quoted echo) must be kept:
        # only the exact `Query: "..."` echo shape is stripped.
        self._feed(streamer, "Query: which country did you mean?\n\nPlan:")

        assert target.displayed == "Query: which country did you mean?\n\nPlan:"
        assert streamer.content == "Query: which country did you mean?\n\nPlan:"

    def test_strip_disabled_streams_query_echo_verbatim(self):
        target = _CollectingTarget()
        streamer = self._streamer(target, strip_leading_query=False)
        self._feed(streamer, 'Query: "x"\n\nPlan:\n')

        assert target.displayed == 'Query: "x"\n\nPlan:\n'
        assert streamer.content == 'Query: "x"\n\nPlan:\n'

    def test_flush_emits_single_line_without_trailing_newline(self):
        target = _CollectingTarget()
        streamer = self._streamer(target, strip_leading_query=True)
        self._feed(streamer, "What frequency?")

        assert target.displayed == "What frequency?"
        assert streamer.content == "What frequency?"

    def test_pure_echo_line_without_newline_shows_nothing(self):
        target = _CollectingTarget()
        streamer = self._streamer(target, strip_leading_query=True)
        self._feed(streamer, 'Query: "x"')

        assert target.displayed == ""
        assert streamer.content == 'Query: "x"'

    def test_strips_echo_with_leading_newline(self):
        target = _CollectingTarget()
        streamer = self._streamer(target, strip_leading_query=True)
        self._feed(streamer, '\nQuery: "x"', '\n\nPlan:')

        assert target.displayed == "Plan:"
        assert streamer.content == '\nQuery: "x"\n\nPlan:'
