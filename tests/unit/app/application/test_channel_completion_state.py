"""Tests for the state `ChannelCompletion` carries across turns and logs (#760)."""

import logging
from unittest.mock import MagicMock

from aidial_sdk.chat_completion import CustomContent
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role

from statgpt.app.application.channel_completion import ChannelCompletion
from statgpt.app.config import StateVarsConfig

_SESSION_KEY = StateVarsConfig.DEEP_RESEARCH_SESSION
_SESSION = {
    "turns": [{"user_message": "q", "assistant_content": "plan?", "deep_research_state": {}}]
}


def _user(content: str) -> DialMessage:
    return DialMessage(role=Role.USER, content=content)


def _answer(state: dict | None) -> DialMessage:
    custom_content = CustomContent(state=state) if state is not None else None
    return DialMessage(role=Role.ASSISTANT, content="a", custom_content=custom_content)


def _init_state(*messages: DialMessage) -> dict:
    return ChannelCompletion.init_state(MagicMock(messages=[*messages, _user("next")]))


def test_first_message_gets_defaults_only() -> None:
    assert _SESSION_KEY not in _init_state()


def test_session_is_carried_from_the_previous_answer() -> None:
    state = _init_state(_user("q"), _answer({_SESSION_KEY: _SESSION}))

    assert state[_SESSION_KEY] == _SESSION


def test_stopped_answer_does_not_lose_the_session() -> None:
    """A stopped answer has no state, so the values come from the answer before it."""
    state = _init_state(
        _user("q"),
        _answer({_SESSION_KEY: _SESSION, StateVarsConfig.SHOW_DEBUG_STAGES: True}),
        _user("go"),
        _answer(None),
    )

    assert state[_SESSION_KEY] == _SESSION
    assert state[StateVarsConfig.SHOW_DEBUG_STAGES] is True


def test_dropped_session_stays_dropped() -> None:
    state = _init_state(
        _user("q"),
        _answer({_SESSION_KEY: _SESSION}),
        _user("approve"),
        _answer({StateVarsConfig.DEEP_RESEARCH_REPORT_DELIVERED: True}),
    )

    assert _SESSION_KEY not in state


def test_state_log_carries_only_keys_and_size(caplog) -> None:
    choice = MagicMock()
    state = {_SESSION_KEY: {"turns": [{"deep_research_state": {"report": "research text"}}]}}

    with caplog.at_level(logging.INFO, logger="statgpt.app.application.channel_completion"):
        ChannelCompletion.set_dial_state(state, choice)

    choice.set_state.assert_called_once_with(state)
    assert _SESSION_KEY in caplog.text
    assert "research text" not in caplog.text
