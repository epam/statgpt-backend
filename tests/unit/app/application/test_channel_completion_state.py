"""Tests for the state `ChannelCompletion` carries across turns and logs (#760, #805)."""

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from aidial_sdk.chat_completion import CustomContent
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from aidial_sdk.exceptions import HTTPException as DIALException

from statgpt.app.application import channel_completion as channel_completion_module
from statgpt.app.application.channel_completion import ChannelCompletion
from statgpt.app.chains.deep_research import DeepResearchFailedError
from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.utils.message_history import History

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


# ~~~~~~~~~~~~~~~~~~~~~ the error shown to the user (#805) ~~~~~~~~~~~~~~~~~~~~~

_RATE_LIMIT_ERROR = openai.RateLimitError(
    "Error code: 429 - Hit cost rate limit. Day limit: $10.30 / $10.00.",
    response=httpx.Response(
        429,
        headers={"retry-after": "52200"},
        request=httpx.Request("POST", "http://core/openai/deployments/gpt/chat/completions"),
    ),
    body=None,
)


def _patch_failing_chain(monkeypatch, error: Exception) -> None:
    chain = MagicMock(ainvoke=AsyncMock(side_effect=error))
    factory = MagicMock(return_value=MagicMock(create_chain=AsyncMock(return_value=chain)))
    monkeypatch.setattr(channel_completion_module, "MainChainFactory", factory)
    monkeypatch.setattr(channel_completion_module, "create_auth_context", AsyncMock())
    monkeypatch.setattr(
        History,
        "from_dial_with_interceptors",
        AsyncMock(side_effect=lambda messages, **kwargs: History(messages=messages)),
    )
    monkeypatch.setattr(ChannelCompletion, "_calc_token_usage_costs", AsyncMock(return_value=[]))
    monkeypatch.setattr(ChannelCompletion, "_emit_deep_research_form_schema", AsyncMock())


async def _run_failing_turn() -> dict:
    """Run a turn whose chain fails, and return the state set on the answer."""
    choice = MagicMock()
    response = MagicMock()
    response.create_choice.return_value.__enter__.return_value = choice
    request = MagicMock(messages=[_user("research US GDP")])
    service = MagicMock()
    service.channel_config.token_usage.debug_only = True

    with pytest.raises(DIALException):
        await ChannelCompletion._channel_completion(
            request, response, service, datetime.now(), StatGPTConfiguration()
        )

    choice.set_state.assert_called_once()
    return choice.set_state.call_args.args[0]


@pytest.mark.parametrize(
    "error, shown",
    [
        pytest.param(
            DeepResearchFailedError("You've exceeded your daily cost limit (error reference: e98)"),
            "You've exceeded your daily cost limit (error reference: e98)",
            id="deep-research",
        ),
        pytest.param(
            _RATE_LIMIT_ERROR,
            "You've exceeded your token limit for the underlying deployment."
            " Please try again in 14 hours 30 minutes",
            id="rate-limit",
        ),
    ],
)
async def test_failed_turn_keeps_the_error_shown_to_the_user(monkeypatch, error, shown) -> None:
    _patch_failing_chain(monkeypatch, error)

    state = await _run_failing_turn()

    assert state[StateVarsConfig.ERROR] == str(error)
    assert state[StateVarsConfig.ERROR_DISPLAY_MESSAGE] == shown
