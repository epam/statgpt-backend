"""Tests for the state `ChannelCompletion` carries across turns and logs (#760, #805)."""

import logging
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from aidial_sdk.chat_completion import CustomContent
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from aidial_sdk.exceptions import HTTPException as DIALException
from langchain_core.messages import AIMessage, ToolMessage

from statgpt.app.application import channel_completion as channel_completion_module
from statgpt.app.application.channel_completion import CONTINUE_NOTE, ChannelCompletion
from statgpt.app.chains.deep_research import DeepResearchFailedError
from statgpt.app.config import ChainParametersConfig as ParamsConfig
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


_RATE_LIMIT_SHOWN = (
    "You've exceeded your token limit for the underlying deployment."
    " The limit resets in 14 hours 30 minutes."
)
_FORBIDDEN_ERROR = openai.PermissionDeniedError(
    "Error code: 403 - forbidden",
    response=httpx.Response(
        403, request=httpx.Request("POST", "http://core/openai/deployments/gpt/chat/completions")
    ),
    body=None,
)
_TOOL_CALL = AIMessage(
    content="",
    tool_calls=[{"id": "call-1", "name": "data_query", "args": {"query": "US GDP"}}],
)
_TOOL_RESPONSE = ToolMessage(content="US GDP: 27.7T", tool_call_id="call-1")


def _patch_chain(
    monkeypatch, error: Exception | None, *, tool_messages: list | None = None
) -> None:
    """Patch the main chain to add `tool_messages` to the history and then raise `error`."""

    async def ainvoke(inputs: dict, config: dict) -> dict:
        history = inputs[ParamsConfig.HISTORY]
        for message in tool_messages or []:
            history.add_tool_message(message)
        if error is not None:
            raise error
        return inputs

    chain = MagicMock(ainvoke=AsyncMock(side_effect=ainvoke))
    factory = MagicMock(return_value=MagicMock(create_chain=AsyncMock(return_value=chain)))
    monkeypatch.setattr(channel_completion_module, "MainChainFactory", factory)
    monkeypatch.setattr(channel_completion_module, "create_auth_context", AsyncMock())
    monkeypatch.setattr(
        History,
        "from_dial_with_interceptors",
        AsyncMock(side_effect=lambda messages, **kwargs: History(messages=messages)),
    )
    monkeypatch.setattr(ChannelCompletion, "_calc_token_usage_costs", AsyncMock(return_value=[]))


@dataclass
class _Turn:
    state: dict
    error: DIALException | None
    form_schema: dict | None


async def _run_turn(
    *,
    stream: bool = True,
    deep_research_available: bool = True,
    earlier_messages: tuple[DialMessage, ...] = (),
) -> _Turn:
    """Run a turn with the patched chain, and return what it set on the answer."""
    choice = MagicMock()
    response = MagicMock()
    response.create_choice.return_value.__enter__.return_value = choice
    request = MagicMock(messages=[*earlier_messages, _user("research US GDP")], stream=stream)
    service = MagicMock()
    service.channel_config.token_usage.debug_only = True
    service.channel_config.is_deep_research_available_for = AsyncMock(
        return_value=deep_research_available
    )

    error: DIALException | None = None
    try:
        await ChannelCompletion._channel_completion(
            request, response, service, datetime.now(), StatGPTConfiguration()
        )
    except DIALException as e:
        error = e

    choice.set_state.assert_called_once()
    assert choice.set_form_schema.call_count <= 1
    form_schema = (
        choice.set_form_schema.call_args.args[0] if choice.set_form_schema.call_args else None
    )
    return _Turn(choice.set_state.call_args.args[0], error, form_schema)


def _has_continue_button(form_schema: dict | None) -> bool:
    if form_schema is None or "button" not in form_schema["properties"]:
        return False
    [option] = form_schema["properties"]["button"]["oneOf"]
    assert option["title"] == "Continue"
    assert option["dial:widgetOptions"]["submit"] is True
    assert option["dial:widgetOptions"]["populateText"] == "Continue"
    return True


@pytest.mark.parametrize(
    "error, shown",
    [
        pytest.param(
            DeepResearchFailedError("You've exceeded your daily cost limit (error reference: e98)"),
            "You've exceeded your daily cost limit (error reference: e98)",
            id="deep-research",
        ),
        pytest.param(_RATE_LIMIT_ERROR, _RATE_LIMIT_SHOWN, id="rate-limit"),
    ],
)
async def test_failed_turn_keeps_the_error_shown_to_the_user(monkeypatch, error, shown) -> None:
    _patch_chain(monkeypatch, error)

    turn = await _run_turn()

    assert turn.state[StateVarsConfig.ERROR] == str(error)
    assert turn.state[StateVarsConfig.ERROR_DISPLAY_MESSAGE] == f"{shown}\n\n{CONTINUE_NOTE}"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ the "Continue" button (#831) ~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize(
    "error, shown",
    [
        pytest.param(_RATE_LIMIT_ERROR, _RATE_LIMIT_SHOWN, id="rate-limit"),
        pytest.param(
            DeepResearchFailedError("Research failed (ref 7f3a). Please retry."),
            "Research failed (ref 7f3a). Please retry.",
            id="deep-research-with-display-message",
        ),
        pytest.param(
            DeepResearchFailedError(),
            "Deep Research could not complete this request.",
            id="deep-research-without-display-message",
        ),
    ],
)
async def test_streamed_resumable_failure_carries_the_button_and_the_note(
    monkeypatch, error, shown
) -> None:
    _patch_chain(monkeypatch, error)

    turn = await _run_turn(stream=True)

    assert _has_continue_button(turn.form_schema)
    assert "deep_research" in turn.form_schema["properties"]
    assert turn.error is not None
    assert turn.error.display_message == f"{shown}\n\n{CONTINUE_NOTE}"
    assert turn.state[StateVarsConfig.ERROR_DISPLAY_MESSAGE] == turn.error.display_message


async def test_deep_research_failure_with_a_session_kept_carries_the_button(monkeypatch) -> None:
    _patch_chain(monkeypatch, DeepResearchFailedError())

    turn = await _run_turn(earlier_messages=(_user("q"), _answer({_SESSION_KEY: _SESSION})))

    assert _has_continue_button(turn.form_schema)
    [toggle] = turn.form_schema["properties"]["deep_research"]["oneOf"]
    assert toggle["const"] is True


async def test_rate_limit_without_deep_research_gets_a_button_only_schema(monkeypatch) -> None:
    _patch_chain(monkeypatch, _RATE_LIMIT_ERROR)

    turn = await _run_turn(stream=True, deep_research_available=False)

    assert _has_continue_button(turn.form_schema)
    assert list(turn.form_schema["properties"]) == ["button"]


@pytest.mark.parametrize(
    "error, shown",
    [
        pytest.param(_RATE_LIMIT_ERROR, _RATE_LIMIT_SHOWN, id="rate-limit"),
        pytest.param(
            DeepResearchFailedError("Research failed (ref 7f3a). Please retry."),
            "Research failed (ref 7f3a). Please retry.",
            id="deep-research",
        ),
    ],
)
async def test_non_streaming_failure_has_no_button_and_no_note(monkeypatch, error, shown) -> None:
    _patch_chain(monkeypatch, error)

    turn = await _run_turn(stream=False)

    assert not _has_continue_button(turn.form_schema)
    assert turn.error is not None
    assert turn.error.display_message == shown


async def test_forbidden_deployment_has_no_button(monkeypatch) -> None:
    _patch_chain(monkeypatch, _FORBIDDEN_ERROR)

    turn = await _run_turn(stream=True)

    assert not _has_continue_button(turn.form_schema)
    assert turn.error is not None
    assert CONTINUE_NOTE not in turn.error.display_message


async def test_successful_turn_has_no_button(monkeypatch) -> None:
    _patch_chain(monkeypatch, None)

    turn = await _run_turn(stream=True)

    assert turn.error is None
    assert not _has_continue_button(turn.form_schema)


async def test_successful_turn_without_deep_research_has_no_form_schema(monkeypatch) -> None:
    _patch_chain(monkeypatch, None)

    turn = await _run_turn(stream=True, deep_research_available=False)

    assert turn.form_schema is None


# ~~~~~~~~~~~~~~~~~~~~~~~~ tool messages of a rate-limited turn ~~~~~~~~~~~~~~~~~~~~~~~~


async def test_rate_limited_turn_saves_its_tool_messages(monkeypatch) -> None:
    _patch_chain(monkeypatch, _RATE_LIMIT_ERROR, tool_messages=[_TOOL_CALL, _TOOL_RESPONSE])

    turn = await _run_turn()

    saved = turn.state[StateVarsConfig.TOOL_MESSAGES]
    assert [message["type"] for message in saved] == ["ai", "tool"]
    assert saved[0]["tool_calls"][0]["id"] == "call-1"
    assert saved[1]["tool_call_id"] == "call-1"
    assert saved[1]["content"] == "US GDP: 27.7T"


async def test_rate_limit_before_the_history_was_built_saves_no_tool_messages(monkeypatch) -> None:
    _patch_chain(monkeypatch, None)
    monkeypatch.setattr(
        History, "from_dial_with_interceptors", AsyncMock(side_effect=_RATE_LIMIT_ERROR)
    )

    turn = await _run_turn()

    assert turn.error is not None
    assert StateVarsConfig.TOOL_MESSAGES not in turn.state
