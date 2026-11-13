"""How `History` shows earlier turns to the agent: a failed turn is not shown as an empty answer,
nor a turn an error stopped mid-answer as a complete one (#805)."""

import pytest
from aidial_sdk.chat_completion import CustomContent
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from statgpt.app.config import StateVarsConfig
from statgpt.app.default_prompts import supreme_agent_default_prompts
from statgpt.app.utils.message_history import History

_FAILED_TURN_MESSAGE = supreme_agent_default_prompts.failed_turn_message_to_agent
_SHOWN_ERROR = "You've exceeded your token limit. Please try again in 14 hours 30 minutes"


def _history(answer: DialMessage) -> History:
    return History(
        messages=[
            DialMessage(role=Role.USER, content="go"),
            answer,
            DialMessage(role=Role.USER, content="continue"),
        ]
    )


def test_failed_turn_is_shown_as_a_note_with_the_error_shown_to_the_user():
    state = {
        StateVarsConfig.ERROR: "Error code: 429 - {'error': {'message': 'Hit cost rate limit.'}}",
        StateVarsConfig.ERROR_DISPLAY_MESSAGE: _SHOWN_ERROR,
    }
    history = _history(
        DialMessage(role=Role.ASSISTANT, content="", custom_content=CustomContent(state=state))
    )

    messages = history.get_langchain_messages(include_tool_messages=True)

    assert [m.type for m in messages] == ["human", "system", "human"]
    assert messages[1] == SystemMessage(
        content=supreme_agent_default_prompts.failed_turn_with_error_message_to_agent.format(
            error=_SHOWN_ERROR
        )
    )
    assert "Hit cost rate limit" not in messages[1].content


@pytest.mark.parametrize(
    "custom_content",
    [
        pytest.param(CustomContent(state={StateVarsConfig.ERROR: "Error code: 429"}), id="error"),
        pytest.param(CustomContent(), id="no-state"),
        pytest.param(None, id="no-custom-content"),
    ],
)
def test_failed_turn_with_an_unknown_error_is_shown_as_a_note(custom_content):
    """The turn failed, but the error the user was shown is not known: e.g. the request was
    rejected, or the connection dropped."""
    history = _history(DialMessage(role=Role.ASSISTANT, content="", custom_content=custom_content))

    messages = history.get_langchain_messages(include_tool_messages=True)

    assert [m.type for m in messages] == ["human", "system", "human"]
    assert messages[1] == SystemMessage(content=_FAILED_TURN_MESSAGE)


def test_failed_turn_keeps_its_tool_messages_before_the_note():
    turn = History.create_empty()
    turn.add_tool_message(
        AIMessage(
            content="",
            tool_calls=[{"name": "resume_deep_research", "args": {}, "id": "c1"}],
        )
    )
    turn.add_tool_message(ToolMessage(content="failed", tool_call_id="c1", status="error"))
    state: dict = {}
    turn.dump_state(state)
    state[StateVarsConfig.ERROR] = "boom"
    state[StateVarsConfig.ERROR_DISPLAY_MESSAGE] = _SHOWN_ERROR

    history = _history(
        DialMessage(role=Role.ASSISTANT, content="", custom_content=CustomContent(state=state))
    )
    messages = history.get_langchain_messages(include_tool_messages=True)

    assert [m.type for m in messages] == ["human", "ai", "tool", "system", "human"]
    assert _SHOWN_ERROR in messages[3].content


def test_turn_stopped_mid_answer_keeps_the_part_shown_before_the_note():
    """The error was raised after part of the answer had streamed: the agent sees that part, and
    then that it was stopped there by the error the user was shown."""
    state = {
        StateVarsConfig.ERROR: "Error code: 429 - {'error': {'message': 'Hit cost rate limit.'}}",
        StateVarsConfig.ERROR_DISPLAY_MESSAGE: _SHOWN_ERROR,
    }
    history = _history(
        DialMessage(
            role=Role.ASSISTANT,
            content="US GDP grew in 2023, from",
            custom_content=CustomContent(state=state),
        )
    )

    messages = history.get_langchain_messages(include_tool_messages=True)

    assert [m.type for m in messages] == ["human", "ai", "system", "human"]
    assert messages[1] == AIMessage(content="US GDP grew in 2023, from")
    assert messages[2] == SystemMessage(
        content=supreme_agent_default_prompts.interrupted_turn_message_to_agent.format(
            error=_SHOWN_ERROR
        )
    )
    assert "Hit cost rate limit" not in messages[2].content


def test_turn_stopped_mid_answer_keeps_its_tool_messages_first():
    turn = History.create_empty()
    turn.add_tool_message(
        AIMessage(content="", tool_calls=[{"name": "data_query", "args": {}, "id": "c1"}])
    )
    turn.add_tool_message(ToolMessage(content="data", tool_call_id="c1"))
    state: dict = {}
    turn.dump_state(state)
    state[StateVarsConfig.ERROR] = "boom"
    state[StateVarsConfig.ERROR_DISPLAY_MESSAGE] = _SHOWN_ERROR

    history = _history(
        DialMessage(
            role=Role.ASSISTANT,
            content="US GDP grew in 2023, from",
            custom_content=CustomContent(state=state),
        )
    )
    messages = history.get_langchain_messages(include_tool_messages=True)

    assert [m.type for m in messages] == ["human", "ai", "tool", "ai", "system", "human"]
    assert messages[3] == AIMessage(content="US GDP grew in 2023, from")
    assert _SHOWN_ERROR in messages[4].content


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(
            DialMessage(
                role=Role.ASSISTANT,
                content="US GDP grew in 2023, from An error occurred while processing your request.",
                custom_content=CustomContent(state={StateVarsConfig.ERROR: "boom"}),
            ),
            id="failed-with-the-error-in-the-answer",
        ),
        pytest.param(
            DialMessage(
                role=Role.ASSISTANT,
                content="Your message contains only commands.",
                custom_content=CustomContent(
                    state={StateVarsConfig.ERROR: "User message consists only of commands"}
                ),
            ),
            id="failed-with-an-explanation",
        ),
        pytest.param(
            DialMessage(
                role=Role.ASSISTANT,
                content="US GDP grew in 2023.",
                custom_content=CustomContent(
                    state={StateVarsConfig.ERROR: None, StateVarsConfig.ERROR_DISPLAY_MESSAGE: None}
                ),
            ),
            id="succeeded",
        ),
        pytest.param(
            DialMessage(
                role=Role.ASSISTANT,
                content="",
                custom_content=CustomContent(state={StateVarsConfig.ERROR: None}),
            ),
            id="succeeded-empty",
        ),
        pytest.param(
            DialMessage(role=Role.ASSISTANT, content="US GDP grew in"), id="no-state-stopped"
        ),
    ],
)
def test_other_answers_are_kept(answer):
    """Kept as they are, with no note: the answer shows the error itself, or there was none to
    show (a response the user stopped has no state)."""
    messages = _history(answer).get_langchain_messages(include_tool_messages=True)

    assert [m.type for m in messages] == ["human", "ai", "human"]
    assert messages[1] == AIMessage(content=answer.content)


def test_failed_turn_is_kept_without_tool_messages():
    """The guardrails and the Deep Research query check see the conversation as it is."""
    history = _history(DialMessage(role=Role.ASSISTANT, content=""))

    messages = history.get_langchain_messages(include_tool_messages=False)

    assert messages[1] == AIMessage(content="")
