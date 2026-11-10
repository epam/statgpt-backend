"""How `History` shows earlier turns to the agent: a failed turn is not shown as an empty answer
(#805)."""

import pytest
from aidial_sdk.chat_completion import CustomContent
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from statgpt.app.config import StateVarsConfig
from statgpt.app.default_prompts import supreme_agent_default_prompts
from statgpt.app.utils.message_history import History

_FAILED_TURN_MESSAGE = supreme_agent_default_prompts.failed_turn_message_to_agent


def _history(answer: DialMessage) -> History:
    return History(
        messages=[
            DialMessage(role=Role.USER, content="go"),
            answer,
            DialMessage(role=Role.USER, content="continue"),
        ]
    )


@pytest.mark.parametrize(
    "custom_content",
    [
        pytest.param(CustomContent(state={StateVarsConfig.ERROR: "Error code: 429"}), id="error"),
        pytest.param(CustomContent(), id="no-state"),
        pytest.param(None, id="no-custom-content"),
    ],
)
def test_failed_turn_is_shown_as_a_note(custom_content):
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

    history = _history(
        DialMessage(role=Role.ASSISTANT, content="", custom_content=CustomContent(state=state))
    )
    messages = history.get_langchain_messages(include_tool_messages=True)

    assert [m.type for m in messages] == ["human", "ai", "tool", "system", "human"]
    assert messages[3].content == _FAILED_TURN_MESSAGE


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(
            DialMessage(
                role=Role.ASSISTANT,
                content="An error occurred while processing your request.",
                custom_content=CustomContent(state={StateVarsConfig.ERROR: "boom"}),
            ),
            id="failed-with-shown-answer",
        ),
        pytest.param(
            DialMessage(
                role=Role.ASSISTANT,
                content="",
                custom_content=CustomContent(state={StateVarsConfig.ERROR: None}),
            ),
            id="succeeded-empty",
        ),
        pytest.param(DialMessage(role=Role.ASSISTANT, content="Hello."), id="no-state-answered"),
    ],
)
def test_other_answers_are_kept(answer):
    messages = _history(answer).get_langchain_messages(include_tool_messages=True)

    assert messages[1] == AIMessage(content=answer.content)


def test_failed_turn_is_kept_without_tool_messages():
    """The guardrails and the Deep Research query check see the conversation as it is."""
    history = _history(DialMessage(role=Role.ASSISTANT, content=""))

    messages = history.get_langchain_messages(include_tool_messages=False)

    assert messages[1] == AIMessage(content="")
