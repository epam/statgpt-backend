from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from langchain_core.messages import ToolMessage

from statgpt.app.utils.message_history import History


def _user_message(content: str) -> DialMessage:
    return DialMessage(role=Role.USER, content=content)


def test_copy_mutations_do_not_affect_original():
    original = History([_user_message("hi")])
    original.add_tool_message(ToolMessage(content="t0", tool_call_id="0"))

    clone = original.copy()
    clone.add_dial_message(_user_message("clone-only"))
    clone.add_tool_message(ToolMessage(content="t1", tool_call_id="1"))
    clone.prepend(History([_user_message("fake")]))

    assert [msg.content for msg in original._messages] == ["hi"]
    assert [msg.content for msg in original.get_tool_messages()] == ["t0"]
    assert [msg.content for msg in clone._messages] == ["fake", "hi", "clone-only"]
    assert [msg.content for msg in clone.get_tool_messages()] == ["t0", "t1"]


def test_original_mutations_do_not_affect_copy():
    original = History([_user_message("hi")])
    clone = original.copy()

    original.add_dial_message(_user_message("original-only"))
    original.add_tool_message(ToolMessage(content="t0", tool_call_id="0"))

    assert [msg.content for msg in clone._messages] == ["hi"]
    assert clone.get_tool_messages() == []


def test_copy_shares_message_objects():
    original = History([_user_message("hi")])
    clone = original.copy()

    assert clone._messages is not original._messages
    assert clone._messages[0] is original._messages[0]
