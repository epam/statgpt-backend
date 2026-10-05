"""Each response continues the citation ids of the conversation (#730).

The tool responses of earlier turns stay in the history the agent reads, citation tags included,
so a new response must not cite another source with an id they already use.
"""

from types import SimpleNamespace
from typing import Any

from aidial_sdk.chat_completion import CustomContent
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role

from statgpt.app.application.channel_completion import ChannelCompletion
from statgpt.app.config import StateVarsConfig


def _user(content: str = "q") -> DialMessage:
    return DialMessage(role=Role.USER, content=content)


def _assistant(state: dict[str, Any] | None) -> DialMessage:
    custom_content = CustomContent(state=state) if state is not None else None
    return DialMessage(role=Role.ASSISTANT, content="a", custom_content=custom_content)


def _next_id(*messages: DialMessage) -> str:
    request: Any = SimpleNamespace(messages=list(messages))
    return ChannelCompletion._init_citation_id_space(request).next_id()


def test_a_new_conversation_starts_at_the_first_id() -> None:
    assert _next_id(_user()) == "citation001"


def test_continues_from_the_count_of_the_last_response() -> None:
    assert (
        _next_id(
            _user(),
            _assistant({StateVarsConfig.CITATION_COUNT: 2}),
            _user(),
            _assistant({StateVarsConfig.CITATION_COUNT: 5}),
            _user(),
        )
        == "citation006"
    )


def test_a_response_without_a_count_does_not_restart_the_numbering() -> None:
    """A response that failed before setting its state carries no count."""
    assert (
        _next_id(
            _user(),
            _assistant({StateVarsConfig.CITATION_COUNT: 3}),
            _user(),
            _assistant(None),
            _user(),
            _assistant({StateVarsConfig.CITATION_COUNT: "9"}),
            _user(),
        )
        == "citation004"
    )
