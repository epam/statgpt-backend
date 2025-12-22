import typing as t

from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role

from statgpt.app.schemas.state import InterceptableCommand, State
from statgpt.app.settings.dial_app import dial_app_settings

from .base import BaseMessageInterceptor


class CommandsInterceptor(BaseMessageInterceptor):
    def __init__(self, commands: list[InterceptableCommand]):
        self._commands = commands

    @classmethod
    def create_default(cls, force_all_commands: bool = False) -> t.Self:
        include_dev_commands = dial_app_settings.enable_dev_commands or force_all_commands
        commands = State.get_intercaptable_commands(include_dev_commands=include_dev_commands)
        return cls(commands=commands)

    async def process_messages(
        self, messages: list[DialMessage], state: State
    ) -> list[DialMessage]:
        """
        1. for last user message, remove commands and update state
        2. for rest of user messages, simply remove commands from message content
        """

        for ix, msg in enumerate(reversed(messages)):
            if msg.role != Role.USER:
                continue
            if not msg.content:
                continue

            state_or_none = state if ix == 0 else None

            for cmd in self._commands:
                msg_edited = cmd.process_query(query=msg.content, state=state_or_none)
                msg.content = msg_edited

        return messages

    def process_query(self, query: str) -> str:
        """Simply remove commands from the query, without updating state"""
        query_upd = query
        for cmd in self._commands:
            query_upd = cmd.process_query(query=query_upd, state=None)
        return query_upd
