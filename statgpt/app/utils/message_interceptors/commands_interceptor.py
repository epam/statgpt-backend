import re
import typing as t

from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from pydantic import BaseModel

from statgpt.app.config import StateVarsConfig
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.common.config import multiline_logger as logger

from .base import BaseMessageInterceptor


class InterceptableCommand(BaseModel):
    command: str
    state_var: str

    @property
    def re_pattern(self) -> str:
        return rf'!{self.command}(\s+)'

    def process_query(self, query: str, state: dict | None) -> str:
        """
        To correctly parse command, it must have a space afterwards, refer to the regex pattern used
        for matching and defined in the `re_pattern` property.
        """
        match = re.search(self.re_pattern, query)
        if not match:
            return query

        # at least one command instance found

        if state is not None:
            state[self.state_var] = True

        # remove all command instances from the query
        query_edited = re.sub(self.re_pattern, '', query)
        query_edited = query_edited.strip()
        return query_edited


class CommandsInterceptor(BaseMessageInterceptor):
    def __init__(self, commands: list[InterceptableCommand]):
        self._commands = commands

    @classmethod
    def create_default(cls, force_all_commands: bool = False) -> 'CommandsInterceptor':
        commands = [
            InterceptableCommand(
                command='show_debug_stages',
                state_var=StateVarsConfig.SHOW_DEBUG_STAGES,
            ),
        ]
        if dial_app_settings.enable_dev_commands or force_all_commands:
            logger.info("CommandsInterceptor: dev commands enabled")
            commands += [
                InterceptableCommand(
                    command='out_of_scope_only',
                    state_var=StateVarsConfig.CMD_OUT_OF_SCOPE_ONLY,
                ),
                InterceptableCommand(
                    command='rag_prefilter_only',
                    state_var=StateVarsConfig.CMD_RAG_PREFILTER_ONLY,
                ),
                InterceptableCommand(
                    command='skip_data_query_summarization',
                    state_var=StateVarsConfig.CMD_SKIP_DATA_QUERY_SUMMARIZATION,
                ),
                InterceptableCommand(
                    command='skip_tools_execution',
                    state_var=StateVarsConfig.CMD_SKIP_TOOLS_EXECUTION,
                ),
            ]
        else:
            logger.info("CommandsInterceptor: dev commands disabled")
        return cls(commands=commands)

    async def process_messages(
        self, messages: list[DialMessage], state: dict[str, t.Any]
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
        """Remove commands from the query. Used in eval pipeline."""
        for cmd in self._commands:
            query = cmd.process_query(query=query, state=None)
        return query
