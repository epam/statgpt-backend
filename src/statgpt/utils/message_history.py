import json
import re
import typing as t

from aidial_sdk.chat_completion import Role
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.messages import ToolCall as LangChainToolCall
from langchain_core.messages import ToolMessage
from pydantic import BaseModel

from common.config import multiline_logger as logger
from common.schemas.dial import Message as DialMessage
from common.schemas.dial import ToolCall as DialToolCall
from statgpt.config import DialAppConfig, StateVarsConfig


def dial_tool_call_to_langchain_tool_call(tool_call: DialToolCall) -> LangChainToolCall:
    return LangChainToolCall(
        id=tool_call.id,
        name=tool_call.function.name,
        args=json.loads(tool_call.function.arguments),
        type='tool_call',
    )


class InterceptableCommand(BaseModel):
    command: str
    state_var: str

    @property
    def re_pattern(self) -> str:
        return rf'!{self.command}(\s+)'

    def process_query(self, query: str, state: dict | None) -> str:
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


class CommandsInterceptor:
    def __init__(self, commands: list[InterceptableCommand]):
        self._commands = commands

    @classmethod
    def create_default(cls) -> 'CommandsInterceptor':
        commands = [
            InterceptableCommand(
                command='show_debug_stages',
                state_var=StateVarsConfig.SHOW_DEBUG_STAGES,
            ),
        ]
        if DialAppConfig.ENABLE_DEV_COMMANDS:
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
            ]
        else:
            logger.info("CommandsInterceptor: dev commands disabled")
        return cls(commands=commands)

    def process_messages(
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


class History:
    def __init__(
        self,
        messages: list[DialMessage],
        tool_messages: list[AIMessage | ToolMessage] | None = None,
    ):
        self._messages = messages
        self._tool_messages = [] if tool_messages is None else tool_messages

    @classmethod
    def create_empty(cls) -> 'History':
        return cls(messages=[])

    @classmethod
    def from_dial_with_commands_interceptor(
        cls, messages: list[DialMessage], state: dict[str, t.Any]
    ) -> 'History':
        """Create an instance of the `History` class from DIAL messages,
        and intercept supported commands from user messages.

        [!] Update the `state` dictionary with the flags corresponding to the commands.
        """
        interceptor = CommandsInterceptor.create_default()
        messages = interceptor.process_messages(messages=messages, state=state)
        return cls(messages=messages)

    def prepend(self, other: 'History') -> None:
        self._messages = other._messages + self._messages
        if other._tool_messages:
            # This should not happen because `fake_history` does not have tool messages.
            # If `other` has values in the `_tool_messages` attribute,
            #   we need to implement this according to the situation.
            raise ValueError(f"Prepending tool messages is not supported!\n{other._tool_messages=}")

    def add_tool_message(self, tool_message: AIMessage | ToolMessage) -> None:
        self._tool_messages.append(tool_message)

    def add_chunk_as_tool_message(self, chunk: AIMessageChunk) -> AIMessage:
        msg_dump = chunk.model_dump(exclude={'type'})
        try:
            message = AIMessage.model_validate(msg_dump)
        except Exception as e:
            logger.info(f"{msg_dump=}")
            raise e
        self.add_tool_message(message)
        return message

    def add_dial_message(self, message: DialMessage) -> None:
        self._messages.append(message)

    def add_tool_message_as_dial_message(self, tool_message: ToolMessage) -> None:
        self._messages.append(
            DialMessage(
                role=Role.TOOL, content=tool_message.content, tool_call_id=tool_message.tool_call_id
            )
        )

    def get_tool_messages(self) -> list[AIMessage | ToolMessage]:
        return self._tool_messages

    def get_last_non_tool_message(self) -> DialMessage:
        return self._messages[-1]

    # def get_dial_messages(self) -> list[DialMessage]:
    #     # TODO: add tool messages
    #     return self._messages

    def get_langchain_messages(
        self, include_tool_messages: bool = True
    ) -> list[AIMessage | HumanMessage | ToolMessage]:
        chat_history: list[AIMessage | HumanMessage | ToolMessage] = []
        for msg in self._messages:
            if msg.role == Role.USER:
                if not (usr_msg_content := msg.content):
                    raise ValueError("User message content is empty")
                chat_history.append(HumanMessage(content=usr_msg_content))
            elif msg.role == Role.ASSISTANT:
                if include_tool_messages:
                    chat_history.extend(self._extract_tool_messages_from_message(msg))
                chat_history.append(
                    AIMessage(
                        content=msg.content or '',
                        tool_calls=(
                            [dial_tool_call_to_langchain_tool_call(t) for t in msg.tool_calls]
                            if msg.tool_calls
                            else []
                        ),
                    )
                )
            elif msg.role == Role.TOOL:
                chat_history.append(ToolMessage(content=msg.content, tool_call_id=msg.tool_call_id))
            else:
                raise ValueError(f"Unknown message role: {msg.role!r}")

        if include_tool_messages:
            chat_history.extend(self.get_tool_messages())
        logger.info(f"Chat history: {chat_history}")
        return chat_history

    # def get_openai_messages(self) -> list[dict[str, str]]:
    #     # TODO: add tool messages
    #     return [
    #         msg.dict(include={'role', 'content'})
    #         for msg in self._messages
    #         if msg.role in {Role.ASSISTANT, Role.USER}
    #     ]

    @staticmethod
    def _extract_tool_messages_from_message(message: DialMessage) -> list[AIMessage | ToolMessage]:
        tool_messages = []

        if message.custom_content is None:
            return tool_messages

        state_dict: dict
        if state_dict := message.custom_content.state:
            for tool_msg in state_dict.get(StateVarsConfig.TOOL_MESSAGES, []):
                msg_type = tool_msg.get('type')
                if msg_type == 'ai':  # Tool Call
                    tool_messages.append(AIMessage.model_validate(tool_msg))
                elif msg_type == 'tool':  # Tool Response
                    tool_messages.append(ToolMessage.model_validate(tool_msg))
                else:
                    logger.info(f"Tool message: {tool_msg}")
                    raise RuntimeError(f"Unknown tool message type: {msg_type!r}")
        return tool_messages
