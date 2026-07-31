import json
import typing as t
from collections.abc import Sequence

from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import MessageContentPart, Role
from aidial_sdk.chat_completion import ToolCall as DialToolCall
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages import ToolCall as LangChainToolCall
from langchain_core.messages import ToolMessage

from statgpt.app.config import StateVarsConfig
from statgpt.app.schemas.tool_artifact import ToolArtifact
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.app.utils.message_interceptors.commands_interceptor import CommandsInterceptor
from statgpt.app.utils.message_interceptors.system_msg_interceptor import SystemMessageInterceptor
from statgpt.common.config import multiline_logger as logger


class InvalidHistoryError(Exception):
    """Raised when DIAL messages cannot be converted into a valid history."""


class CommandOnlyMessageError(Exception):
    """Raised when a user message consists solely of interceptable commands."""


class InvalidToolCallError(Exception):
    """Raised when a DIAL tool call cannot be converted into a LangChain tool call.

    Callers map this to a request- or history-level error, depending on whether the tool
    call arrived in the current request or was echoed back to us as part of the history.
    """


def dump_dial_messages(messages: Sequence[DialMessage]) -> list[dict]:
    """Serialize DIAL messages for debugging.

    `custom_content` is dropped entirely from assistant messages - it carries the tool
    messages, stages and attachments we echo back, which would dwarf the rest of the dump.
    Attachment payloads are dropped from the other messages.
    """
    result = []
    for msg in messages:
        exclude: set[str] | dict[str, t.Any] = (
            {'custom_content'}
            if msg.role == Role.ASSISTANT
            else {'custom_content': {'attachments': {'__all__': {'data'}}}}
        )
        result.append(msg.model_dump(mode='json', exclude_none=True, exclude=exclude))
    return result


def _convert_content(
    content: str | list[MessageContentPart],
) -> str | list[str | dict[str, t.Any]]:
    """Convert DIAL message content to LangChain-compatible format."""
    if isinstance(content, str):
        return content
    parts: list[str | dict[str, t.Any]] = [part.model_dump(exclude_none=True) for part in content]
    return parts


def dial_tool_call_to_langchain_tool_call(tool_call: DialToolCall) -> LangChainToolCall:
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        raise InvalidToolCallError(
            f"Tool call {tool_call.id!r} ({tool_call.function.name!r})"
            f" has invalid JSON arguments: {e}"
        ) from e

    return LangChainToolCall(
        id=tool_call.id,
        name=tool_call.function.name,
        args=args,
        type='tool_call',
    )


class History:
    def __init__(
        self,
        messages: list[DialMessage],
        tool_messages: list[AIMessage | ToolMessage | SystemMessage] | None = None,
    ):
        self._messages: list[DialMessage] = messages
        self._tool_messages: list[AIMessage | ToolMessage | SystemMessage] = (
            [] if tool_messages is None else tool_messages
        )

    @classmethod
    def create_empty(cls) -> 'History':
        return cls(messages=[])

    @classmethod
    async def from_dial_with_interceptors(
        cls,
        messages: list[DialMessage],
        state: dict[str, t.Any],
        data_service: ChannelServiceFacade,
    ) -> 'History':
        """Create an instance of the `History` class from DIAL messages,
        and intercept supported commands from user messages.

        [!] Update the `state` dictionary with the flags corresponding to the commands.
        """
        cls._validate_received_messages(messages)

        interceptors = [
            CommandsInterceptor.create_default(),
            SystemMessageInterceptor(data_service=data_service),
        ]
        for interceptor in interceptors:
            messages = await interceptor.process_messages(messages=messages, state=state)

        cls._validate_processed_messages(messages)
        return cls(messages=messages)

    @staticmethod
    def _validate_received_messages(messages: list[DialMessage]) -> None:
        """Reject messages we cannot build a history from, as they were received.

        Runs before the interceptors so that the reported index matches the request, and
        so that messages emptied by interception are not reported as malformed input.
        """
        for ix, msg in enumerate(messages):
            if msg.role == Role.USER and not msg.content:
                raise InvalidHistoryError(f"User message at index {ix} has empty content")
            if msg.role == Role.TOOL and not msg.tool_call_id:
                raise InvalidHistoryError(f"Tool message at index {ix} has no tool_call_id")

    @staticmethod
    def _validate_processed_messages(messages: list[DialMessage]) -> None:
        """Reject user messages that the interceptors emptied.

        `_validate_received_messages` has already rejected empty user content, so anything
        empty at this point was stripped by the `CommandsInterceptor`, i.e. the message
        consisted of nothing but commands.
        """
        for msg in messages:
            if msg.role == Role.USER and not msg.content:
                raise CommandOnlyMessageError("User message consists only of commands")

    def prepend(self, other: 'History') -> None:
        self._messages = other._messages + self._messages
        if other._tool_messages:
            # This should not happen because `fake_history` does not have tool messages.
            # If `other` has values in the `_tool_messages` attribute,
            #   we need to implement this according to the situation.
            raise ValueError(f"Prepending tool messages is not supported!\n{other._tool_messages=}")

    def add_tool_message(self, tool_message: AIMessage | ToolMessage | SystemMessage) -> None:
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

    def get_tool_messages(self) -> list[AIMessage | ToolMessage | SystemMessage]:
        return self._tool_messages

    def get_last_non_tool_message(self) -> DialMessage:
        return self._messages[-1]

    def get_ai_messages(self) -> list[DialMessage]:
        return [msg for msg in self._messages if msg.role == Role.ASSISTANT]

    # def get_dial_messages(self) -> list[DialMessage]:
    #     # TODO: add tool messages
    #     return self._messages

    @classmethod
    def dial_to_langchain_message(
        cls, msg: DialMessage
    ) -> AIMessage | HumanMessage | ToolMessage | SystemMessage:
        """Convert a DialMessage to a LangChain message."""
        if msg.role == Role.USER:
            if not (usr_msg_content := msg.content):
                raise InvalidHistoryError("User message content is empty")
            return HumanMessage(content=_convert_content(usr_msg_content))
        elif msg.role == Role.ASSISTANT:
            try:
                tool_calls = (
                    [dial_tool_call_to_langchain_tool_call(t) for t in msg.tool_calls]
                    if msg.tool_calls
                    else []
                )
            except InvalidToolCallError as e:
                # Echoed back to us as part of the history, so the history is unusable.
                raise InvalidHistoryError(f"Assistant message has an invalid tool call: {e}") from e
            return AIMessage(
                content=_convert_content(msg.content) if msg.content else '',
                tool_calls=tool_calls,
            )
        elif msg.role == Role.TOOL:
            if not msg.tool_call_id:
                raise InvalidHistoryError("Tool message has no tool_call_id")
            msg_content = _convert_content(msg.content) if msg.content else ''
            return ToolMessage(content=msg_content, tool_call_id=msg.tool_call_id)
        elif msg.role == Role.SYSTEM:
            msg_content = _convert_content(msg.content) if msg.content else ''
            return SystemMessage(content=msg_content)
        else:
            raise ValueError(f"Unknown message role: {msg.role!r}")

    @staticmethod
    def _log_messages(messages: Sequence[BaseMessage]) -> None:
        # Debug logging with readable format
        logger.debug("=" * 60)
        logger.debug("MESSAGE HISTORY (Total: %d messages)", len(messages))
        logger.debug("=" * 60)
        for i, msg in enumerate(messages, 1):
            if isinstance(msg, HumanMessage):
                msg_content = msg.content if msg.content else ''
                if not isinstance(msg_content, str):
                    raise ValueError("ToolMessage content must be a string")
                content_preview = (
                    (msg_content[:100] + "...") if len(msg_content) > 100 else msg_content
                )
                logger.debug("[%d] USER: %s", i, content_preview)
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in msg.tool_calls]
                    logger.debug("[%d] ASSISTANT: [Tool calls: %s]", i, ", ".join(tool_names))
                else:
                    msg_content = msg.content if msg.content else ''
                    if not isinstance(msg_content, str):
                        raise ValueError("ToolMessage content must be a string")
                    content_preview = (
                        (msg_content[:100] + "...") if len(msg_content) > 100 else msg_content
                    )
                    logger.debug("[%d] ASSISTANT: %s", i, content_preview or "[Empty content]")
            elif isinstance(msg, ToolMessage):
                msg_content = msg.content if msg.content else ''
                if not isinstance(msg_content, str):
                    raise ValueError("ToolMessage content must be a string")
                content_preview = (
                    (msg_content[:100] + "...") if len(msg_content) > 100 else msg_content
                )
                logger.debug("[%d] TOOL (id=%s): %s", i, msg.tool_call_id, content_preview)
            elif isinstance(msg, SystemMessage):
                msg_content = msg.content if msg.content else ''
                if not isinstance(msg_content, str):
                    raise ValueError("ToolMessage content must be a string")
                content_preview = (
                    (msg_content[:100] + "...") if len(msg_content) > 100 else msg_content
                )
                logger.debug("[%d] SYSTEM: %s", i, content_preview or "[Empty content]")
            else:
                logger.debug("[%d] %s: [Unknown message type]", i, type(msg).__name__)
        logger.debug("=" * 60)

    def get_langchain_messages(
        self, include_tool_messages: bool
    ) -> list[AIMessage | HumanMessage | ToolMessage | SystemMessage]:
        chat_history: list[AIMessage | HumanMessage | ToolMessage | SystemMessage] = []
        for msg in self._messages:
            if msg.role == Role.USER:
                chat_history.append(self.dial_to_langchain_message(msg))
            elif msg.role == Role.ASSISTANT:
                if include_tool_messages:
                    chat_history.extend(self._extract_tool_messages_from_message(msg))
                chat_history.append(self.dial_to_langchain_message(msg))
            elif msg.role == Role.TOOL:
                chat_history.append(self.dial_to_langchain_message(msg))
            elif msg.role == Role.SYSTEM:
                chat_history.append(self.dial_to_langchain_message(msg))
            else:
                raise ValueError(f"Unknown message role: {msg.role!r}")

        if include_tool_messages:
            chat_history.extend(self.get_tool_messages())

        self._log_messages(chat_history)

        return chat_history

    def _dump_tool_messages_to_state(self, state: dict) -> None:
        result = []
        for msg in self._tool_messages:
            msg_dump: dict = msg.model_dump(mode='json', exclude={'artifact'}, exclude_none=True)

            artifact: ToolArtifact | None
            if (artifact := getattr(msg, 'artifact', None)) is not None:
                if msg_dump.get('custom_content') is None:
                    msg_dump['custom_content'] = {}

                msg_dump['custom_content']['state'] = artifact.state.model_dump(mode='json')
            result.append(msg_dump)
        state[StateVarsConfig.TOOL_MESSAGES] = result

    def dump_state(self, state: dict) -> None:
        self._dump_tool_messages_to_state(state)

    @staticmethod
    def _extract_tool_messages_from_message(
        message: DialMessage,
    ) -> list[AIMessage | ToolMessage | SystemMessage]:
        tool_messages: list[AIMessage | ToolMessage | SystemMessage] = []

        if message.custom_content is None:
            return tool_messages

        state_dict: dict
        if message.custom_content.state and isinstance(message.custom_content.state, dict):
            state_dict = message.custom_content.state
            for tool_msg in state_dict.get(StateVarsConfig.TOOL_MESSAGES, []):
                msg_type = tool_msg.get('type')
                if msg_type == 'ai':  # Tool Call
                    tool_messages.append(AIMessage.model_validate(tool_msg))
                elif msg_type == 'tool':  # Tool Response
                    tool_messages.append(ToolMessage.model_validate(tool_msg))
                elif msg_type == 'system':
                    tool_messages.append(SystemMessage.model_validate(tool_msg))
                else:
                    logger.info(f"Tool message: {tool_msg}")
                    # Our own state, echoed back: an unknown type means the conversation
                    # predates a change to the tool message format.
                    raise InvalidHistoryError(f"Unknown tool message type: {msg_type!r}")
        return tool_messages
