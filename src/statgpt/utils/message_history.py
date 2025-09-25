import json
import typing as t
from collections.abc import Sequence

from aidial_sdk.chat_completion import Role
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages import ToolCall as LangChainToolCall
from langchain_core.messages import ToolMessage

from common.config import multiline_logger as logger
from common.schemas.dial import Message as DialMessage
from common.schemas.dial import ToolCall as DialToolCall
from statgpt.config import StateVarsConfig
from statgpt.schemas.tool_artifact import ToolArtifact
from statgpt.services import ChannelServiceFacade
from statgpt.utils.message_interceptors.commands_interceptor import CommandsInterceptor
from statgpt.utils.message_interceptors.system_msg_interceptor import SystemMessageInterceptor


def dial_tool_call_to_langchain_tool_call(tool_call: DialToolCall) -> LangChainToolCall:
    return LangChainToolCall(
        id=tool_call.id,
        name=tool_call.function.name,
        args=json.loads(tool_call.function.arguments),
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
        interceptors = [
            CommandsInterceptor.create_default(),
            SystemMessageInterceptor(data_service=data_service),
        ]
        for interceptor in interceptors:
            messages = await interceptor.process_messages(messages=messages, state=state)
        return cls(messages=messages)

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
                raise ValueError("User message content is empty")
            return HumanMessage(content=usr_msg_content)
        elif msg.role == Role.ASSISTANT:
            return AIMessage(
                content=msg.content or '',
                tool_calls=(
                    [dial_tool_call_to_langchain_tool_call(t) for t in msg.tool_calls]
                    if msg.tool_calls
                    else []
                ),
            )
        elif msg.role == Role.TOOL:
            msg_content = msg.content if msg.content else ''
            return ToolMessage(content=msg_content, tool_call_id=msg.tool_call_id)
        elif msg.role == Role.SYSTEM:
            msg_content = msg.content if msg.content else ''
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
                    raise RuntimeError(f"Unknown tool message type: {msg_type!r}")
        return tool_messages
