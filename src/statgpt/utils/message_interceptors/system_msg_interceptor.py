import logging
import typing as t

from aidial_sdk.chat_completion import Role
from aidial_sdk.exceptions import InvalidRequestError
from pydantic import ValidationError

from common.data.sdmx.v21.query import JsonQuery
from common.schemas.dial import Message as DialMessage
from statgpt.services.chat_facade import ChannelServiceFacade

from .base import BaseMessageInterceptor

_log = logging.getLogger(__name__)


class SystemMessageInterceptor(BaseMessageInterceptor):
    def __init__(self, data_service: ChannelServiceFacade):
        self._data_service = data_service

    async def process_messages(
        self, messages: list[DialMessage], state: dict[str, t.Any]
    ) -> list[DialMessage]:
        result = []

        for msg in messages:
            if msg.role != Role.SYSTEM:
                result.append(msg)
                continue
            content = msg.content
            if content is not None and len(content) > 0:
                raise InvalidRequestError(
                    "System messages with non-empty content are not supported when using StatGPT"
                )
            if msg.custom_content is None or not msg.custom_content.attachments:
                raise InvalidRequestError(
                    "System messages must have custom_content with attachments when using StatGPT"
                )
            attachments = msg.custom_content.attachments
            for attachment in attachments:
                if attachment.type != "application/json":
                    raise InvalidRequestError(
                        f"Unsupported system message attachment type: {attachment.type}. Only application/json is supported."
                    )
                if not attachment.data:
                    raise InvalidRequestError("System message attachment must have data field")
                try:
                    JsonQuery.model_validate_json(attachment.data)
                except ValidationError as e:
                    raise InvalidRequestError(
                        "Failed to parse system message attachment data as JsonQuery"
                    ) from e

            # result.append(msg)  # ToDo: finish implementation

        return result
