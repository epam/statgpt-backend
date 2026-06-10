import json
from unittest.mock import MagicMock

import pytest
from aidial_sdk.chat_completion import Attachment, CustomContent
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role
from aidial_sdk.exceptions import InvalidRequestError

from statgpt.app.utils.message_interceptors.system_msg_interceptor import SystemMessageInterceptor
from statgpt.common.utils.media_types import MediaTypes


def _system_message_with_json_attachment(data: dict) -> DialMessage:
    return DialMessage(
        role=Role.SYSTEM,
        content="",
        custom_content=CustomContent(
            attachments=[Attachment(type=MediaTypes.JSON, data=json.dumps(data))]
        ),
    )


def _json_query_body(disabled: bool | str | None = None) -> dict:
    body: dict = {
        "urn": "IMF:WEO(1.0)",
        "filters": [
            {"componentCode": "COUNTRY", "operator": "in", "values": ["USA"]},
        ],
    }
    if disabled is not None:
        body["disabled"] = disabled
    return body


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", [None, True, False])
async def test_system_message_json_query_attachment_passes(disabled: bool | None) -> None:
    interceptor = SystemMessageInterceptor(data_service=MagicMock())
    messages = [_system_message_with_json_attachment(_json_query_body(disabled))]

    result = await interceptor.process_messages(messages=messages, state={})

    assert result == []  # system messages are validated and dropped


@pytest.mark.asyncio
async def test_system_message_json_query_attachment_rejects_non_bool_disabled() -> None:
    interceptor = SystemMessageInterceptor(data_service=MagicMock())
    messages = [_system_message_with_json_attachment(_json_query_body(disabled="banana"))]

    with pytest.raises(InvalidRequestError):
        await interceptor.process_messages(messages=messages, state={})


@pytest.mark.asyncio
async def test_system_message_invalid_json_query_attachment_rejected() -> None:
    interceptor = SystemMessageInterceptor(data_service=MagicMock())
    messages = [_system_message_with_json_attachment({"urn": "not-a-valid-urn"})]

    with pytest.raises(InvalidRequestError):
        await interceptor.process_messages(messages=messages, state={})
