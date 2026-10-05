"""The Web Search Agent shows its answer in the tool stage once (#730).

Its producer streams the answer into the stage as it arrives, so the tool must not append it
again: the second copy would differ from the first in its citation ids, since only the response
handed to the agent is renumbered.
"""

import asyncio
import itertools
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
from aidial_sdk.chat_completion import Choice
from openai import APIError
from openai.types.chat import ChatCompletionChunk

from statgpt.app.chains.tools import StatGptTool
from statgpt.app.chains.web_search import response_producer
from statgpt.app.config import ChainParametersConfig
from statgpt.app.utils.citation_ids import CitationIdSpace
from statgpt.common.schemas import ChannelConfig, WebSearchAgentTool
from statgpt.common.schemas.tool_details import WebSearchAgentDetails

_ANSWER = 'GDP rose <cit data-id="web-1"></cit>.'
_ANNOTATION = {
    "index": 0,
    "target": {"selector": {"type": "html_tag", "tag": "cit", "id": "web-1"}},
    "body": {"title": "example.com"},
}


class FakeStage:
    def __init__(self) -> None:
        self.content = ""

    def append_content(self, content: str) -> None:
        self.content += content

    def add_attachment(self, **kwargs: Any) -> None:
        pass


def _chunk(content: str | None = None, **custom_content: Any) -> ChatCompletionChunk:
    delta: dict[str, Any] = {"content": content}
    if custom_content:
        delta["custom_content"] = custom_content
    return ChatCompletionChunk.model_validate(
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "web-rag",
            "choices": [{"index": 0, "finish_reason": None, "delta": delta}],
        }
    )


async def _stream(chunks: list[ChatCompletionChunk], error: Exception | None = None):
    for chunk in chunks:
        yield chunk
    if error is not None:
        raise error


def _tool() -> StatGptTool:
    channel_config = ChannelConfig.model_validate(
        {"supreme_agent": {"name": "Test Bot", "domain": "test", "terminology_domain": "test"}}
    )
    tool_config = WebSearchAgentTool(
        name="web_search_agent",
        description="Web search agent tool",
        details=WebSearchAgentDetails(deployment_id="dep", urls_only=False),
    )
    return StatGptTool.from_config(tool_config=tool_config, channel_config=channel_config)


async def _run(monkeypatch, stream) -> tuple[str, FakeStage]:
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=stream)
    monkeypatch.setattr(response_producer.openai, "get_async_client", lambda **_: client)

    target = FakeStage()
    choice = Choice(asyncio.Queue(), 0)
    choice.open()
    inputs = {
        ChainParametersConfig.AUTH_CONTEXT: Mock(),
        ChainParametersConfig.TARGET: target,
        ChainParametersConfig.CHOICE: choice,
        ChainParametersConfig.STATE: {},
        ChainParametersConfig.ANNOTATION_INDEX_SPACE: itertools.count(),
        ChainParametersConfig.CITATION_ID_SPACE: CitationIdSpace(),
    }
    response, _ = await _tool()._arun(inputs, query="US GDP")  # type: ignore[attr-defined]
    return response, target


async def test_the_stage_shows_the_streamed_answer_once(monkeypatch):
    response, target = await _run(
        monkeypatch, _stream([_chunk(_ANSWER), _chunk(annotations=[_ANNOTATION])])
    )

    assert target.content == _ANSWER
    assert response == 'GDP rose <cit data-id="citation001"></cit>.'


async def test_a_failed_stream_still_shows_the_error_in_the_stage(monkeypatch):
    error = APIError("boom", httpx.Request("POST", "https://dial"), body=None)

    response, target = await _run(monkeypatch, _stream([_chunk("GDP rose")], error))

    assert response == "<error>Something went wrong</error>"
    assert target.content == "GDP rose<error>Something went wrong</error>"
