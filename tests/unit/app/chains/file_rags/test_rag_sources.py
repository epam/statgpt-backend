"""A RAG answer is accepted when it cites at least one source, as an attachment or as an
annotation. Otherwise the tool response is replaced with "unable to find the relevant data".

The tool response names the source of every citation tag of the answer, so the agent can tell
which publication backs a statement (#730).
"""

import asyncio
import itertools
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from aidial_sdk.chat_completion import Choice
from aidial_sdk.chat_completion.chunks import BaseChunk
from openai.types.chat import ChatCompletionChunk

from statgpt.app.chains.file_rags.dial_rag import DialRagAgentFactory
from statgpt.app.chains.file_rags.file_rag_tool import _RAG_IMPLEMENTATIONS
from statgpt.app.config import ChainParametersConfig
from statgpt.app.utils.citation_ids import CitationIdSpace
from statgpt.common.schemas import FileRagTool as FileRagToolConfig
from statgpt.common.schemas import RAGVersion
from statgpt.common.schemas.tool_details import FileRagDetails

_DIAL_URL = "files/bucket/appdata/statgpt-generic-rag/sigma/report.pdf"
_PUBLIC_URL = "https://public/report.pdf"
_REWRITE_RULES = [
    {
        "selector": {"applies_to": "both", "url": "^files/"},
        "rewrites": [{"field": "url", "pattern": "^files/.+/", "replacement": "https://public/"}],
    }
]
_ANSWER = 'US GDP growth slows to 1.5%. <cit data-id="94c80f43b36f40ffbd9063db0cdc35c6"></cit>'
_RELAYED_ANSWER = 'US GDP growth slows to 1.5%. <cit data-id="citation004"></cit>'
_ANNOTATION = {
    "index": 0,
    "target": {
        "selector": {"type": "html_tag", "tag": "cit", "id": "94c80f43b36f40ffbd9063db0cdc35c6"}
    },
    "body": {
        "title": "sigma 2/2025 – World insurance, page 8",
        "quote": "Real GDP growth in the US is expected to slow to 1.5% in 2025.",
        "source": {
            "type": "attachment",
            "attachment": {"type": "application/pdf", "title": "report.pdf", "url": _DIAL_URL},
        },
    },
}
_ATTACHMENT = {"type": "application/pdf", "title": "report.pdf", "url": _DIAL_URL}


class FakeStage:
    def __init__(self) -> None:
        self.content = ""
        self.attachments: list[dict[str, Any]] = []

    def append_content(self, content: str) -> None:
        self.content += content

    def add_attachment(self, **kwargs: Any) -> None:
        self.attachments.append(kwargs)


def _chunk(content: str | None = None, **custom_content: Any) -> ChatCompletionChunk:
    delta: dict[str, Any] = {"content": content}
    if custom_content:
        delta["custom_content"] = custom_content
    return ChatCompletionChunk.model_validate(
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "generic-rag",
            "choices": [{"index": 0, "finish_reason": None, "delta": delta}],
        }
    )


async def _stream(chunks: list[ChatCompletionChunk]):
    for chunk in chunks:
        yield chunk


def _factory() -> DialRagAgentFactory:
    tool_config = FileRagToolConfig(
        name="File_RAG",
        description="publications",
        details=FileRagDetails.model_validate(
            {"version": RAGVersion.GENERIC, "rewrite_rules": _REWRITE_RULES}
        ),
    )
    return _RAG_IMPLEMENTATIONS[RAGVersion.GENERIC](tool_config, channel_config=None)  # type: ignore[arg-type]


def _sent_annotations(choice: Choice) -> list[dict[str, Any]]:
    sent = []
    while not choice._queue.empty():
        queued = choice._queue.get_nowait()
        if isinstance(queued, BaseChunk):
            for ch in queued.to_dict().get("choices", []):
                sent.extend(ch.get("delta", {}).get("custom_content", {}).get("annotations", []))
    return sent


async def _run(
    chunks: list[ChatCompletionChunk],
) -> tuple[dict[str, Any], FakeStage, Choice]:
    factory = _factory()
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=_stream(chunks))
    factory._init_dial_rag_client = Mock(return_value=client)  # type: ignore[method-assign]

    target = FakeStage()
    choice = Choice(asyncio.Queue(), 0)
    choice.open()
    inputs = {
        ChainParametersConfig.AUTH_CONTEXT: Mock(),
        ChainParametersConfig.TARGET: target,
        ChainParametersConfig.CHOICE: choice,
        ChainParametersConfig.QUERY: "US economic outlook",
        ChainParametersConfig.TARGET_PREFILTER: None,
        ChainParametersConfig.SEARCH_ALL_PUBLICATIONS: True,
        ChainParametersConfig.STATE: {},
        ChainParametersConfig.ANNOTATION_INDEX_SPACE: itertools.count(),
        # three citations were handed out by the earlier turns of the conversation
        ChainParametersConfig.CITATION_ID_SPACE: CitationIdSpace(3),
    }
    result = await factory._stream_response(inputs)
    return result, target, choice


@pytest.mark.asyncio
async def test_annotations_only_answer_is_accepted_and_rewritten():
    result, target, choice = await _run([_chunk(_ANSWER, annotations=[_ANNOTATION])])

    assert result[DialRagAgentFactory.FIELD_ANSWERED_BY] == 'RAG'
    assert result[DialRagAgentFactory.FIELD_RESPONSE].startswith(_RELAYED_ANSWER)
    assert _RELAYED_ANSWER in target.content
    [annotation] = _sent_annotations(choice)
    assert annotation["body"]["source"]["attachment"]["url"] == _PUBLIC_URL
    assert annotation["target"]["selector"]["id"] == "citation004"


@pytest.mark.asyncio
async def test_response_names_the_source_of_every_citation_tag():
    result, target, choice = await _run([_chunk(_ANSWER, annotations=[_ANNOTATION])])

    response = result[DialRagAgentFactory.FIELD_RESPONSE]
    assert (
        '### Sources of the citation tags:\n\n```json\n'
        '[{"id": "citation004", "title": "sigma 2/2025 – World insurance, page 8"}]\n```'
    ) in response
    assert _ANNOTATION["body"]["quote"] not in response
    assert "Sources of the citation tags" not in target.content
    # the client still receives the whole annotation, quote included
    [annotation] = _sent_annotations(choice)
    assert annotation["body"]["quote"] == _ANNOTATION["body"]["quote"]


@pytest.mark.asyncio
async def test_attachments_only_answer_is_accepted_and_rewritten():
    result, target, _ = await _run([_chunk("Answer"), _chunk(attachments=[_ATTACHMENT])])

    assert result[DialRagAgentFactory.FIELD_ANSWERED_BY] == 'RAG'
    [attachment] = [a for a in target.attachments if a["title"] == "report.pdf"]
    assert attachment["url"] == _PUBLIC_URL
    assert "Sources of the citation tags" not in result[DialRagAgentFactory.FIELD_RESPONSE]


@pytest.mark.asyncio
async def test_answer_without_sources_is_replaced():
    result, target, choice = await _run([_chunk("I don't know.")])

    assert result[DialRagAgentFactory.FIELD_ANSWERED_BY] == 'LLM'
    assert "unable to find the relevant data" in result[DialRagAgentFactory.FIELD_RESPONSE]
    assert "I don't know." not in target.content
    assert _sent_annotations(choice) == []
