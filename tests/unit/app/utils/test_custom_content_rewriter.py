"""Rewriting the annotations and attachments relayed from Deep Research."""

import asyncio
import copy
import itertools
from typing import Any
from unittest.mock import Mock

from aidial_sdk.chat_completion import Choice
from aidial_sdk.chat_completion.chunks import BaseChunk
from openai.types.chat import ChatCompletionChunk

from statgpt.app.utils import CustomContentRewriter, OpenAiToDialStreamer
from statgpt.app.utils.citation_ids import CitationIdSpace
from statgpt.common.schemas import CustomContentRewriteRule

_DIAL_URL = "files/bucket/appdata/rag/sigma/Report%20E.pdf"
_PUBLIC_URL = "https://public/sigma/Report%20E.pdf"
_TO_PUBLIC = {
    "field": "url",
    "pattern": r"^files/.+/([^/]+)$",
    "replacement": r"https://public/sigma/\1",
}


def _annotation(url: str = _DIAL_URL, mime: str = "application/pdf") -> dict[str, Any]:
    return {
        "index": 0,
        "target": {"selector": {"type": "html_tag", "tag": "cit", "id": "cit-1"}},
        "body": {
            "title": "Report, page 3",
            "source": {
                "type": "attachment",
                "attachment": {"type": mime, "url": url, "title": "Report"},
            },
        },
    }


def _attachment(**fields: Any) -> dict[str, Any]:
    return {"type": "application/pdf", "title": "Report", "url": _DIAL_URL, **fields}


def _rewriter(*rules: dict[str, Any]) -> CustomContentRewriter:
    return CustomContentRewriter([CustomContentRewriteRule.model_validate(r) for r in rules])


def _url(annotation: dict[str, Any]) -> str:
    return annotation["body"]["source"]["attachment"]["url"]


def test_annotation_url_rewritten_with_backreference() -> None:
    rewriter = _rewriter(
        {"selector": {"applies_to": "annotation", "url": "^files/"}, "rewrites": [_TO_PUBLIC]}
    )
    assert _url(rewriter.rewrite_annotation(_annotation())) == _PUBLIC_URL


def test_selector_conditions_are_and_ed() -> None:
    rewriter = _rewriter(
        {
            "selector": {"applies_to": "both", "type": "pdf", "url": "^files/"},
            "rewrites": [_TO_PUBLIC],
        }
    )
    assert _url(rewriter.rewrite_annotation(_annotation(mime="text/html"))) == _DIAL_URL
    assert _url(rewriter.rewrite_annotation(_annotation())) == _PUBLIC_URL


def test_selector_matches_both_titles() -> None:
    rewriter = _rewriter(
        {
            "selector": {
                "applies_to": "annotation",
                "attachment_title": "^Report$",
                "body_title": "page 3",
            },
            "rewrites": [_TO_PUBLIC],
        }
    )
    assert _url(rewriter.rewrite_annotation(_annotation())) == _PUBLIC_URL


def test_first_matching_rule_wins() -> None:
    rewriter = _rewriter(
        {"selector": {"applies_to": "both", "url": "^files/"}, "rewrites": [_TO_PUBLIC]},
        {
            "selector": {"applies_to": "both", "url": "^files/"},
            "rewrites": [{**_TO_PUBLIC, "replacement": "other"}],
        },
    )
    assert rewriter.rewrite_attachment(_attachment())["url"] == _PUBLIC_URL


def test_rule_skipped_for_other_kind() -> None:
    rewriter = _rewriter(
        {"selector": {"applies_to": "attachment", "url": "^files/"}, "rewrites": [_TO_PUBLIC]}
    )
    assert _url(rewriter.rewrite_annotation(_annotation())) == _DIAL_URL


def test_missing_selector_field_does_not_match() -> None:
    rewriter = _rewriter(
        {"selector": {"applies_to": "annotation", "url": "^files/"}, "rewrites": [_TO_PUBLIC]}
    )
    annotation = _annotation()
    del annotation["body"]["source"]
    assert rewriter.rewrite_annotation(annotation) is annotation


def test_reference_url_selected_and_rewritten_separately() -> None:
    rewriter = _rewriter(
        {
            "selector": {"applies_to": "attachment", "reference_url": "^files/"},
            "rewrites": [{**_TO_PUBLIC, "field": "reference_url"}],
        }
    )
    attachment = _attachment(url=None, reference_url=_DIAL_URL)
    assert rewriter.rewrite_attachment(attachment) == {**attachment, "reference_url": _PUBLIC_URL}
    # The selector reads `reference_url` only: an attachment with just a `url` is left alone.
    assert rewriter.rewrite_attachment(_attachment())["url"] == _DIAL_URL


def test_missing_target_field_skipped() -> None:
    rewriter = _rewriter(
        {
            "selector": {"applies_to": "attachment", "type": "pdf"},
            "rewrites": [{**_TO_PUBLIC, "field": "reference_url"}, _TO_PUBLIC],
        }
    )
    result = rewriter.rewrite_attachment(_attachment())
    assert result["url"] == _PUBLIC_URL
    assert "reference_url" not in result


def test_input_not_mutated() -> None:
    rewriter = _rewriter(
        {"selector": {"applies_to": "both", "url": "^files/"}, "rewrites": [_TO_PUBLIC]}
    )
    annotation = _annotation()
    original = copy.deepcopy(annotation)
    rewriter.rewrite_annotation(annotation)
    assert annotation == original


def _chunk(custom_content: dict[str, Any]) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "deep-research",
            "choices": [
                {"index": 0, "finish_reason": None, "delta": {"custom_content": custom_content}}
            ],
        }
    )


def _sent_custom_content(choice: Choice) -> list[dict[str, Any]]:
    """The `custom_content` of every chunk the choice's queue holds, in the order sent."""
    sent = []
    while not choice._queue.empty():
        queued = choice._queue.get_nowait()
        if isinstance(queued, BaseChunk):
            for ch in queued.to_dict().get("choices", []):
                if custom_content := ch.get("delta", {}).get("custom_content"):
                    sent.append(custom_content)
    return sent


def test_streamer_rewrites_annotations_report_and_stage_attachments() -> None:
    choice = Choice(asyncio.Queue(), 0)
    choice.open()
    rewriter = _rewriter(
        {"selector": {"applies_to": "both", "url": "^files/"}, "rewrites": [_TO_PUBLIC]}
    )
    streamer = OpenAiToDialStreamer(
        choice,
        choice,
        deployment="deep-research",
        show_debug_stages=True,
        stages_config=Mock(debug_only=False),
        annotation_index_space=itertools.count(5),
        citation_id_space=CitationIdSpace(),
        stream_content=False,
        rewriter=rewriter,
    )

    with streamer:
        streamer.send_chunk(
            _chunk(
                {
                    "annotations": [_annotation()],
                    "attachments": [_attachment()],
                    "stages": [{"index": 0, "name": "Search", "attachments": [_attachment()]}],
                }
            )
        )

    sent = _sent_custom_content(choice)
    [annotation] = [a for cc in sent for a in cc.get("annotations", [])]
    assert _url(annotation) == _PUBLIC_URL
    assert annotation["index"] == 5
    assert streamer.annotations == [annotation]
    assert streamer.attachments[0]["url"] == _PUBLIC_URL
    [stage_attachment] = [
        a for cc in sent for stage in cc.get("stages", []) for a in stage.get("attachments", [])
    ]
    assert stage_attachment["url"] == _PUBLIC_URL
