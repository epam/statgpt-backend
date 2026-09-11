"""Relaying `custom_content.annotations` from a sub-deployment to the user's message (#659).

A sub-deployment such as Deep Research writes an empty `<cit data-id="...">` tag at each cited
position and describes the tags in `custom_content.annotations`. The streamer forwards that array
so the client can turn each claimed tag into a citation pill instead of showing the raw markup.
"""

import asyncio
import gc
import itertools
import weakref
from typing import Any
from unittest.mock import Mock

from aidial_sdk.chat_completion import Choice
from aidial_sdk.chat_completion.chunks import BaseChunk
from openai.types.chat import ChatCompletionChunk

from statgpt.app.utils.dial_annotations import send_annotations
from statgpt.app.utils.dial_stages import NullChoice
from statgpt.app.utils.openai_to_dial_streamer import AnnotationIndexSpace, OpenAiToDialStreamer


def _annotation(index: int | None, tag_id: str = "cit-1", page: int = 3) -> dict[str, Any]:
    """One entry of the annotations array, in the shape Deep Research sends."""
    annotation: dict[str, Any] = {
        "target": {"selector": {"type": "html_tag", "tag": "cit", "id": tag_id}},
        "body": {
            "title": "Report.pdf, page 3",
            "source": {
                "type": "attachment",
                "attachment": {
                    "type": "application/pdf",
                    "url": "files/bucket/Report.pdf",
                    "title": "Report.pdf",
                },
            },
            "selector": {"type": "pdf_bbox", "page": page, "x1": 0, "y1": 0, "x2": 0, "y2": 0},
        },
    }
    if index is not None:
        annotation["index"] = index
    return annotation


def _open_choice() -> Choice:
    """A real SDK choice whose queue the test reads the emitted chunks from."""
    choice = Choice(asyncio.Queue(), 0)
    choice.open()
    return choice


def _annotation_chunks(choice: Choice) -> list[dict[str, Any]]:
    """The chunks carrying annotations that the choice's queue holds, in the order sent.

    Chunks of every other kind — the one `open()` sends, content, stages — are skipped.
    """
    chunks = []
    while not choice._queue.empty():
        queued = choice._queue.get_nowait()
        # The queue also carries the end-of-stream and error markers, which have no payload.
        if not isinstance(queued, BaseChunk):
            continue
        chunk = queued.to_dict()
        for ch in chunk.get("choices", []):
            if "annotations" in ch.get("delta", {}).get("custom_content", {}):
                chunks.append(chunk)
    return chunks


def _sent_annotations(choice: Choice) -> list[dict[str, Any]]:
    """Every annotation the choice's queue carries, in the order it was sent."""
    return [
        annotation
        for chunk in _annotation_chunks(choice)
        for annotation in chunk["choices"][0]["delta"]["custom_content"]["annotations"]
    ]


def _relayed(choice: Choice) -> list[tuple[str, int]]:
    """The tag id and the relayed index of every annotation on the choice, in the order sent.

    The tag id says which sub-deployment an annotation came from, which is what tells the two
    relays apart once both have written to the one choice of the response.
    """
    return [
        (annotation["target"]["selector"]["id"], annotation["index"])
        for annotation in _sent_annotations(choice)
    ]


def _streamer(
    choice: Choice, index_space: AnnotationIndexSpace | None = None, target: Any = None
) -> OpenAiToDialStreamer:
    """A streamer on `choice`. Pass `index_space` to share one counter between streamers."""
    return OpenAiToDialStreamer(
        target if target is not None else choice,
        choice,
        deployment="deep-research",
        show_debug_stages=False,
        stages_config=Mock(debug_only=True),
        annotation_index_space=index_space if index_space is not None else itertools.count(),
    )


def _chunk(annotations: list[dict[str, Any]]) -> ChatCompletionChunk:
    """A completion chunk of the sub-deployment carrying an annotations array."""
    return ChatCompletionChunk.model_validate(
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "deep-research",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "role": "assistant",
                        "content": 'GDP rose <cit data-id="cit-1"></cit>.',
                        "custom_content": {"annotations": annotations},
                    },
                }
            ],
        }
    )


class TestSendAnnotations:
    def test_emits_one_delta_on_the_choice(self):
        choice = _open_choice()
        send_annotations(choice, [_annotation(0), _annotation(1, tag_id="cit-2")])

        chunks = _annotation_chunks(choice)
        assert len(chunks) == 1, "the array goes out as a single delta"
        assert chunks[0]["choices"][0]["index"] == choice.index
        assert chunks[0]["choices"][0]["delta"]["custom_content"]["annotations"] == [
            _annotation(0),
            _annotation(1, tag_id="cit-2"),
        ]

    def test_is_a_no_op_on_a_choice_that_cannot_stream(self, caplog):
        send_annotations(NullChoice(), [_annotation(0)])

        assert "cannot stream" in caplog.text


class TestStreamerRelaysAnnotations:
    def test_relays_every_field_unchanged(self):
        choice = _open_choice()
        annotation = _annotation(0)
        _streamer(choice)._process_custom_content({"annotations": [annotation]})

        assert _sent_annotations(choice) == [annotation]

    def test_relays_to_the_choice_and_not_to_the_target_stage(self):
        choice = _open_choice()
        stage = Mock()
        _streamer(choice, target=stage)._process_custom_content({"annotations": [_annotation(0)]})

        assert len(_sent_annotations(choice)) == 1
        stage.send_chunk.assert_not_called()

    def test_annotations_survive_the_openai_chunk_model(self):
        """`custom_content` is not part of the OpenAI schema; the whole array has to come
        through `ChatCompletionChunk` parsing for the relay to have anything to send."""
        choice = _open_choice()
        _streamer(choice).send_chunk(_chunk([_annotation(0)]))

        assert _sent_annotations(choice) == [_annotation(0)]

    def test_custom_content_without_annotations_sends_nothing(self):
        choice = _open_choice()
        _streamer(choice)._process_custom_content({"state": {"research_started": True}})

        assert _sent_annotations(choice) == []


class TestAnnotationIndexSpace:
    def test_two_streamers_do_not_reuse_each_other_indexes(self):
        """Two tool calls of one response write to the one choice of that response, and each
        sub-deployment numbers its own annotations from zero. Sharing one index space keeps the
        pills of one from swallowing the pills of the other."""
        index_space = itertools.count()
        choice = _open_choice()

        _streamer(choice, index_space)._process_custom_content(
            {"annotations": [_annotation(0, tag_id="dr-1"), _annotation(1, tag_id="dr-2")]}
        )
        _streamer(choice, index_space)._process_custom_content(
            {"annotations": [_annotation(0, tag_id="rag-1")]}
        )

        assert _relayed(choice) == [("dr-1", 0), ("dr-2", 1), ("rag-1", 2)]

    def test_interleaved_streamers_produce_non_overlapping_indexes(self):
        """The two tool calls run concurrently, so their deltas reach the shared choice
        interleaved. Every annotation of the response must still end up with its own index."""
        index_space = itertools.count()
        choice = _open_choice()
        deep_research = _streamer(choice, index_space)
        rag = _streamer(choice, index_space)

        deep_research._process_custom_content({"annotations": [_annotation(0, tag_id="dr-1")]})
        rag._process_custom_content({"annotations": [_annotation(0, tag_id="rag-1")]})
        deep_research._process_custom_content({"annotations": [_annotation(1, tag_id="dr-2")]})
        rag._process_custom_content({"annotations": [_annotation(1, tag_id="rag-2")]})

        relayed = _relayed(choice)
        assert relayed == [("dr-1", 0), ("rag-1", 1), ("dr-2", 2), ("rag-2", 3)]
        assert len({index for _, index in relayed}) == len(relayed), "indexes must not repeat"

    def test_one_streamer_keeps_an_index_stable_across_deltas(self):
        """The sub-deployment may re-send an annotation to extend it; the same incoming index
        must keep resolving to the same outgoing one, or the client would store two entries."""
        index_space = itertools.count()
        choice = _open_choice()
        streamer = _streamer(choice, index_space)

        streamer._process_custom_content({"annotations": [_annotation(0)]})
        streamer._process_custom_content({"annotations": [_annotation(1, tag_id="cit-2")]})
        streamer._process_custom_content({"annotations": [_annotation(0)]})

        assert [a["index"] for a in _sent_annotations(choice)] == [0, 1, 0]

    def test_an_index_less_annotation_stays_index_less(self):
        index_space = itertools.count()
        choice = _open_choice()
        _streamer(choice, index_space)._process_custom_content({"annotations": [_annotation(None)]})

        assert _sent_annotations(choice) == [_annotation(None)]
        assert next(index_space) == 0, "an index-less annotation consumes no index"

    def test_the_index_space_does_not_retain_a_streamer(self):
        """Why the index space is a counter rather than a map keyed by streamer: it lives for
        the whole response, and a streamer holds its sub-deployment's buffered report, its
        attachments and its stages. Keying on the streamer would pin all of that until the
        response ends, however many tool calls the turn makes."""
        index_space = itertools.count()
        choice = _open_choice()
        streamer = _streamer(choice, index_space)
        streamer._process_custom_content({"annotations": [_annotation(0)]})
        collected = weakref.ref(streamer)

        del streamer
        gc.collect()

        assert collected() is None
