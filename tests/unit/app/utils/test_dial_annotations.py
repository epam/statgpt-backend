"""Relaying `custom_content.annotations` from a sub-deployment to the user's message (#659).

A sub-deployment such as Deep Research writes an empty `<cit data-id="...">` tag at each cited
position and describes the tags in `custom_content.annotations`. The streamer forwards that array
so the client can turn each claimed tag into a citation pill instead of showing the raw markup.
"""

import asyncio
import copy
import gc
import itertools
import weakref
from typing import Any
from unittest.mock import Mock

from aidial_sdk.chat_completion import Choice
from aidial_sdk.chat_completion.chunks import BaseChunk
from openai.types.chat import ChatCompletionChunk

from statgpt.app.utils.citation_ids import Citation, CitationIdSpace, with_tag_id
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
    """The citation id and the relayed index of every annotation on the choice, in the order sent.

    Both are handed out in the order the annotations of the response first arrive, whichever
    sub-deployment sent them.
    """
    return [
        (annotation["target"]["selector"]["id"], annotation["index"])
        for annotation in _sent_annotations(choice)
    ]


def _streamer(
    choice: Choice,
    index_space: AnnotationIndexSpace | None = None,
    target: Any = None,
    citation_id_space: CitationIdSpace | None = None,
) -> OpenAiToDialStreamer:
    """A streamer on `choice`. Pass `index_space` and `citation_id_space` to share them between
    streamers."""
    return OpenAiToDialStreamer(
        target if target is not None else choice,
        choice,
        deployment="deep-research",
        show_debug_stages=False,
        stages_config=Mock(debug_only=True),
        annotation_index_space=index_space if index_space is not None else itertools.count(),
        citation_id_space=citation_id_space if citation_id_space is not None else CitationIdSpace(),
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
    def test_relays_every_field_but_the_tag_id_unchanged(self):
        choice = _open_choice()
        annotation = _annotation(0)
        _streamer(choice)._process_custom_content({"annotations": [annotation]})

        assert _sent_annotations(choice) == [with_tag_id(annotation, "citation001")]

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

        assert _sent_annotations(choice) == [with_tag_id(_annotation(0), "citation001")]

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
        citation_id_space = CitationIdSpace()
        choice = _open_choice()

        _streamer(choice, index_space, citation_id_space=citation_id_space)._process_custom_content(
            {"annotations": [_annotation(0, tag_id="dr-1"), _annotation(1, tag_id="dr-2")]}
        )
        _streamer(choice, index_space, citation_id_space=citation_id_space)._process_custom_content(
            {"annotations": [_annotation(0, tag_id="rag-1")]}
        )

        assert _relayed(choice) == [("citation001", 0), ("citation002", 1), ("citation003", 2)]

    def test_interleaved_streamers_produce_non_overlapping_indexes(self):
        """The two tool calls run concurrently, so their deltas reach the shared choice
        interleaved. Every annotation of the response must still end up with its own index."""
        index_space = itertools.count()
        citation_id_space = CitationIdSpace()
        choice = _open_choice()
        deep_research = _streamer(choice, index_space, citation_id_space=citation_id_space)
        rag = _streamer(choice, index_space, citation_id_space=citation_id_space)

        deep_research._process_custom_content({"annotations": [_annotation(0, tag_id="dr-1")]})
        rag._process_custom_content({"annotations": [_annotation(0, tag_id="rag-1")]})
        deep_research._process_custom_content({"annotations": [_annotation(1, tag_id="dr-2")]})
        rag._process_custom_content({"annotations": [_annotation(1, tag_id="rag-2")]})

        relayed = _relayed(choice)
        assert relayed == [
            ("citation001", 0),
            ("citation002", 1),
            ("citation003", 2),
            ("citation004", 3),
        ]
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

        assert _sent_annotations(choice) == [with_tag_id(_annotation(None), "citation001")]
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


def _titled(annotation: dict[str, Any], title: str | None) -> dict[str, Any]:
    """The annotation with the body `title` replaced, or removed if `title` is None."""
    body = {key: value for key, value in annotation["body"].items() if key != "title"}
    if title is not None:
        body["title"] = title
    return {**annotation, "body": body}


def _tag_ids(choice: Choice) -> list[str]:
    return [annotation["target"]["selector"]["id"] for annotation in _sent_annotations(choice)]


class TestCitationIds:
    def test_renumbers_the_tag_in_the_annotation_and_in_the_content_alike(self):
        """The agent copies the tags of a tool response into its answer, and the client resolves
        each tag against the relayed annotations by id, so both must carry the same id."""
        choice = _open_choice()
        streamer = _streamer(choice)
        streamer.send_chunk(_chunk([_annotation(0)]))

        assert _tag_ids(choice) == ["citation001"]
        assert streamer.content == 'GDP rose <cit data-id="citation001"></cit>.'

    def test_verbatim_content_keeps_the_ids_of_the_sub_deployment(self):
        streamer = _streamer(_open_choice())
        streamer.send_chunk(_chunk([_annotation(0)]))

        assert streamer.verbatim_content == 'GDP rose <cit data-id="cit-1"></cit>.'

    def test_content_streamed_before_its_annotation_is_renumbered_too(self):
        streamer = _streamer(_open_choice())
        streamer._process_content('GDP rose <cit data-id="abc"></cit>.')
        streamer._process_custom_content({"annotations": [_annotation(0, tag_id="abc")]})

        assert streamer.content == 'GDP rose <cit data-id="citation001"></cit>.'

    def test_one_tag_keeps_one_id(self):
        """A re-sent annotation, or another source cited by the same tag, still claims the tag
        the first annotation claimed."""
        choice = _open_choice()
        streamer = _streamer(choice)
        streamer._process_custom_content({"annotations": [_annotation(0), _annotation(1)]})
        streamer._process_custom_content({"annotations": [_annotation(0)]})

        assert _tag_ids(choice) == ["citation001"] * 3

    def test_two_streamers_reusing_a_tag_id_cite_with_distinct_ids(self):
        """A tool may start its tag ids over on every call (`cit-1`, `cit-2`, ...); the ids of the
        conversation must not, or one id would name the sources of two tool responses."""
        citation_id_space = CitationIdSpace()
        choice = _open_choice()
        first = _streamer(choice, citation_id_space=citation_id_space)
        second = _streamer(choice, citation_id_space=citation_id_space)
        first.send_chunk(_chunk([_annotation(0)]))
        second.send_chunk(_chunk([_annotation(0)]))

        assert _tag_ids(choice) == ["citation001", "citation002"]
        assert first.content == 'GDP rose <cit data-id="citation001"></cit>.'
        assert second.content == 'GDP rose <cit data-id="citation002"></cit>.'
        assert citation_id_space.count == 2

    def test_continues_the_numbering_of_the_earlier_turns(self):
        choice = _open_choice()
        _streamer(choice, citation_id_space=CitationIdSpace(7)).send_chunk(_chunk([_annotation(0)]))

        assert _tag_ids(choice) == ["citation008"]

    def test_a_tag_no_annotation_claims_keeps_its_id(self):
        streamer = _streamer(_open_choice())
        streamer._process_content(
            'GDP <cit data-id="cit-1"></cit> rose <cit data-id="cit-9"></cit>'
        )
        streamer._process_custom_content({"annotations": [_annotation(0)]})

        assert streamer.content == (
            'GDP <cit data-id="citation001"></cit> rose <cit data-id="cit-9"></cit>'
        )

    def test_an_annotation_claiming_no_tag_is_relayed_unchanged(self):
        citation_id_space = CitationIdSpace()
        choice = _open_choice()
        annotation = {**_annotation(0), "target": {"selector": {"type": "text_quote"}}}
        _streamer(choice, citation_id_space=citation_id_space)._process_custom_content(
            {"annotations": [annotation]}
        )

        assert _sent_annotations(choice) == [annotation]
        assert citation_id_space.count == 0, "an annotation claiming no tag consumes no id"

    def test_the_received_annotation_is_not_mutated(self):
        annotation = _annotation(0)
        original = copy.deepcopy(annotation)
        _streamer(_open_choice())._process_custom_content({"annotations": [annotation]})

        assert annotation == original

    def test_warns_of_a_claimed_tag_missing_from_the_content(self, caplog):
        """Its annotation goes out with the new id while the text keeps the old one, so the
        client shows the tag as literal text: worth a line in the log."""
        streamer = _streamer(_open_choice())
        with streamer:
            streamer._process_content('GDP rose <cit data-id=cit-1></cit>.')
            streamer._process_custom_content({"annotations": [_annotation(0)]})

        assert "not in its content" in caplog.text
        assert "cit-1" in caplog.text

    def test_does_not_warn_when_every_claimed_tag_is_renumbered(self, caplog):
        streamer = _streamer(_open_choice())
        with streamer:
            streamer.send_chunk(_chunk([_annotation(0)]))

        assert "not in its content" not in caplog.text


class TestCitations:
    def test_list_each_source_of_a_tag_once(self):
        streamer = _streamer(_open_choice())
        streamer._process_custom_content(
            {
                "annotations": [
                    _titled(_annotation(0), "Report, page 3"),
                    _titled(_annotation(1), "Review, page 17"),
                    _titled(_annotation(0), "Report, page 3"),
                ]
            }
        )

        assert streamer.citations == [
            Citation(id="citation001", title="Report, page 3"),
            Citation(id="citation001", title="Review, page 17"),
        ]

    def test_an_annotation_without_a_title_is_named_by_the_file_it_cites(self):
        streamer = _streamer(_open_choice())
        streamer._process_custom_content({"annotations": [_titled(_annotation(0), None)]})

        assert streamer.citations == [Citation(id="citation001", title="Report.pdf")]

    def test_a_tag_whose_annotations_name_no_source_is_listed_without_a_title(self):
        """The agent still learns the tag cites something, rather than meeting an id it cannot
        look up."""
        streamer = _streamer(_open_choice())
        streamer._process_custom_content({"annotations": [{**_annotation(0), "body": {}}]})

        assert streamer.citations == [Citation(id="citation001", title=None)]

    def test_a_resent_annotation_naming_no_source_adds_no_untitled_entry(self):
        streamer = _streamer(_open_choice())
        streamer._process_custom_content(
            {"annotations": [_annotation(0), {**_annotation(0), "body": {"quote": "more"}}]}
        )

        assert streamer.citations == [Citation(id="citation001", title="Report.pdf, page 3")]

    def test_the_note_lists_the_id_and_the_title_of_every_citation(self):
        streamer = _streamer(_open_choice())
        streamer.send_chunk(_chunk([_titled(_annotation(0), "sigma 2/2025 – World insurance")]))

        assert streamer.citations_note == (
            '\n\n### Sources of the citation tags:\n\n```json\n'
            '[{"id": "citation001", "title": "sigma 2/2025 – World insurance"}]\n```'
        )

    def test_the_note_is_empty_without_citations(self):
        streamer = _streamer(_open_choice())
        streamer._process_content("GDP rose.")

        assert streamer.citations_note == ""
