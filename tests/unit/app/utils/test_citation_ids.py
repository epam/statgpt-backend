"""The citation tag ids of a conversation (#730)."""

import re
from typing import Any

import pytest

from statgpt.app.utils.citation_ids import (
    CitationIdSpace,
    find_tag_ids,
    get_source_title,
    get_tag_id,
    replace_tag_ids,
    with_tag_id,
)


def _annotation(selector: dict[str, Any]) -> dict[str, Any]:
    return {"index": 0, "target": {"selector": selector}, "body": {"title": "Report, page 3"}}


class TestCitationIdSpace:
    def test_hands_out_short_numbered_ids(self):
        space = CitationIdSpace()

        assert [space.next_id(), space.next_id()] == ["citation001", "citation002"]
        assert space.count == 2

    def test_continues_from_the_count(self):
        space = CitationIdSpace(998)

        assert [space.next_id(), space.next_id()] == ["citation999", "citation1000"]

    def test_a_scope_is_appended_to_every_id(self):
        space = CitationIdSpace(scope="k3f9")

        assert [space.next_id(), space.next_id()] == ["citation001-k3f9", "citation002-k3f9"]

    def test_a_scoped_space_gets_a_short_random_scope(self):
        """For an MCP tool call, which cannot continue the ids of the calls before it."""
        assert re.fullmatch(r"citation001-[a-z0-9]{4}", CitationIdSpace.scoped().next_id())


class TestTagId:
    def test_reads_the_id_of_the_claimed_tag(self):
        annotation = _annotation({"type": "html_tag", "tag": "cit", "id": "abc"})

        assert get_tag_id(annotation) == "abc"

    @pytest.mark.parametrize(
        "annotation",
        [
            {"index": 0},
            _annotation({"type": "text_quote", "id": "abc"}),
            _annotation({"type": "html_tag", "tag": "cit"}),
            _annotation({"type": "html_tag", "tag": "cit", "id": 7}),
        ],
    )
    def test_an_annotation_claiming_no_tag_has_no_id(self, annotation: dict[str, Any]):
        assert get_tag_id(annotation) is None

    def test_with_tag_id_copies_the_annotation(self):
        annotation = _annotation({"type": "html_tag", "tag": "cit", "id": "abc"})

        renumbered = with_tag_id(annotation, "citation001")

        assert renumbered == _annotation({"type": "html_tag", "tag": "cit", "id": "citation001"})
        assert get_tag_id(annotation) == "abc"


class TestSourceTitle:
    _ATTACHMENT = {"title": "report.pdf", "url": "https://public/report.pdf"}

    def test_the_label_of_the_pill_comes_first(self):
        body = {"title": "Report, page 3", "source": {"attachment": self._ATTACHMENT}}

        assert get_source_title({"body": body}) == "Report, page 3"

    def test_falls_back_to_the_title_of_the_cited_file(self):
        body = {"title": "", "source": {"attachment": self._ATTACHMENT}}

        assert get_source_title({"body": body}) == "report.pdf"

    def test_then_to_the_url_of_the_cited_file(self):
        body = {"source": {"attachment": {"url": "https://public/report.pdf"}}}

        assert get_source_title({"body": body}) == "https://public/report.pdf"

    @pytest.mark.parametrize(
        "annotation",
        [{}, {"body": None}, {"body": {"source": {"attachment": None}}}, {"body": {"title": 7}}],
    )
    def test_an_annotation_naming_no_source_has_no_title(self, annotation: dict[str, Any]):
        assert get_source_title(annotation) is None


class TestReplaceTagIds:
    def test_replaces_the_mapped_ids_only(self):
        content = 'A <cit data-id="a"></cit>, B <cit data-id="b"></cit>, A <cit data-id="a"></cit>'

        assert replace_tag_ids(content, {"a": "citation001"}) == (
            'A <cit data-id="citation001"></cit>, B <cit data-id="b"></cit>,'
            ' A <cit data-id="citation001"></cit>'
        )

    def test_keeps_the_other_attributes_of_the_tag(self):
        content = '<cit class="pill" data-id="a" title="t"></cit>'

        assert replace_tag_ids(content, {"a": "citation001"}) == (
            '<cit class="pill" data-id="citation001" title="t"></cit>'
        )

    def test_leaves_text_outside_a_tag(self):
        content = 'the attribute data-id="a" names the tag'

        assert replace_tag_ids(content, {"a": "citation001"}) == content

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("<cit data-id='a'></cit>", "<cit data-id='citation001'></cit>"),
            ('<cit data-id = "a"></cit>', '<cit data-id = "citation001"></cit>'),
            ('<cit\ndata-id="a"></cit>', '<cit\ndata-id="citation001"></cit>'),
            ("""<cit data-id='a"b'></cit>""", """<cit data-id='a"b'></cit>"""),
        ],
    )
    def test_recognises_either_quote_and_whitespace_around_the_equals_sign(
        self, content: str, expected: str
    ):
        assert replace_tag_ids(content, {"a": "citation001"}) == expected


def test_find_tag_ids_reads_the_tags_replace_tag_ids_recognises():
    content = (
        '<cit data-id="a"></cit> <cit data-id=\'b\'></cit> <cit data-id = "c"></cit> data-id="d"'
    )

    assert find_tag_ids(content) == {"a", "b", "c"}
