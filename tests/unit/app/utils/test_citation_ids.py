"""The citation tags of a sub-deployment's response and the sources they cite (#747)."""

from typing import Any

import pytest

from statgpt.app.utils.citation_ids import get_source_quote, get_source_title, get_tag_id


def _annotation(selector: dict[str, Any]) -> dict[str, Any]:
    return {"index": 0, "target": {"selector": selector}, "body": {"title": "Report, page 3"}}


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


class TestSourceTitle:
    _ATTACHMENT = {"title": "report.pdf", "url": "https://public/report.pdf"}

    def test_the_title_of_the_citation_card_comes_first(self):
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


class TestSourceQuote:
    def test_reads_the_quoted_passage(self):
        annotation = {"body": {"title": "Economic data", "quote": "* Country: United States"}}

        assert get_source_quote(annotation) == "* Country: United States"

    @pytest.mark.parametrize(
        "annotation",
        [
            {},
            {"body": None},
            {"body": {"title": "Report"}},
            {"body": {"quote": ""}},
            {"body": {"quote": 7}},
        ],
    )
    def test_an_annotation_quoting_nothing_has_no_quote(self, annotation: dict[str, Any]):
        assert get_source_quote(annotation) is None
