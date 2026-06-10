"""Unit tests for HybridSearcher.HybridMatch candidate assembly."""

from unittest.mock import Mock

import pytest

from statgpt.app.services.hybrid_searcher import HarmonizedItemScored, HybridSearcher
from statgpt.common.hybrid_indexer.schemas import IndicatorIndex


def _indicator(id_: str, dataset_id: str = "ds1") -> IndicatorIndex:
    return IndicatorIndex(
        id=id_,
        dataset_id=dataset_id,
        dataset_name="Dataset One",
        version_id=1,
        series="[]",
        name=f"name {id_}",
        name_normalized=f"name {id_}",
        where=[],
        primary=f"primary {id_}",
        primary_normalized=f"primary {id_}",
    )


def _candidate(id_: str, dataset_id: str = "ds1") -> dict:
    """A candidate dict in the shape produced by hybrid diversification."""
    return {"id": id_, "metadata": _indicator(id_, dataset_id).model_dump()}


def _lex(id_: str, score: float, dataset_id: str = "ds1") -> HarmonizedItemScored:
    return HarmonizedItemScored(score=score, metadata=_indicator(id_, dataset_id))


@pytest.fixture
def match() -> HybridSearcher.HybridMatch:
    return HybridSearcher.HybridMatch(outer=Mock())


def test_disabled_returns_candidates_unchanged(match):
    candidates = [_candidate("a"), _candidate("b")]
    lex_filtered = {"c": _lex("c", 0.9)}

    result = match._include_lexical_only_candidates(candidates, lex_filtered, max_lexical_only=0)

    assert [c["id"] for c in result] == ["a", "b"]


def test_appends_top_lexical_only_sorted_by_score(match):
    candidates = [_candidate("a")]
    lex_filtered = {
        "x": _lex("x", 0.2),
        "y": _lex("y", 0.9),
        "z": _lex("z", 0.5),
    }

    result = match._include_lexical_only_candidates(candidates, lex_filtered, max_lexical_only=2)

    # existing candidate kept; top-2 lexical-only added in descending score order
    assert [c["id"] for c in result] == ["a", "y", "z"]


def test_skips_ids_already_present(match):
    candidates = [_candidate("a"), _candidate("b")]
    lex_filtered = {
        "a": _lex("a", 0.99),  # already a candidate -> must not be duplicated
        "c": _lex("c", 0.7),
    }

    result = match._include_lexical_only_candidates(candidates, lex_filtered, max_lexical_only=5)

    assert [c["id"] for c in result] == ["a", "b", "c"]
    assert [c["id"] for c in result].count("a") == 1


def test_cap_limits_number_added(match):
    candidates: list[dict] = []
    lex_filtered = {str(i): _lex(str(i), float(i)) for i in range(10)}

    result = match._include_lexical_only_candidates(candidates, lex_filtered, max_lexical_only=3)

    # highest scores first: 9, 8, 7
    assert [c["id"] for c in result] == ["9", "8", "7"]


def test_appended_candidate_has_consumable_shape(match):
    """Appended entries must match the {id, metadata-dict} shape _prepare_for_relevance expects."""
    candidates: list[dict] = []
    lex_filtered = {"c": _lex("c", 0.7)}

    result = match._include_lexical_only_candidates(candidates, lex_filtered, max_lexical_only=1)

    appended = result[0]
    assert appended["id"] == "c"
    metadata = appended["metadata"]
    for key in ("dataset_id", "primary_normalized", "name_normalized", "where", "series"):
        assert key in metadata
