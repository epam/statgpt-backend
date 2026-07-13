"""Unit tests for HybridSearcher.HybridMatch candidate assembly."""

import asyncio
import uuid
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from statgpt.app.schemas.query_builder import HybridMatchTimings
from statgpt.app.services.hybrid_searcher import (
    HarmonizedItemScored,
    HybridCandidateScored,
    HybridSearcher,
    PlainItemScored,
    RelevanceScore,
    RelevancyResponse,
    SearchParams,
)
from statgpt.common.hybrid_indexer.schemas import IndicatorIndex, MatchingIndex
from statgpt.common.utils.elastic import SearchResult
from statgpt.common.vectorstore import ScoredVectorStoreDocument


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


def _plain(id_: str, score: float, dataset_id: str = "ds1") -> PlainItemScored:
    metadata = MatchingIndex.model_validate(_indicator(id_, dataset_id).model_dump())
    return PlainItemScored(score=score, metadata=metadata)


def _doc(id_: str, score: float, dataset_id: str = "ds1") -> ScoredVectorStoreDocument:
    return ScoredVectorStoreDocument(
        page_content=f"name {id_}",
        metadata=_indicator(id_, dataset_id).model_dump(),
        table_name="indicators",
        document_id=1,
        dataset_id=uuid.uuid4(),
        version_id=1,
        score=score,
    )


def _search_result(indicators: list[IndicatorIndex]) -> SearchResult:
    return SearchResult.model_validate(
        {
            "took": 1,
            "timed_out": False,
            "hits": {
                "total": {"value": len(indicators), "relation": "eq"},
                "max_score": 1.0 if indicators else None,
                "hits": [
                    {
                        "_index": "indicators",
                        "_id": indicator.id,
                        "_score": 1.0,
                        "_source": indicator.model_dump(),
                    }
                    for indicator in indicators
                ],
            },
        }
    )


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


async def test_es_get_by_ids_no_query_for_empty_ids(match):
    match._outer._indicators_index.search = AsyncMock()

    assert await match._es_get_by_ids([]) == {}
    match._outer._indicators_index.search.assert_not_awaited()


async def test_es_get_by_ids_warns_per_missing_id(match, monkeypatch):
    logger_mock = Mock()
    monkeypatch.setattr("statgpt.app.services.hybrid_searcher.logger", logger_mock)
    match._outer._indicators_index.search = AsyncMock(
        return_value=_search_result([_indicator("a")])
    )

    found = await match._es_get_by_ids(["a", "b", "c"])

    assert set(found) == {"a"}
    warned = [call.args[0] for call in logger_mock.warning.call_args_list]
    assert any("b" in msg for msg in warned)
    assert any("c" in msg for msg in warned)
    assert len(warned) == 2


async def test_hybrid_combination_fetches_missing_ids_in_one_query(match):
    """Ids absent from the lexical results are fetched with a single terms query (no N+1)."""
    sem_raw = [
        _doc("a", 0.9),  # present in lexical -> no ES lookup
        _doc("b", 0.8),  # missing from lexical -> fetched from ES
        _doc("c", 0.7),  # missing from lexical and from ES -> dropped
        _doc("d", 0.6),  # filtered out by availability -> not fetched
        _doc("b", 0.5),  # duplicate id -> requested once
    ]
    lexical = {"a": _lex("a", 0.5)}
    semantic = {"a": _plain("a", 0.9), "b": _plain("b", 0.8), "c": _plain("c", 0.7)}
    search_mock = AsyncMock(return_value=_search_result([_indicator("b")]))
    match._outer._indicators_index.search = search_mock

    hybrid = await match._hybrid_combination(
        sem_raw=sem_raw, lexical=lexical, semantic=semantic, alpha=0.5
    )

    search_mock.assert_awaited_once_with(query={"terms": {"id.keyword": ["b", "c"]}}, size=2)
    assert set(hybrid) == {"a", "b"}
    assert hybrid["a"].score == pytest.approx(0.5 * 0.9 + 0.5 * 0.5)
    assert hybrid["b"].score == pytest.approx(0.5 * 0.8)


async def test_hybrid_candidates_runs_lexical_and_semantic_concurrently(match):
    match._outer.config.max_output_div = 2
    match._outer.config.max_lexical_only_candidates = 0

    semantic_started = asyncio.Event()

    async def fake_lexical(query, version_ids, max_query):
        # Resolves only after the semantic search has started: sequential
        # execution would hang here (bounded by the wait_for timeouts).
        await asyncio.wait_for(semantic_started.wait(), timeout=1)
        return {"a": _lex("a", 0.5)}

    async def fake_semantic_raw(query, version_ids, max_query):
        semantic_started.set()
        return [_doc("a", 0.9)]

    match._lexical = fake_lexical
    match._semantic_raw = fake_semantic_raw

    timings = HybridMatchTimings()
    search_params = SearchParams(
        alpha=0.5, max_candidates=10, max_semantic_candidates=10, max_lexical_candidates=10
    )
    lex_filtered, sem_filtered, candidates = await asyncio.wait_for(
        match._hybrid_candidates(
            query="q",
            version_ids={1},
            availability={"ds1": {}},
            search_params=search_params,
            timings=timings,
        ),
        timeout=1,
    )

    assert set(lex_filtered) == {"a"}
    assert set(sem_filtered) == {"a"}
    assert [c["id"] for c in candidates] == ["a"]
    # per-part timings are measured inside each coroutine
    assert timings.lexical > 0
    assert timings.semantic_raw > 0


def _indexed_entry(num: str, real_id: str, dataset_id: str = "ds1") -> dict:
    """An `indexed` entry in the shape produced by _prepare_for_relevance."""
    return {
        "id": real_id,
        "dataset_id": dataset_id,
        "primary": f"primary {num}",
        "name": f"name {num}",
        "name_original": f"Name {num}",
        "where": [],
        "series": [],
    }


def test_add_llm_scores_maps_scores(match):
    """The structured RelevanceScore list maps to scored candidates, preserving
    the real candidate ids/scores."""
    indexed = {
        "1": _indexed_entry("1", "real-a"),
        "2": _indexed_entry("2", "real-b"),
    }
    scores = [
        RelevanceScore(number=1, score=2),
        RelevanceScore(number=2, score=0),
    ]

    result = match._add_llm_scores_to_indexed(indexed=indexed, scores=scores)

    assert all(isinstance(c, HybridCandidateScored) for c in result)
    assert [(c.id, c.score) for c in result] == [("real-a", 2), ("real-b", 0)]


def _relevance_item(num: str) -> dict:
    """An item in the shape produced by _prepare_for_relevance / _pre_append_confirmed."""
    return {
        "id": num,
        "dataset_id": "ds1",
        "primary": f"primary {num}",
        "name": f"name {num}",
        "where": [{"dim": "value"}],
    }


async def test_relevance_candidates_drops_anchor_and_invented_numbers(match, monkeypatch):
    """LLM scores are filtered at the source: the cross-batch anchor (0) is dropped
    silently, numbers absent from the batch are dropped with a warning, so consumers
    can look candidates up in `indexed` without guards."""
    logger_mock = Mock()
    monkeypatch.setattr("statgpt.app.services.hybrid_searcher.logger", logger_mock)
    match._outer._relevancy_chain.ainvoke = AsyncMock(
        return_value=RelevancyResponse(
            relevance=[
                RelevanceScore(number=0, score=3),  # cross-batch anchor
                RelevanceScore(number=1, score=2),
                RelevanceScore(number=7, score=3),  # invented by the LLM
            ]
        )
    )
    items = [_relevance_item("0"), _relevance_item("1")]

    scores = await match._relevance_candidates("some query", items)

    assert [(s.number, s.score) for s in scores] == [(1, 2)]
    warned = [call.args[0] for call in logger_mock.warning.call_args_list]
    assert len(warned) == 1 and "7" in warned[0]


def test_relevance_score_rejects_out_of_range_values():
    """score is constrained to 0-3 and number to >= 0; anything else must fail validation."""
    with pytest.raises(ValidationError):
        RelevanceScore(number=1, score=4)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RelevanceScore(number=-1, score=1)
