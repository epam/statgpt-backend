import pytest

from statgpt.app.services.hybrid_searcher import HybridSearcher


@pytest.mark.parametrize(
    "needle, haystack, expected",
    [
        (["a", "b"], ["x", "a", "b", "y"], True),
        (["a", "b"], ["a", "x", "b"], False),  # present but not adjacent
        (["b", "a"], ["a", "b"], False),  # wrong order
        ([], ["a"], False),
        (["a", "b", "c"], ["a", "b"], False),  # longer than haystack
        (["a"], ["a"], True),
    ],
)
def test_is_contiguous_sublist(needle, haystack, expected):
    assert HybridSearcher._is_contiguous_sublist(needle, haystack) is expected


def _searcher_with_fake_tokenizer() -> HybridSearcher:
    """A HybridSearcher whose _tokenize just lowercases (no ES / stemming needed)."""
    searcher = object.__new__(HybridSearcher)

    async def fake_tokenize(value: str) -> str:
        return value.lower()

    searcher._tokenize = fake_tokenize  # type: ignore[method-assign]
    return searcher


async def test_filter_drops_phrases_not_contiguous_in_query():
    # Reproduces the reported case: an indicator named "... country groups" surfaces
    # "country groups" for a query about Group of Seven (G7) - it must be dropped.
    searcher = _searcher_with_fake_tokenizer()
    query = "Economic growth for Group of Seven (G7) countries"
    kept = await searcher._filter_candidates_present_in_query(
        query, {"country groups", "economic growth"}
    )
    assert kept == {"economic growth"}


async def test_filter_keeps_real_indicator_phrases():
    searcher = _searcher_with_fake_tokenizer()
    query = "net trade flow of manufacturing items"
    kept = await searcher._filter_candidates_present_in_query(
        query, {"net trade flow", "manufacturing items"}
    )
    assert kept == {"net trade flow", "manufacturing items"}


async def test_filter_empty_returns_empty():
    searcher = _searcher_with_fake_tokenizer()
    assert await searcher._filter_candidates_present_in_query("anything", set()) == set()
