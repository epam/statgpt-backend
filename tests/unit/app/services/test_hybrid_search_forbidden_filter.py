from types import SimpleNamespace

from statgpt.app.services.hybrid_searcher import HybridSearcher
from statgpt.common.utils import AsyncLoadingCache


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


async def test_filter_drops_partial_token_matches():
    # The match must land on whole token boundaries: "grow" inside "growth" and "nomic"
    # inside "economic" must NOT count as present.
    searcher = _searcher_with_fake_tokenizer()
    kept = await searcher._filter_candidates_present_in_query(
        "economic growth", {"grow", "nomic grow"}
    )
    assert kept == set()


async def test_filter_drops_phrase_spanning_token_boundary():
    # "c d" is a substring of "abc def" but spans the abc/def token boundary - drop it.
    searcher = _searcher_with_fake_tokenizer()
    kept = await searcher._filter_candidates_present_in_query("abc def", {"c d"})
    assert kept == set()


async def test_filter_drops_empty_tokenization():
    # A candidate made only of stop-words tokenizes to "". The empty guard matters precisely
    # when the query ALSO reduces to "": without it both operands render as padded blanks
    # (" {} " -> "  ") and the candidate would spuriously match. It must be dropped.
    searcher = object.__new__(HybridSearcher)

    async def fake_tokenize(value: str) -> str:
        # emulate ES stop-word removal: pure stop-word phrases reduce to nothing
        return "" if value in {"of the", "the"} else value.lower()

    searcher._tokenize = fake_tokenize  # type: ignore[method-assign]
    kept = await searcher._filter_candidates_present_in_query("the", {"of the"})
    assert kept == set()


async def test_filter_empty_returns_empty():
    searcher = _searcher_with_fake_tokenizer()
    assert await searcher._filter_candidates_present_in_query("anything", set()) == set()


async def test_tokenize_memoizes_and_is_case_insensitive():
    # The same string (regardless of case) is analyzed by Elasticsearch at most once.
    searcher = object.__new__(HybridSearcher)
    searcher._tokenize_cache = AsyncLoadingCache()
    analyze_calls: list[str] = []

    class _FakeIndex:
        async def analyze(self, *, text: str) -> list[SimpleNamespace]:
            analyze_calls.append(text)
            return [SimpleNamespace(token=tok) for tok in text.split()]

    searcher._matching_index = _FakeIndex()  # type: ignore[assignment]

    first = await searcher._tokenize("Gross Domestic Product")
    second = await searcher._tokenize("gross domestic product")  # differs only by case

    assert first == second == "gross domestic product"
    assert analyze_calls == ["gross domestic product"]  # exactly one ES round-trip
