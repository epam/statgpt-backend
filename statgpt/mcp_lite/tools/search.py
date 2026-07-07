import json
from typing import Annotated

from fastmcp.dependencies import Depends
from fastmcp.exceptions import ToolError

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import CategoricalDimension
from statgpt.common.utils.elastic import ElasticSearchFactory
from statgpt.mcp_lite.deps import get_channel_facade
from statgpt.mcp_lite.schemas import (
    CodeMatch,
    DatasetIndicatorGroup,
    IndicatorMatch,
    IndicatorSearchResult,
    SearchCodesResult,
)

from ._provider import mcp_tools

_SEARCH_MAX_K = 100
# Convex combination weight for the indicator hybrid: alpha * sem + (1 - alpha) * lex.
# Matches HybridSearcher's default — semantic-leaning since lex is sharper for
# exact-name hits, sem catches semantic neighbours; we want both but trust sem more.
_HYBRID_ALPHA = 0.9


def _indicator_dimensions(metadata: dict) -> dict[str, str] | None:
    """Extract dim_id -> value-code mapping from a hybrid-indexed indicator document.

    Hybrid-indexed docs store the indicator's dim breakdown as a JSON-encoded
    `series` list of `{dim_id: value}` singletons in metadata (same shape on
    both the ES and pgvector sides).
    """
    series_raw = metadata.get("series")
    if not series_raw:
        return None
    try:
        series = json.loads(series_raw) if isinstance(series_raw, str) else series_raw
    except (TypeError, ValueError):
        return None
    dims: dict[str, str] = {}
    for entry in series:
        if not isinstance(entry, dict) or not entry:
            continue
        for k, v in entry.items():
            dims[k] = v
    return dims or None


async def _hybrid_indicator_search(
    facade: ChannelServiceFacade,
    query: str,
    k: int,
    version_ids: list[int],
    entity_to_source: dict[str, str],
    auth_context: AuthContext,
) -> list[tuple[str, IndicatorMatch]]:
    """Hybrid lex + sem indicator search.

    Mirrors `HybridSearcher`'s pipeline at a smaller scope: ES BM25 on the
    `indicators_index` (lex side) plus pgvector cosine on the indicator vector
    store (sem side), normalized and combined via convex combination. Skips
    the LLM rerank that HybridSearcher does on top.
    """
    es_index = await ElasticSearchFactory.get_index(facade.channel.indicators_index_name)
    pool = max(k * 4, 50)

    # -- lex side
    es_query = {
        "bool": {
            "must": [{"match": {"primary_normalized": {"query": query}}}],
            "should": [{"match": {"name_normalized": {"query": query, "boost": 0.3}}}],
            "filter": [{"terms": {"version_id": list(version_ids)}}],
        }
    }
    lex_result = await es_index.search(query=es_query, size=pool)
    lex_max = lex_result.hits.max_score or 0.0
    lex_norm: dict[str, float] = {}
    lex_meta: dict[str, dict] = {}
    for hit in lex_result.hits.hits:
        meta = hit.source
        _id = meta.get("id")
        if _id and _id not in lex_norm:
            lex_norm[_id] = (hit.score / lex_max) if lex_max > 0 else 0.0
            lex_meta[_id] = meta

    # -- sem side
    vector_store = await facade._get_indicators_vector_store(auth_context)
    sem_docs = await vector_store.search_with_similarity_score(
        query, k=pool, version_ids=set(version_ids)
    )
    sem_max = sem_docs[0].score if sem_docs else 0.0
    sem_norm: dict[str, float] = {}
    sem_meta: dict[str, dict] = {}
    sem_ds: dict[str, str] = {}
    for doc in sem_docs:
        _id = doc.metadata.get("id")
        if _id and _id not in sem_norm:
            sem_norm[_id] = (doc.score + 1) / (sem_max + 1) if (sem_max + 1) > 0 else 0.0
            sem_meta[_id] = doc.metadata
            sem_ds[_id] = str(doc.dataset_id)

    # -- combine
    alpha = _HYBRID_ALPHA
    all_ids = set(sem_norm) | set(lex_norm)
    scored = [
        (_id, alpha * sem_norm.get(_id, 0.0) + (1 - alpha) * lex_norm.get(_id, 0.0))
        for _id in all_ids
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    results: list[tuple[str, IndicatorMatch]] = []
    for _id, score in scored[:k]:
        meta = sem_meta.get(_id) or lex_meta.get(_id) or {}
        ds_uuid = sem_ds.get(_id) or str(meta.get("dataset_id", ""))
        dims = _indicator_dimensions(meta) or {}
        # `code` is the SDMX-style dotted key of the indicator's pinned dim values,
        # in series order (e.g. "CPI._T.IX"). The raw indexer id (`metadata.id`) is a
        # composite "<dataset_uuid> <version> <dim-key>" — useless to the agent and
        # eats ~70 bytes per row. The dim breakdown remains in `dimensions`.
        code = ".".join(dims.values()) if dims else str(meta.get("id", _id))
        ds_source = entity_to_source.get(ds_uuid, ds_uuid)
        results.append(
            (
                ds_source,
                IndicatorMatch(
                    code=code,
                    name=str(meta.get("name", "")),
                    score=score,
                    dimensions=dims,
                ),
            )
        )
    return results


@mcp_tools.tool
async def search_indicators(
    query: Annotated[str, "Free-text concept (e.g. 'GDP growth', 'consumer prices')."],
    dataset_id: Annotated[
        str | None,
        "Optional: scope to one dataset. Null = search every dataset in the channel.",
    ] = None,
    top_k: Annotated[
        int,
        "Total indicators to return. Default 50, cap 100. Values below 50 are NOT "
        "advised — recall on broad/vague concepts depends on having multiple datasets "
        "represented in the result, and shrinking the budget consistently drops "
        "borderline-but-relevant datasets. Use 50 for the verbatim user-question "
        "search; bump to 100 for broadening paraphrases that target hard-to-reach "
        "datasets.",
    ] = 50,
    facade: ChannelServiceFacade = Depends(get_channel_facade),  # type: ignore[arg-type]
) -> IndicatorSearchResult:
    """Find indicators (compound series) by free-text concept, grouped by dataset.

    Top-K indicators are bucketed into `datasets[]` (sorted by best_score desc);
    each `matches[i].dimensions` is a ready selection for `execute_sdmx_query`.

    **Multi-dataset answers are the norm, not the exception, in this channel.**
    A single concept is typically exposed by several datasets, each with a
    different framing: definition, periodicity, unit, valuation, or sectoral
    aggregation. When the user's question does not pin a specific framing,
    treat **every** group in `datasets[]` as a distinct candidate worth
    fetching, not just the top-scoring one. A high score on a group ranked
    2nd, 3rd, ... means that dataset genuinely answers the question through
    a different lens — it is not a duplicate of group 1. Selecting one
    dataset when the channel offers several is the most common failure mode
    here.

    For atomic dim values (countries, sectors, instruments), use `search_codes`.
    `score` is per-query max-relative in [0, 1].
    """
    if top_k <= 0:
        raise ToolError("top_k must be a positive integer")
    if top_k > _SEARCH_MAX_K:
        top_k = _SEARCH_MAX_K

    auth = SystemUserAuthContext()
    versioned = await facade.list_available_datasets(auth)
    if dataset_id is not None:
        versioned = [v for v in versioned if v.data.source_id == dataset_id]
        if not versioned:
            raise ToolError(f"Dataset not found in this channel: {dataset_id!r}")

    if not versioned:
        return IndicatorSearchResult(query=query, n_total_matches=0, datasets=[])

    version_ids = [v.version.version_data_id for v in versioned]
    entity_to_source = {str(v.data.entity_id): v.data.source_id for v in versioned}

    try:
        scored_matches = await _hybrid_indicator_search(
            facade=facade,
            query=query,
            k=top_k,
            version_ids=version_ids,
            entity_to_source=entity_to_source,
            auth_context=auth,
        )
    except Exception as e:
        raise ToolError(f"indicator search failed: {e!r}")

    # Group by dataset_id while preserving the within-dataset ranking
    # (input `scored_matches` is already sorted globally by score desc).
    grouped: dict[str, list[IndicatorMatch]] = {}
    for ds_source, match in scored_matches:
        grouped.setdefault(ds_source, []).append(match)

    groups = [
        DatasetIndicatorGroup(
            dataset_id=ds_id,
            best_score=ms[0].score,
            matches=ms,
        )
        for ds_id, ms in grouped.items()
    ]
    groups.sort(key=lambda g: g.best_score, reverse=True)

    return IndicatorSearchResult(
        query=query,
        n_total_matches=len(scored_matches),
        datasets=groups,
    )


def _classify_dim(dataset, dim) -> str:
    """Same classification logic as `_classify_dim` in tools/dataset.py.

    Returns one of: 'time', 'special', 'indicator', 'non_indicator'.
    """
    if dim.is_time_dimension:
        return "time"
    special_ids = {d.entity_id for d in dataset.special_dimensions().values()}
    if dim.entity_id in special_ids:
        return "special"
    indicator_ids = {d.entity_id for d in dataset.indicator_dimensions()}
    if dim.entity_id in indicator_ids:
        return "indicator"
    return "non_indicator"


@mcp_tools.tool
async def search_codes(
    dataset_id: Annotated[
        str,
        "Dataset id from `list_datasets`. Required — dim codes are dataset-specific.",
    ],
    query: Annotated[str, "Free-text concept (e.g. 'Germany', 'EUR', 'banks total')."],
    dim_id: Annotated[
        str | None,
        "Optional dim scope (e.g. 'COUNTRY'). When set, must be a non-indicator or special "
        "dim; only that dim's codelist is searched. When null, searches all non-indicator + "
        "special dims of the dataset — useful to find which dim contains a concept "
        "(e.g. 'Japan' might appear in `L_CP_COUNTRY`, `L_REP_CTY`, `L_PARENT_CTY`).",
    ] = None,
    top_k: Annotated[
        int,
        "How many matches to return. Default 20, cap 100. With `dim_id` set, 10 is usually plenty.",
    ] = 20,
    facade: ChannelServiceFacade = Depends(get_channel_facade),  # type: ignore[arg-type]
) -> SearchCodesResult:
    """Find atomic dim-value codes by free-text within one dataset.

    Searches the non-indicator and special-dim vector stores, scoped to one
    dataset. Returns codes you can put directly into `availability_query.filter`
    or `execute_sdmx_query.selection`.

    Not for indicator dims: their values are properties of compound indicators —
    use `search_indicators` and read `dimensions` from a matching result.
    Not for the time dim: filter by `time_start` / `time_end` on `execute_sdmx_query`.

    Score gap between rank 1 and rank 2 is the strongest signal: when rank 1
    dominates, prefer it; when several rank similarly, the dim has multiple
    legitimate codes for the query.
    """
    if top_k <= 0:
        raise ToolError("top_k must be a positive integer")
    if top_k > _SEARCH_MAX_K:
        top_k = _SEARCH_MAX_K

    auth = SystemUserAuthContext()
    dataset = await facade.get_dataset_by_source_id(auth, dataset_id)
    if dataset is None:
        raise ToolError(f"Dataset not found in this channel: {dataset_id!r}")

    versioned = await facade.list_available_datasets(auth)
    matching = next((v for v in versioned if v.data.source_id == dataset_id), None)
    if matching is None:
        raise ToolError(f"Dataset {dataset_id!r} has no current version in this channel.")
    version_id = matching.version.version_data_id

    # Up-front dim validation when caller specified one: redirect bad dim-types
    # with a concrete recovery hint before issuing any vector call.
    if dim_id is not None:
        try:
            dim = dataset.dimension(dim_id)
        except KeyError:
            raise ToolError(f"Dimension {dim_id!r} not found in dataset {dataset_id!r}")
        dim_type = _classify_dim(dataset, dim)
        if dim_type == "indicator":
            raise ToolError(
                f"{dim_id!r} is an indicator dimension in {dataset_id!r}. Its values are "
                f"properties of compound indicators, not standalone codes. Either:\n"
                f"  - call `search_indicators(query=<concept>, dataset_id={dataset_id!r})` "
                f"and read `dimensions.{dim_id}` from a matching result; or\n"
                f"  - call `sample_dim_values(dataset_id={dataset_id!r}, "
                f"dim_id={dim_id!r}, limit=-1)` to list every value."
            )
        if dim_type == "time":
            raise ToolError(
                f"{dim_id!r} is the time dimension; it has no codelist to search. "
                f"Filter by date range on `execute_sdmx_query("
                f"time_start='YYYY-MM-DD', time_end='YYYY-MM-DD')`."
            )
        if not isinstance(dim, CategoricalDimension):
            raise ToolError(
                f"{dim_id!r} in {dataset_id!r} isn't a categorical dim; no codelist to search."
            )

    by_code: dict[tuple[str, str], CodeMatch] = {}

    # -- non-indicator path: existing facade method, optional dim_id pre-filter on the
    # non-indicator vector store. Runs only when dim_id is unset (search all non-indicator
    # dims) OR when the specified dim is non-indicator-classified.
    if dim_id is None or _classify_dim(dataset, dataset.dimension(dim_id)) == "non_indicator":
        try:
            ni_candidates = await facade.search_non_indicator_dimensions_scored(
                query=query,
                auth_context=auth,
                k=top_k * 2,
                dataset_versions=[version_id],
                dimension_id=dim_id,
            )
        except Exception as e:
            raise ToolError(f"non_indicator search failed: {e!r}")
        ni_max = max((c.score for c in ni_candidates), default=0.0)
        denom = ni_max + 1
        for c in ni_candidates:
            if c.dataset_id != str(matching.data.entity_id):
                continue
            norm = (c.score + 1) / denom if denom > 0 else 0.0
            key = (c.dimension_id, c.query_id)
            existing = by_code.get(key)
            if existing is None or norm > existing.score:
                by_code[key] = CodeMatch(
                    source="non_indicator",
                    dataset_id=dataset_id,
                    dim_id=c.dimension_id,
                    code=c.query_id,
                    name=c.name,
                    score=norm,
                )

    # -- special-dim path: separate vector store keyed by dim_id. Runs only when
    # dim_id is unset (search all special dims) OR when the specified dim is special.
    special_dim_ids: set[str] = {d.entity_id for d in dataset.special_dimensions().values()}
    targets = (
        [dim_id]
        if dim_id is not None and dim_id in special_dim_ids
        else (list(special_dim_ids) if dim_id is None else [])
    )
    for tgt_dim_id in targets:
        try:
            sp_candidates = await facade.search_special_dim_values_by_id(
                query=query,
                dim_id=tgt_dim_id,
                auth_context=auth,
                k=top_k * 2,
                dataset_versions=[version_id],
            )
        except Exception as e:
            raise ToolError(f"special-dim search failed for {tgt_dim_id!r}: {e!r}")
        sp_max = max((c.score for c in sp_candidates), default=0.0)
        denom = sp_max + 1
        for c in sp_candidates:
            if c.dataset_id != str(matching.data.entity_id):
                continue
            norm = (c.score + 1) / denom if denom > 0 else 0.0
            key = (c.dimension_id, c.query_id)
            existing = by_code.get(key)
            if existing is None or norm > existing.score:
                by_code[key] = CodeMatch(
                    source="special",
                    dataset_id=dataset_id,
                    dim_id=c.dimension_id,
                    code=c.query_id,
                    name=c.name,
                    score=norm,
                )

    matches = sorted(by_code.values(), key=lambda m: m.score, reverse=True)[:top_k]
    return SearchCodesResult(query=query, matches=matches)
