import random
from typing import Annotated

from fastmcp.dependencies import Depends
from fastmcp.exceptions import ToolError

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.common.data.base import CategoricalDimension, DataSetAvailabilityQuery, DimensionQuery
from statgpt.common.data.base.dimension import VirtualDimension
from statgpt.common.data.base.enums import QueryOperator
from statgpt.mcp_lite.deps import get_channel_facade
from statgpt.mcp_lite.schemas import (
    AvailabilityResult,
    Datasets,
    DatasetStructure,
    DatasetSummary,
    DimensionInfo,
    DimensionType,
    DimValue,
    DimValuesSample,
    TimePeriod,
)

from ._provider import mcp_tools


@mcp_tools.tool
async def list_datasets(
    facade: ChannelServiceFacade = Depends(get_channel_facade),  # type: ignore[arg-type]
) -> Datasets:
    """List datasets exposed by this channel.

    Returns each dataset's stable id (the source-system identifier, e.g.
    `BIS_CPI`), human-readable title, and source URL if available. Use the
    returned `id` as the `dataset_id` argument to `dataset_structure`,
    `sample_dim_values`, `availability_query`, and `execute_sdmx_query`.
    """
    auth_context = SystemUserAuthContext()
    versioned = await facade.list_available_datasets(auth_context)
    return Datasets(
        datasets=[
            DatasetSummary(
                id=v.data.source_id,
                name=v.data.name,
                url=v.data.dataset_url,
            )
            for v in versioned
        ]
    )


def _classify_dim(dataset, dim) -> DimensionType:
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
async def dataset_structure(
    dataset_id: Annotated[str, "Dataset id as returned by `list_datasets` (e.g. 'BIS_CPI')."],
    facade: ChannelServiceFacade = Depends(get_channel_facade),  # type: ignore[arg-type]
) -> DatasetStructure:
    """Describe a dataset's dimensions: id, type, and codelist size.

    Dimension `type` is one of:
    - `indicator`     — the indicator dimension(s) (use `search_indicators`)
    - `non_indicator` — categorical filter dim (use `sample_dim_values`)
    - `special`       — special-purpose dim (e.g. unit/frequency, treated like non_indicator)
    - `time`          — the time dimension (no codelist; filter via `time_period`)

    `codelist_size` is the number of available values, or null for time.
    Large codelists aren't enumerated here; call `sample_dim_values` to peek.
    """
    auth_context = SystemUserAuthContext()
    dataset = await facade.get_dataset_by_source_id(auth_context, dataset_id)
    if dataset is None:
        raise ToolError(f"Dataset not found in this channel: {dataset_id!r}")

    dims: list[DimensionInfo] = []
    for dim in dataset.dimensions():
        if isinstance(dim, VirtualDimension):
            codelist_size: int | None = 1
        elif dim.is_time_dimension:
            codelist_size = None
        elif isinstance(dim, CategoricalDimension):
            available = getattr(dim, "available_values", None)
            codelist_size = len(available) if available is not None else len(dim.values)
        else:
            codelist_size = None

        dims.append(
            DimensionInfo(
                id=dim.entity_id,
                name=dim.name,
                type=_classify_dim(dataset, dim),
                alias=dim.alias,
                codelist_size=codelist_size,
            )
        )

    return DatasetStructure(id=dataset.source_id, name=dataset.name, dims=dims)


@mcp_tools.tool
async def sample_dim_values(
    dataset_id: Annotated[str, "Dataset id from `list_datasets` (e.g. 'IMF.STA:CPI(5.0.0)')."],
    dim_id: Annotated[
        str,
        "Dim id from `dataset_structure` (e.g. 'COUNTRY').",
    ],
    limit: Annotated[
        int,
        "How many values to return: random sample of `min(limit, total)`. "
        "Pass -1 to return every value (the full codelist — can be hundreds). "
        "Default 20.",
    ] = 20,
    facade: ChannelServiceFacade = Depends(get_channel_facade),  # type: ignore[arg-type]
) -> DimValuesSample:
    """Peek at the values of one dimension in a dataset.

    Returns codes + human-readable names. `code` is the source-system value
    id (e.g. 'DE' for Germany) that you pass to `availability_query` or
    `execute_sdmx_query`; `name` is the label for display.

    Pass `limit=-1` to retrieve every available value (full codelist).
    Otherwise returns a random sample of size `min(limit, total)`.
    Not valid for time dimensions — use a time-period argument on
    `execute_sdmx_query` instead.
    """
    if limit != -1 and limit <= 0:
        raise ToolError("limit must be -1 (all values) or a positive integer")

    auth_context = SystemUserAuthContext()
    dataset = await facade.get_dataset_by_source_id(auth_context, dataset_id)
    if dataset is None:
        raise ToolError(f"Dataset not found in this channel: {dataset_id!r}")

    try:
        dim = dataset.dimension(dim_id)
    except KeyError:
        raise ToolError(f"Dimension {dim_id!r} not found in dataset {dataset_id!r}")

    if dim.is_time_dimension:
        raise ToolError(
            f"Dimension {dim_id!r} is a time dimension; sampling doesn't apply. "
            "Filter by time period on `execute_sdmx_query` instead."
        )
    if not isinstance(dim, CategoricalDimension):
        raise ToolError(f"Dimension {dim_id!r} is not categorical; nothing to sample.")

    available = list(dim.available_values)
    total = len(available)

    if limit == -1 or limit >= total:
        picked = available
    else:
        picked = random.sample(available, limit)

    return DimValuesSample(
        dataset_id=dataset.source_id,
        dim_id=dim.entity_id,
        total=total,
        returned=len(picked),
        is_full=(len(picked) == total),
        values=[DimValue(code=v.entity_id, name=v.name) for v in picked],
    )


@mcp_tools.tool
async def availability_query(
    dataset_id: Annotated[str, "Dataset id from `list_datasets` (e.g. 'IMF.STA:CPI(5.0.0)')."],
    filter: Annotated[
        dict[str, str | list[str]],
        "Partial filter: dim_id -> code or list of codes. Scalar for one "
        "('FREQ': 'Q'), list for several ('COUNTRY': ['USA','DEU']). Empty list = "
        "all values for that dim. At least one entry required (an empty filter "
        "would return everything). Use codes (not names) from `sample_dim_values` "
        "or `search_codes`.",
    ],
    facade: ChannelServiceFacade = Depends(get_channel_facade),  # type: ignore[arg-type]
) -> AvailabilityResult:
    """Given partial dimension filters, return reachable values for the *other* dimensions.

    Core search primitive — call iteratively to narrow a query. Example:
    after picking an indicator and a country, call this to see which other
    dimensions still have valid combinations (which frequencies, which
    counterpart areas, etc.) under that filter.

    Returns `available` as `{dim_id: [{code, name}, …]}` for every dimension
    of the dataset (including the ones in your filter — narrowed to the
    chosen values). If the dataset exposes a time range, it's returned in
    `time_period`. Raises ToolError when the filter selects zero observations.
    """
    if not filter:
        raise ToolError(
            "filter must include at least one dim to avoid oversized responses. "
            "Call `sample_dim_values` first to pick at least one dimension value."
        )

    # Normalize the union-typed input: a scalar code is treated as a one-element list.
    # Declared at the tool boundary (in the annotation above), not silent.
    normalized: dict[str, list[str]] = {
        k: [v] if isinstance(v, str) else list(v) for k, v in filter.items()
    }

    auth_context = SystemUserAuthContext()
    dataset = await facade.get_dataset_by_source_id(auth_context, dataset_id)
    if dataset is None:
        raise ToolError(f"Dataset not found in this channel: {dataset_id!r}")

    queries = [
        DimensionQuery(
            dimension_id=dim_id,
            values=values,
            operator=QueryOperator.IN if values else QueryOperator.ALL,
        )
        for dim_id, values in normalized.items()
    ]
    av_query = DataSetAvailabilityQuery.from_dimension_queries_list(queries)

    try:
        response = await dataset.availability_query(query=av_query, auth_context=auth_context)
    except Exception as e:
        raise ToolError(f"availability_query failed for {dataset_id!r}: {e!r}")

    available: dict[str, list[DimValue]] = {}
    map_id_to_name = getattr(dataset, "map_component_values_id_2_name", None)
    for dim_id, dim_query in response.dimensions_queries_dict.items():
        code_ids = list(dim_query.values)
        name_map = (
            map_id_to_name(value_ids=code_ids, component_id=dim_id) if map_id_to_name else None
        ) or {}
        available[dim_id] = [DimValue(code=cid, name=name_map.get(cid) or cid) for cid in code_ids]

    if not available:
        invalid = _invalid_filter_codes(dataset, normalized)
        if invalid:
            raise ToolError(
                f"Filter selects no observations: codes not in the dataset's codelist: {invalid}. "
                "Use `search_codes(dataset_id=..., dim_id=...)` or `sample_dim_values` to find "
                "valid replacements."
            )
        raise ToolError(
            "Filter selects no observations, though all codes are valid for the dataset. "
            "Try widening the filter or removing one dim."
        )

    time_period: TimePeriod | None = None
    if response.time_period_start or response.time_period_end:
        time_period = TimePeriod(start=response.time_period_start, end=response.time_period_end)

    return AvailabilityResult(
        dataset_id=dataset.source_id,
        filter=normalized,
        available=available,
        time_period=time_period,
    )


def _invalid_filter_codes(dataset, filter_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    """For each (dim_id, codes) in the filter, return codes not in the dim's codelist.

    Local check (no SDMX call). Only categorical dims are validated — virtual / time /
    non-categorical dims are skipped. Dims that don't exist in the dataset are surfaced
    via the data-layer's KeyError before we ever get here, so we don't re-check.
    """
    bad: dict[str, list[str]] = {}
    for dim_id, codes in filter_dict.items():
        if not codes:
            continue
        try:
            dim = dataset.dimension(dim_id)
        except KeyError:
            continue
        if not isinstance(dim, CategoricalDimension):
            continue
        known = {v.entity_id for v in dim.values}
        missing = [c for c in codes if c not in known]
        if missing:
            bad[dim_id] = missing
    return bad
