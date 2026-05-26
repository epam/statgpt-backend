"""MCP tool `execute_sdmx_query` plus its supporting helpers.

Split from `tools/dataset.py` because the execute path carries enough machinery
(bare-year workaround, time-period default applier, dim auto-filler) that it
makes the dataset-introspection tools harder to read when colocated.

All public surface goes through `execute_sdmx_query`; the leading-underscore
helpers are private to this module.
"""

import json
import re
from typing import Annotated

from fastmcp.dependencies import Depends
from fastmcp.exceptions import ToolError

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.app.chains.utils import time_period_utils
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.common.data.base import DataSetAvailabilityQuery, DataSetQuery, DimensionQuery
from statgpt.common.data.base.dimension import VirtualDimension
from statgpt.common.data.base.enums import DimensionDataType, QueryOperator
from statgpt.common.data.base.query import Query, create_time_period_query_from
from statgpt.common.schemas.enums import DataRequestStatus
from statgpt.mcp_lite.deps import get_channel_facade
from statgpt.mcp_lite.schemas import ExecuteResult, TimePeriod

from ._provider import mcp_tools

_EXECUTE_MAX_LIMIT = 500

_BARE_YEAR_RE = re.compile(r"^\d{4}$")


def _expand_bare_year(value: str | None, *, end: bool) -> str | None:
    """Expand bare-year time periods like '2022' to ISO date.

    Workaround for an upstream SDMX 3.0 proxy bug: bare-year periods
    (e.g. '2022') combined with BETWEEN/EQ operators are silently rejected
    with an empty body. Full ISO dates ('2022-01-01') work fine.

    Start of range → 'YYYY-01-01'; end of range → 'YYYY-12-31'.
    Anything that isn't exactly four digits is returned unchanged
    (already a quarter/month/ISO date, or None).
    """
    if value is None or not _BARE_YEAR_RE.match(value):
        return value
    return f"{value}-12-31" if end else f"{value}-01-01"


def _default_time_period_query(dataset) -> DimensionQuery | None:
    """Build a DimensionQuery from the dataset's configured default time period.

    Lifts the relevant slice of `_apply_default_time_period_if_possible` from
    statgpt's `FinalizeQueryStage` ([statgpt/app/chains/data_query/query_builder/
    query/finalize_query.py:204]) without the availability-overlap optimization —
    if the default doesn't intersect the available range, `execute_sdmx_query`
    will return `row_count=0` and the caller can retry.

    Reads `dataset.config.time_period_dimension`'s `default_queries` (e.g.
    `["-5y","now"]` BETWEEN), resolves any relative references via
    `time_period_utils.get_relative_aware_time_period_query`, and returns a
    `DimensionQuery` marked `is_default=True`. Returns None if the dataset has
    no time dim or no configured default.
    """
    try:
        time_dim = dataset.get_time_dimension()
    except (ValueError, AttributeError):
        return None
    if time_dim is None:
        return None

    try:
        dim_id, dim_config = dataset.config.time_period_dimension
    except (AttributeError, ValueError):
        return None

    defaults = getattr(dim_config, "default_queries", None) or []
    if not defaults:
        return None

    resolved = time_period_utils.get_relative_aware_time_period_query(defaults[0])
    if not resolved.values:
        return None

    return DimensionQuery.from_default_query(resolved, dimension_id=dim_id)


_AUTO_FILL_AVAILABILITY_K_LOW = 10
"""When availability returns up to this many values for an un-pinned dim, auto-set them all."""


def _auto_fill_dim_query(
    dim,
    dataset,
    availability: Query | None,
) -> DimensionQuery | None:
    """Try to construct a DimensionQuery for a dim the caller didn't pin.

    Mirrors statgpt's [`SimpleQueryConstructor._set_dimension_query_from_default_or_available_values`](
    statgpt/app/chains/data_query/query_constructor/simple.py:27) — the auto-fill
    logic that the LangChain `data_query` pipeline runs at finalize-query time so
    end-users in the chat UI don't have to specify every required dim. We need
    the same here so agents don't either.

    Priority order (first hit wins):
        1. `dim.config.default_queries` (per-dim, dataset-configured) — filtered
           against availability when categorical.
        2. `dataset.default_value_codes` (dataset-/source-wide, e.g. ['_T','_Z']
           "total" markers) ∩ availability.
        3. Availability has ≤ K_LOW values → use them all (operator=ALL).

    Returns None if none of the above can produce a valid pin; the data layer
    will then raise "missing dimensions" through `execute_sdmx_query`.
    """
    dim_id = dim.entity_id

    # --- (1) per-dim configured default
    default_queries: list[Query] | None = None
    try:
        dim_config = dataset.config.dimensions[dim_id]
        default_queries = dim_config.default_queries
    except (AttributeError, KeyError):
        pass

    if default_queries:
        default_query = default_queries[0]
        if default_query.values:
            if dim.dimension_type != DimensionDataType.CATEGORY:
                # Non-categorical (time) is handled by `_default_time_period_query` already.
                return None
            if availability is not None and availability.values:
                filtered = set(default_query.values) & set(availability.values)
                if filtered:
                    return DimensionQuery(
                        values=list(filtered),
                        operator=default_query.operator,
                        dimension_id=dim_id,
                        is_default=True,
                    )

    if dim.dimension_type != DimensionDataType.CATEGORY:
        return None
    if availability is None or not availability.values:
        return None

    # --- (2) dataset-wide default value codes ∩ availability
    default_value_codes = getattr(dataset, "default_value_codes", None) or []
    if default_value_codes:
        intersection = set(default_value_codes) & set(availability.values)
        if intersection:
            return DimensionQuery(
                values=list(intersection),
                operator=QueryOperator.IN,
                dimension_id=dim_id,
                is_default=True,
            )

    # --- (3) low-cardinality availability → take everything
    if len(list(availability.values)) <= _AUTO_FILL_AVAILABILITY_K_LOW:
        return DimensionQuery(
            values=list(availability.values),
            operator=QueryOperator.ALL,
            dimension_id=dim_id,
            is_default=False,
        )

    return None


@mcp_tools.tool
async def execute_sdmx_query(
    dataset_id: Annotated[str, "Dataset id from `list_datasets` (e.g. 'IMF.STA:CPI(5.0.0)')."],
    selection: Annotated[
        dict[str, str | list[str]],
        "SDMX key/selection: dim_id -> code or list of codes. Scalar for one "
        "('FREQ': 'Q'), list for several ('COUNTRY': ['USA','DEU']). Empty list "
        "= all values for that dim. Required dims must be present (see "
        "`dataset_structure`); codes come from `sample_dim_values` / `search_codes`.",
    ],
    time_start: Annotated[
        str | None,
        "Start of time range — SDMX period: '2010-Q1', '2024-01', '2022-01-01'. "
        "Bare years ('2022') are auto-expanded to 'YYYY-01-01'. "
        "If null along with `time_end`, the dataset's default range applies.",
    ] = None,
    time_end: Annotated[
        str | None,
        "End of time range — same format as `time_start`. Bare years auto-expand "
        "to 'YYYY-12-31'.",
    ] = None,
    limit: Annotated[
        int,
        "Max rows in `data` (full count always in `row_count`). Default 50, cap 500.",
    ] = 50,
    facade: ChannelServiceFacade = Depends(get_channel_facade),  # type: ignore[arg-type]
) -> ExecuteResult:
    """Compose and run an SDMX data query, return rows + URL.

    The caller is responsible for choosing dimension values that produce a
    non-empty result (use `availability_query` first if unsure). On success
    returns the resolved upstream `query_url`, the total `row_count`, the
    actual time range observed in the data, and up to `limit` preview rows.
    """
    if limit <= 0:
        raise ToolError("limit must be a positive integer")
    if limit > _EXECUTE_MAX_LIMIT:
        limit = _EXECUTE_MAX_LIMIT

    auth_context = SystemUserAuthContext()
    dataset = await facade.get_dataset_by_source_id(auth_context, dataset_id)
    if dataset is None:
        raise ToolError(f"Dataset not found in this channel: {dataset_id!r}")

    # Normalize the union-typed input: a scalar code is treated as a one-element list.
    # Declared at the tool boundary (in the annotation above), not silent.
    normalized: dict[str, list[str]] = {
        k: [v] if isinstance(v, str) else list(v) for k, v in selection.items()
    }

    queries: list[DimensionQuery] = [
        DimensionQuery(
            dimension_id=dim_id,
            values=values,
            operator=QueryOperator.IN if values else QueryOperator.ALL,
        )
        for dim_id, values in normalized.items()
    ]
    if time_start or time_end:
        time_dim_id = dataset.get_time_dimension().entity_id
        # Auto-expand bare years to ISO dates as a workaround for the SDMX 3.0
        # proxy bug (see `_expand_bare_year` docstring).
        start = _expand_bare_year(time_start, end=False)
        end = _expand_bare_year(time_end, end=True)
        tp_query = create_time_period_query_from(start=start, end=end, dimension_id=time_dim_id)
        if tp_query is not None:
            queries.append(tp_query)
    else:
        # Caller omitted `time_period`. Fall back to the dataset's configured default
        # (e.g. {"values": ["-5y", "now"], "operator": "between"}). Same source the
        # statgpt chain pipeline uses — no invented behaviour. If the dataset has no
        # default configured, the data layer will raise "missing dimensions:
        # ['TIME_PERIOD']" and we surface it as a clear ToolError below.
        default_tp_query = _default_time_period_query(dataset)
        if default_tp_query is not None:
            queries.append(default_tp_query)

    # Auto-fill unspecified non-time dims, same mechanism statgpt's chain uses for
    # end-users in the chat UI (see `_auto_fill_dim_query` docstring). Reads dataset
    # config + an availability call to choose a default per dim. Skipped entirely if
    # caller already pinned every required dim — costs one extra SDMX call only when
    # auto-fill is needed.
    pinned_ids = {q.dimension_id for q in queries}
    unspecified = [d for d in dataset.dimensions() if d.entity_id not in pinned_ids]
    if unspecified:
        try:
            av_resp = await dataset.availability_query(
                DataSetAvailabilityQuery.from_dimension_queries_list(queries),
                auth_context=auth_context,
            )
        except Exception:
            av_resp = None

        for dim in unspecified:
            if isinstance(dim, VirtualDimension):
                continue
            availability_for_dim = (
                av_resp.dimensions_queries_dict.get(dim.entity_id) if av_resp else None
            )
            fill_query = _auto_fill_dim_query(dim, dataset, availability_for_dim)
            if fill_query is not None:
                queries.append(fill_query)

    query = DataSetQuery(dimensions_queries=queries)

    try:
        response = await dataset.query(query, auth_context)
    except ValueError as e:
        raise ToolError(f"Invalid query for {dataset_id!r}: {e}")
    except Exception as e:
        raise ToolError(f"execute_sdmx_query failed for {dataset_id!r}: {e!r}")

    if response is None:
        raise ToolError(f"Upstream returned no response for {dataset_id!r}")

    upstream_warning: str | None = None
    if response.status.request_status != DataRequestStatus.SUCCESS:
        # The StatGPT SDMX proxy returns HTTP 200 + empty body for queries with no
        # matching observations; the SDMX-JSON reader fails to parse, and the data
        # layer surfaces this as `request_status=FAILED, parsing_status=NA`. Rather
        # than raising — which the agent can't act on — return an empty result with a
        # diagnostic so the agent can retry with a different filter / wider time period.
        upstream_url = getattr(response, "url_query", None) or "<unknown>"
        upstream_warning = (
            f"Upstream returned non-success status "
            f"(request_status={response.status.request_status.value}, "
            f"parsing_status={response.status.parsing_status.value}). "
            "Most often this means the chosen dim combination has no data; less often "
            "the proxy is unreachable or the query is malformed. Use `availability_query` "
            "to confirm the filter is reachable, or widen `time_period`. "
            f"Upstream URL: {upstream_url}"
        )

    df = getattr(response, "df", None)
    row_count = int(len(df)) if df is not None else 0

    time_range: TimePeriod | None = None
    tp = getattr(response, "time_period", None)
    if tp:
        time_range = TimePeriod(start=tp[0], end=tp[1])

    rows: list[dict] = []
    if df is not None and row_count > 0:
        preview = df.head(limit).reset_index()
        rows = json.loads(preview.to_json(orient="records", date_format="iso"))

    query_url = getattr(response, "url_query", None)

    return ExecuteResult(
        dataset_id=dataset.source_id,
        query_url=query_url,
        row_count=row_count,
        truncated=row_count > len(rows),
        time_range_actual=time_range,
        data=rows,
        warning=upstream_warning,
    )
