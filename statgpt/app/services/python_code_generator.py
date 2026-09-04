import logging
from typing import Never

from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.common.data.sdmx.python_code import generate_python_query_body
from statgpt.common.schemas.query import JsonComponentQuery, JsonQueryOperator
from statgpt.common.schemas.record_id import build_sdmx_series_key

_log = logging.getLogger(__name__)

PYTHON_SDMX1_HEADER = """\
# Uses the [sdmx1 library](https://pypi.org/project/sdmx1/)
# Install with:
# ```bash
# pip install sdmx1
# ```

import sdmx"""


def _build_flow_ref(agency_id: str, resource_id: str, version: str) -> str:
    return f"{agency_id},{resource_id},{version}"


def _invalid_time_filter(reason: str, f: JsonComponentQuery) -> Never:
    raise ValueError(
        f"{reason} (time filter on {f.component_code!r}: "
        f"operator={f.operator!r}, n_values={len(f.values)})."
    )


def _time_filter_rest_params(f: JsonComponentQuery) -> dict[str, str]:
    """Return ``startPeriod`` / ``endPeriod`` entries implied by one time filter."""
    op = f.operator
    values = f.values
    n = len(values)

    if op == JsonQueryOperator.BETWEEN:
        if n != 2:
            _invalid_time_filter(
                f"BETWEEN requires exactly two period values; got {n}",
                f,
            )
        return {"startPeriod": values[0], "endPeriod": values[1]}

    if op == JsonQueryOperator.GE:
        if not values:
            _invalid_time_filter("GE requires at least one period", f)
        return {"startPeriod": values[0]}

    if op == JsonQueryOperator.LE:
        if not values:
            _invalid_time_filter("LE requires at least one period", f)
        return {"endPeriod": values[0]}

    if op == JsonQueryOperator.IN:
        if not values:
            _invalid_time_filter("IN requires at least one period", f)
        if n == 1:
            p = values[0]
            return {"startPeriod": p, "endPeriod": p}
        _invalid_time_filter(
            "Time dimension 'in' with multiple values cannot be expressed as a single "
            "sdmx1 startPeriod/endPeriod request; use BETWEEN or separate queries",
            f,
        )

    if op in (JsonQueryOperator.GT, JsonQueryOperator.LT):
        _invalid_time_filter(
            f"Exclusive time bound {op!r} is not supported for Python sdmx1 snippets: "
            "SDMX REST only supports inclusive startPeriod/endPeriod; "
            "use ge/le/between instead",
            f,
        )

    _invalid_time_filter(f"Unsupported time filter operator {op!r}", f)


def _build_params_from_filters(filters: list[JsonComponentQuery]) -> dict[str, str]:
    """Map time-dimension filters to SDMX REST ``startPeriod`` / ``endPeriod``.

    SDMX uses inclusive bounds only; ``gt`` / ``lt`` cannot be translated faithfully
    without calendar-aware stepping, so they are rejected.
    """
    params: dict[str, str] = {"detail": "full"}
    for f in filters:
        params.update(_time_filter_rest_params(f))
    return params


def generate_python_code_from_query(query: AppJsonQueryWithMetadata, suffix: str = "") -> str:
    flow_ref = _build_flow_ref(query.agency_id, query.resource_id, query.version)

    provider = query.sdmx1_source or query.agency_id

    time_component = query.metadata.time_period_dimension
    key = build_sdmx_series_key(
        query.filters,
        time_component,
        query.metadata.key_dimension_ids_in_dsd_order,
    )

    time_filters = [f for f in query.filters if f.component_code == time_component]
    params = _build_params_from_filters(time_filters)

    return generate_python_query_body(
        provider=provider,
        flow_ref=flow_ref,
        key=key,
        params=params,
        suffix=suffix,
    )


def _snippet_or_placeholder(query: AppJsonQueryWithMetadata, suffix: str = "") -> str:
    """Return a query's sdmx1 snippet, or a commented placeholder if it can't be expressed.

    Some queries have no faithful SDMX REST representation (e.g. exclusive ``gt`` / ``lt``
    time bounds, or a multi-value ``in`` on the time dimension) and raise ValueError. We
    keep the rest of the merged snippet usable by substituting an explanatory comment for
    just the offending query instead of dropping the whole snippet.
    """
    try:
        return generate_python_code_from_query(query, suffix=suffix)
    except ValueError:
        _log.exception("Failed to generate sdmx1 snippet for query %s", query.urn)
        return f"# Unable to generate a reproducible sdmx1 snippet for {query.urn}."


def generate_merged_python_code(queries: list[AppJsonQueryWithMetadata]) -> str:
    queries = [
        q
        for q in queries
        if not q.disabled and not any(f.operator == JsonQueryOperator.EXCLUDED for f in q.filters)
    ]
    if not queries:
        return PYTHON_SDMX1_HEADER
    if len(queries) == 1:
        body = _snippet_or_placeholder(queries[0])
    else:
        sections = [
            f"# Dataset: {query.urn}\n{_snippet_or_placeholder(query, suffix=f'_{i}')}"
            for i, query in enumerate(queries, start=1)
        ]
        body = "\n\n".join(sections)

    return PYTHON_SDMX1_HEADER + "\n\n" + body
