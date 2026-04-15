import re

from statgpt.common.data.sdmx.python_code import generate_python_query_body
from statgpt.common.schemas.query import (
    JsonComponentQuery,
    JsonQueryOperator,
    JsonQueryWithMetadata,
)

PYTHON_SDMX1_HEADER = """\
# Uses the [sdmx1 library](https://pypi.org/project/sdmx1/)
# Install with:
# ```bash
# pip install sdmx1
# ```

import sdmx"""

_URN_PATTERN = re.compile(r"^(?P<agency>[^:]+):(?P<resource>[^(]+)\((?P<version>[^)]+)\)$")


def _parse_urn(urn: str) -> tuple[str, str, str]:
    """Parse short URN like 'AGENCY:RESOURCE(VERSION)' into (agency_id, resource_id, version)."""
    match = _URN_PATTERN.match(urn)
    if not match:
        raise ValueError(f"Invalid URN format: {urn!r}")
    return match.group("agency"), match.group("resource"), match.group("version")


def _build_flow_ref(agency_id: str, resource_id: str, version: str) -> str:
    return f"{agency_id},{resource_id},{version}"


def _build_key_from_filters(filters: list[JsonComponentQuery], time_component: str) -> str:
    """Build SDMX key string from non-time categorical filters.

    Joins multiple values per dimension with '+' and dimensions with '.'.
    """
    parts = []
    for f in filters:
        if f.component_code == time_component:
            continue
        parts.append("+".join(f.values))
    return ".".join(parts)


def _build_params_from_filters(filters: list[JsonComponentQuery]) -> dict[str, str]:
    params: dict[str, str] = {"detail": "full"}
    for f in filters:
        if f.operator == JsonQueryOperator.BETWEEN and len(f.values) == 2:
            params["startPeriod"] = f.values[0]
            params["endPeriod"] = f.values[1]
        elif f.operator == JsonQueryOperator.GE and f.values:
            params["startPeriod"] = f.values[0]
        elif f.operator == JsonQueryOperator.LE and f.values:
            params["endPeriod"] = f.values[0]
    return params


def _detect_time_component(filters: list[JsonComponentQuery]) -> str:
    """Detect the time dimension component code from filters."""
    time_operators = {JsonQueryOperator.BETWEEN, JsonQueryOperator.GE, JsonQueryOperator.LE}
    for f in filters:
        if f.operator in time_operators:
            return f.component_code
    return "TIME_PERIOD"


def generate_python_code_from_query(
    query: JsonQueryWithMetadata,
    suffix: str = "",
) -> str:
    agency_id, resource_id, version = _parse_urn(query.urn)
    flow_ref = _build_flow_ref(agency_id, resource_id, version)

    provider = query.sdmx1_source or agency_id

    time_component = _detect_time_component(query.filters)
    key = _build_key_from_filters(query.filters, time_component)

    time_filters = [f for f in query.filters if f.component_code == time_component]
    params = _build_params_from_filters(time_filters)

    return generate_python_query_body(
        provider=provider,
        flow_ref=flow_ref,
        key=key,
        params=params,
        suffix=suffix,
    )


def generate_merged_python_code(queries: list[JsonQueryWithMetadata]) -> str:
    if len(queries) == 1:
        body = generate_python_code_from_query(queries[0])
        return PYTHON_SDMX1_HEADER + "\n\n" + body

    bodies: list[str] = []
    for i, query in enumerate(queries, start=1):
        body = generate_python_code_from_query(query, suffix=f"_{i}")
        bodies.append(f"# Dataset: {query.urn}\n{body}")

    return PYTHON_SDMX1_HEADER + "\n\n" + "\n\n".join(bodies)
