"""Configuration and builders for data explorer deep links (SDMX-backed queries)."""

import logging
from collections.abc import Callable
from enum import StrEnum
from typing import Any
from urllib.parse import quote, urlencode

from pydantic import Field, model_validator

from statgpt.common.data.base import BaseModel

from .query import SdmxDataSetQuery


class TimeEncodingSdmx(StrEnum):
    """Where the time range is encoded in the data explorer URL."""

    start_end_sdmx21 = "start_end_sdmx21"
    """Use separate query params (`startPeriod`, `endPeriod`)."""

    none = "none"
    """Do not add explicit time to the URL."""

    in_aggregated_filter = "in_aggregated_filter"
    """Put time into the aggregated filter entry."""


class FilterFormatSdmx(StrEnum):
    """How categorical filters are serialized in the URL."""

    sdmx_key_string = "sdmx_key_string"
    """Use positional SDMX key format."""

    key_value_aggregated = "key_value_aggregated"
    """Use `NAME=value` entries joined by delimiters."""


class AggregatedValueModeSdmx(StrEnum):
    """How each aggregated filter value is represented."""

    code = "code"
    """Keep raw SDMX code values."""

    name = "name"
    """Resolve SDMX codes to display labels where available."""


DimensionValuesResolver = Callable[[str, list[str]], list[str]]
SdmxKeyBuilder = Callable[[Any, dict[str, list[str]]], str]

_log = logging.getLogger(__name__)


class DataExplorerUrlConfig(BaseModel):
    """Parameters for building data-explorer deep links from SDMX queries."""

    view_in_data_explorer: bool = Field(
        default=True,
        description=(
            "Whether to include a 'View data in explorer' link with query results. "
            "When False, query results have no URL attached."
        ),
    )
    include_dataflow_urn_param: bool = Field(
        default=True,
        description=(
            "If false, do not add a query parameter for the dataflow URN. "
            "If true, the parameter name is `datasetUrnParam`."
        ),
    )
    dataset_urn_param: str = Field(
        default="urn",
        description="Query parameter name for the dataflow URN when includeDataflowUrnParam is true.",
    )
    filter_format: FilterFormatSdmx = Field(
        default=FilterFormatSdmx.sdmx_key_string,
        description="How the filter value is built from the query key and DSD.",
    )
    include_series_key_filter: bool = Field(
        default=True,
        description="If true, add a filter value under seriesKeyFilterParam (when the portal uses one).",
    )
    series_key_filter_param: str = Field(
        default="filter",
        description="Query parameter name for the series filter string.",
    )
    time_encoding: TimeEncodingSdmx = Field(
        default=TimeEncodingSdmx.start_end_sdmx21,
        description=(
            "start_end_sdmx21: add startPeriod/endPeriod. "
            "none: no separate time query parameters. "
            "in_aggregated_filter: append a time range inside the aggregated filter (see aggregatedTimeParam)."
        ),
    )
    aggregated_entry_delimiter: str = Field(
        default="^",
        description="Between NAME=value units when filterFormat is key_value_aggregated.",
    )
    aggregated_key_value_separator: str = Field(
        default="=",
        description="Between the dimension label and code(s) in each unit.",
    )
    aggregated_values_separator: str = Field(
        default="+",
        description="Between multiple values for the same dimension in one unit.",
    )
    aggregated_time_param: str | None = Field(
        default=None,
        description=(
            "Left-hand name for the time range segment when timeEncoding is in_aggregated_filter "
            "(e.g. a TIMESPAN field)."
        ),
    )
    aggregated_time_range_separator: str = Field(
        default="_",
        description="Between start and end in the time segment value.",
    )
    aggregated_dimension_param_names: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map SDMX dimension IDs (from the DSD) to the portal's filter token names. "
            "Omitted dimensions use the DSD id as the name."
        ),
    )
    aggregated_dimension_value_mode: dict[str, AggregatedValueModeSdmx] = Field(
        default_factory=dict,
        description=(
            "How to encode each dimension's values in key_value_aggregated mode: "
            "'code' (default) keeps SDMX codes; 'name' converts codes to labels when available."
        ),
    )

    @model_validator(mode='after')
    def _aggregated_time_consistent(self) -> "DataExplorerUrlConfig":
        if (
            self.time_encoding == TimeEncodingSdmx.in_aggregated_filter
            and not self.aggregated_time_param
        ):
            raise ValueError(
                "aggregatedTimeParam is required when timeEncoding is in_aggregated_filter"
            )
        if (
            self.time_encoding == TimeEncodingSdmx.in_aggregated_filter
            and self.filter_format != FilterFormatSdmx.key_value_aggregated
        ):
            raise ValueError("in_aggregated_filter requires filterFormat key_value_aggregated")
        return self


def _dataflow_component_ids_in_order(dsd: Any) -> list[str]:
    try:
        return [c.id for c in dsd.dimensions.components]
    except (AttributeError, TypeError):
        _log.debug(
            "Could not read dimension order from DSD; falling back to alphabetical.",
            exc_info=True,
        )
        return []


def _ordered_dimension_keys(dsd: Any, key_dict: dict[str, list[str]]) -> list[str]:
    order = _dataflow_component_ids_in_order(dsd)
    pos: dict[str, int] = {d: i for i, d in enumerate(order)}

    def _sort_key(dim_id: str) -> tuple[int, str]:
        return (pos.get(dim_id, 10_000), dim_id)

    return sorted(key_dict.keys(), key=_sort_key)


def _build_aggregated_filter_value(
    dsd: Any,
    key_dict: dict[str, list[str]],
    sdmx_query: SdmxDataSetQuery,
    cfg: DataExplorerUrlConfig,
    dimension_values_resolver: DimensionValuesResolver | None = None,
) -> str:
    dsep, kvsep, vsep = (
        cfg.aggregated_entry_delimiter,
        cfg.aggregated_key_value_separator,
        cfg.aggregated_values_separator,
    )
    name_map = cfg.aggregated_dimension_param_names
    parts: list[str] = []
    warned_dims: set[str] = set()
    for dim_id in _ordered_dimension_keys(dsd, key_dict):
        codes = key_dict[dim_id]
        if not codes:
            continue
        label = name_map.get(dim_id, dim_id)
        values = list(codes)
        wants_names = (
            cfg.aggregated_dimension_value_mode.get(dim_id) == AggregatedValueModeSdmx.name
        )
        if wants_names:
            if dimension_values_resolver is not None:
                values = dimension_values_resolver(dim_id, codes)
            elif dim_id not in warned_dims:
                _log.warning(
                    "value_mode='name' requested for dim %s but no resolver provided; "
                    "codes will be used as-is",
                    dim_id,
                )
                warned_dims.add(dim_id)
        value = vsep.join(values)
        parts.append(f"{label}{kvsep}{value}")

    tq = sdmx_query.time_dimension_query
    if (
        cfg.time_encoding == TimeEncodingSdmx.in_aggregated_filter
        and cfg.aggregated_time_param
        and tq is not None
    ):
        sp, ep = tq.start_period, tq.end_period
        tsep = cfg.aggregated_time_range_separator
        if sp and ep:
            parts.append(f"{cfg.aggregated_time_param}{kvsep}{sp}{tsep}{ep}")
        elif sp:
            parts.append(f"{cfg.aggregated_time_param}{kvsep}{sp}{tsep}")
        elif ep:
            parts.append(f"{cfg.aggregated_time_param}{kvsep}{tsep}{ep}")
    if not parts:
        return ""
    return dsep.join(parts)


def build_data_explorer_url_query(
    base_url: str,
    short_urn: str,
    dsd: Any,
    key_dict: dict[str, list[str]],
    sdmx_query: SdmxDataSetQuery,
    config: DataExplorerUrlConfig | None,
    sdmx_key_builder: SdmxKeyBuilder | None = None,
    dimension_values_resolver: DimensionValuesResolver | None = None,
) -> str:
    """Return a full explorer URL: ``base`` + ``?`` + query (legacy if ``config`` is None)."""
    cfg = config or DataExplorerUrlConfig()
    params: dict[str, str] = {}

    if cfg.include_dataflow_urn_param:
        params[cfg.dataset_urn_param] = short_urn

    if cfg.include_series_key_filter:
        if cfg.filter_format == FilterFormatSdmx.sdmx_key_string:
            if sdmx_key_builder is None:
                raise ValueError("sdmx_key_builder is required for filterFormat='sdmx_key_string'")
            params[cfg.series_key_filter_param] = sdmx_key_builder(dsd, key_dict)
        else:
            filter_body = _build_aggregated_filter_value(
                dsd=dsd,
                key_dict=key_dict,
                sdmx_query=sdmx_query,
                cfg=cfg,
                dimension_values_resolver=dimension_values_resolver,
            )
            if filter_body:
                params[cfg.series_key_filter_param] = filter_body

    if cfg.time_encoding == TimeEncodingSdmx.start_end_sdmx21:
        _add_sdmx21_time_params(params, sdmx_query)
    # in_aggregated_filter: time is inside the filter; none: no time

    safe_chars = "+,.:*()" if cfg.filter_format == FilterFormatSdmx.sdmx_key_string else ",.:*()"
    query_string = urlencode(params, quote_via=quote, safe=safe_chars)
    if not query_string:
        return base_url
    return f"{base_url}?{query_string}"


def _add_sdmx21_time_params(params: dict[str, str], sdmx_query: SdmxDataSetQuery) -> None:
    td_query = sdmx_query.time_dimension_query
    if td_query is None:
        return

    start_period = td_query.start_period
    end_period = td_query.end_period

    if start_period:
        params['startPeriod'] = start_period if '-' in start_period else f"{start_period}-A"
    if end_period:
        params['endPeriod'] = end_period if '-' in end_period else f"{end_period}-A"


def build_data_explorer_dataset_url(
    base_url: str, short_urn: str, config: DataExplorerUrlConfig | None
) -> str:
    """Build a link for ``dataset_url`` when ``useDataExplorerForDatasetUrl`` is set."""
    cfg = config or DataExplorerUrlConfig()
    if not cfg.include_dataflow_urn_param:
        return base_url.rstrip("?&")
    param = {cfg.dataset_urn_param: short_urn}
    query_string = urlencode(param, quote_via=quote, safe="+,.:*()")
    return f"{base_url}?{query_string}"
