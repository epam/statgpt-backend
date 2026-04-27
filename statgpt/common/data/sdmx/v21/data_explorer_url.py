"""Build data explorer / viewer query URLs from ``DataExplorerUrlConfig`` and a query."""

import logging
from collections.abc import Callable
from urllib.parse import quote, urlencode

from sdmx.model.v21 import DataStructureDefinition

from statgpt.common.data.sdmx.common.data_explorer_url import DataExplorerUrlConfig

from .query import SdmxDataSetQuery
from .utils import convert_keys_to_str

_log = logging.getLogger(__name__)

DimensionValuesResolver = Callable[[str, list[str]], list[str]]


def _dataflow_component_ids_in_order(dsd: DataStructureDefinition) -> list[str]:
    try:
        return [c.id for c in dsd.dimensions.components]
    except (AttributeError, TypeError):
        return []


def _ordered_dimension_keys(
    dsd: DataStructureDefinition, key_dict: dict[str, list[str]]
) -> list[str]:
    order = _dataflow_component_ids_in_order(dsd)
    pos: dict[str, int] = {d: i for i, d in enumerate(order)}

    def _sort_key(dim_id: str) -> tuple[int, str]:
        return (pos.get(dim_id, 10_000), dim_id)

    return sorted(key_dict.keys(), key=_sort_key)


def _build_aggregated_filter_value(
    dsd: DataStructureDefinition,
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
    for dim_id in _ordered_dimension_keys(dsd, key_dict):
        codes = key_dict[dim_id]
        if not codes:
            continue
        label = name_map.get(dim_id, dim_id)
        values = list(codes)
        if (
            cfg.aggregated_dimension_value_mode.get(dim_id) == "name"
            and dimension_values_resolver is not None
        ):
            values = dimension_values_resolver(dim_id, codes)
        value = vsep.join(values)
        parts.append(f"{label}{kvsep}{value}")

    if (
        cfg.time_encoding == "in_aggregated_filter"
        and cfg.aggregated_time_param
        and sdmx_query.time_dimension_query
    ):
        tq = sdmx_query.time_dimension_query
        sp, ep = tq.start_period, tq.end_period
        tsep = cfg.aggregated_time_range_separator
        if sp and ep:
            parts.append(f"{cfg.aggregated_time_param}{kvsep}{sp}{tsep}{ep}")
        elif sp or ep:
            one = (sp or ep) or ""
            parts.append(f"{cfg.aggregated_time_param}{kvsep}{one}")
    if not parts:
        return ""
    return dsep.join(parts)


def build_data_explorer_url_query(
    base_url: str,
    short_urn: str,
    dsd: DataStructureDefinition,
    key_dict: dict[str, list[str]],
    sdmx_query: SdmxDataSetQuery,
    config: DataExplorerUrlConfig | None,
    dimension_values_resolver: DimensionValuesResolver | None = None,
) -> str:
    """Return a full explorer URL: ``base`` + ``?`` + query (legacy if ``config`` is None)."""
    cfg = config or DataExplorerUrlConfig()
    params: dict[str, str] = {}

    if cfg.include_dataflow_urn_param:
        params[cfg.dataset_urn_param] = short_urn

    if cfg.include_series_key_filter:
        if cfg.filter_format == "sdmx_key_string":
            params[cfg.series_key_filter_param] = convert_keys_to_str(dsd, key_dict)
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

    if cfg.time_encoding == "start_end_sdmx21":
        _add_sdmx21_time_params(params, sdmx_query)
    # in_aggregated_filter: time is inside the filter; none: no time

    safe_chars = "+,.:*" if cfg.filter_format == "sdmx_key_string" else ",.:*"
    query_string = urlencode(params, quote_via=quote, safe=safe_chars)
    if not query_string:
        return base_url
    return f"{base_url}?{query_string}"


def _add_sdmx21_time_params(params: dict[str, str], sdmx_query: SdmxDataSetQuery) -> None:
    try:
        if td_query := sdmx_query.time_dimension_query:
            start_period = td_query.start_period
            end_period = td_query.end_period
        else:
            start_period = None
            end_period = None

        if start_period:
            params['startPeriod'] = start_period if '-' in start_period else f"{start_period}-A"
        if end_period:
            params['endPeriod'] = end_period if '-' in end_period else f"{end_period}-A"
    except Exception as e:  # noqa: BLE001
        _log.exception(e)


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
