"""Configuration for building public web *data explorer* deep links (SDMX-backed queries).

Default behaviour matches the original product: a dataflow URN as one query parameter, a
**filter** value produced from the DSD key as an SDMX-style key string, and
``startPeriod``/``endPeriod`` when the portal expects SDMX 2.1 time parameters.
"""

from typing import Literal

from pydantic import Field, model_validator

from statgpt.common.data.base import BaseModel

TimeEncodingSdmx = Literal['start_end_sdmx21', 'none', 'in_aggregated_filter']
FilterFormatSdmx = Literal['sdmx_key_string', 'key_value_aggregated']
AggregatedValueModeSdmx = Literal['code', 'name']


class DataExplorerUrlConfig(BaseModel):
    """Parameters for the **View in data explorer** link (SDMX 2.1 queries).

    ``filterFormat``:

    * ``sdmx_key_string`` (default): the filter is built with ``convert_keys_to_str`` (DSD + key).
    * ``key_value_aggregated``: a single **filter** query value built from
      ``NAME=value`` segments joined by ``aggregatedEntryDelimiter``, for portals that expect
      a delimited list instead of a dotted SDMX key.

    ``includeDataflowUrnParam``:

    * ``true`` (default): add the dataflow short URN under ``datasetUrnParam``.
    * ``false``: omit it (e.g. when the dataflow is implied by the path in ``dataExplorerUrl`` and
      only a filter query parameter is used).

    If unset on both dataset and data source, the legacy default is: SDMX key filter + SDMX time
    query parameters + URN in the query.
    """

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
        default="sdmx_key_string",
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
        default="start_end_sdmx21",
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
        if self.time_encoding == "in_aggregated_filter" and not self.aggregated_time_param:
            raise ValueError(
                "aggregatedTimeParam is required when timeEncoding is in_aggregated_filter"
            )
        if (
            self.time_encoding == "in_aggregated_filter"
            and self.filter_format != "key_value_aggregated"
        ):
            raise ValueError("in_aggregated_filter requires filterFormat key_value_aggregated")
        return self
