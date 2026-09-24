from enum import StrEnum
from typing import Literal, Self

from pydantic import ConfigDict, Field, computed_field, model_validator

from statgpt.app.schemas.data_query_outcome import (
    DataQueryStatus,
    DataSetChoice,
    MissingDimensionsInfo,
)
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.common.schemas.base import BaseYamlModel
from statgpt.common.schemas.query import JsonQueryOperator

# One version number for the whole data query response: every `_meta` audience payload carries it
# and they are bumped together. `structuredContent` does not: the calling model cannot act on it.
DATA_QUERY_RESPONSE_VERSION = 3


class DataQueryToolsInfo(BaseYamlModel):
    """Names of companion MCP tools the caller can use to act on the queries."""

    model_config = ConfigDict(serialize_by_alias=True)

    sdmx_proxy: str | None = Field(
        default=None,
        description="Name of the SDMX-proxy MCP tool, if configured on the channel.",
    )


class FilterValue(BaseYamlModel):
    """One value of a dimension: the code a query uses, with its display name when known."""

    model_config = ConfigDict(serialize_by_alias=True)

    id: str = Field(description="The value's query id (the code used in queries).")
    name: str | None = Field(
        default=None, description="Human-readable name of the value, when known."
    )


class PeriodRange(BaseYamlModel):
    """A time period, named after the SDMX REST query parameters."""

    model_config = ConfigDict(serialize_by_alias=True)

    start_period: str | None = Field(default=None, description="First period covered.")
    end_period: str | None = Field(default=None, description="Last period covered.")


class RequestedPeriod(PeriodRange):
    """The time period a query asked for."""

    is_default: bool = Field(
        default=False,
        description="Whether the period is the dataset's default, applied because the user did"
        " not specify one.",
    )


class QueryFilter(BaseYamlModel):
    """The filter applied to one dimension.

    A dimension with no filter is not listed: every one of its values is included.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    dimension_id: str = Field(description="Entity id of the filtered dimension.")
    dimension_name: str | None = Field(
        default=None, description="Human-readable name of the dimension, when known."
    )
    operator: JsonQueryOperator = Field(description="How the values are applied.")
    values: list[FilterValue] = Field(
        default_factory=list, description="Every value the filter applies."
    )
    is_indicator: bool = Field(
        default=False, description="Whether the dimension is one of the dataset's indicators."
    )
    is_default: bool = Field(
        default=False,
        description="Whether the filter is the dimension's default, applied because the user did"
        " not specify it.",
    )

    @computed_field(  # type: ignore[prop-decorator]
        description="How many values the filter applies."
    )
    @property
    def value_count(self) -> int:
        return len(self.values)


class ExecutionResult(StrEnum):
    """How the execution of one query went."""

    DATA_RECEIVED = "data_received"
    PARTIALLY_PARSED = "partially_parsed"
    PARSING_FAILED = "parsing_failed"
    REQUEST_FAILED = "request_failed"
    NO_DATA = "no_data"


class QueryExecution(BaseYamlModel):
    """The outcome of executing one query, with what to do about it."""

    model_config = ConfigDict(serialize_by_alias=True)

    result: ExecutionResult = Field(description="How the execution went.")
    reason: str | None = Field(default=None, description="Why the execution did not succeed.")
    advice: str | None = Field(default=None, description="How to proceed.")


class MissingDimensionRecord(BaseYamlModel):
    """A required dimension the query does not specify yet."""

    model_config = ConfigDict(serialize_by_alias=True)

    dimension_id: str = Field(description="Entity id of the missing dimension.")
    name: str = Field(description="Human-readable name of the dimension.")
    total_values: int = Field(description="Total number of values available for it.")
    sample_values: list[FilterValue] = Field(
        default_factory=list,
        description="Values available given the rest of the query: all of them when there are at"
        " most 10, otherwise the first 10 (so this is not the full list when `totalValues` exceeds"
        " the sample size).",
    )


class InvalidPeriodRecord(BaseYamlModel):
    """Why the requested time period was rejected: one bound is outside the available range. The
    rejected period is not applied, so the query's `requestedPeriod` does not carry it."""

    model_config = ConfigDict(serialize_by_alias=True)

    rejected_bound: Literal["startPeriod", "endPeriod"] = Field(description="The rejected bound.")
    requested_value: str = Field(description="The value requested for the rejected bound.")
    available_period: PeriodRange = Field(description="The period the dataset has data for.")


class InvalidityReason(StrEnum):
    """Why a constructed query cannot run."""

    INVALID_TIME_PERIOD = "invalid_time_period"
    MISSING_DIMENSIONS = "missing_dimensions"


class QueryInvalidity(BaseYamlModel):
    """Why a constructed query cannot run, with what a follow-up query must change."""

    model_config = ConfigDict(serialize_by_alias=True)

    reason: InvalidityReason = Field(description="Why the query cannot run.")
    explanation: str = Field(description="The reason, in words.")
    rejected_period: InvalidPeriodRecord | None = Field(
        default=None, description="For `invalid_time_period`: the rejected bound of the period."
    )
    missing_dimensions: list[MissingDimensionRecord] | None = Field(
        default=None,
        description="For `missing_dimensions`: the required dimensions the query does not specify.",
    )


class QueryRecord(BaseYamlModel):
    """One dataset query the pipeline produced."""

    model_config = ConfigDict(serialize_by_alias=True)

    query_id: str = Field(
        description="Id of this query within the response; joins it to the result's resources and"
        " `_meta` payloads. Stable for the same query executed on the same date."
    )
    dataset_urn: str = Field(description="URN of the queried dataset, e.g. 'IMF:CPI(1.0.0)'.")
    dataset_name: str | None = Field(default=None, description="Dataset name, when known.")
    is_official: bool | None = Field(
        default=None, description="Whether the dataset is official, when the channel marks it."
    )
    provider: str | None = Field(default=None, description="The dataset's provider, when known.")
    dataset_last_updated: str | None = Field(
        default=None, description="Date the dataset was last updated (ISO 8601), when known."
    )
    dataset_url: str | None = Field(default=None, description="Link to the dataset, when known.")
    query_summary: str | None = Field(
        default=None, description="A short summary of what the query asks for."
    )
    executed: bool = Field(
        description="Whether this query was executed. A constructed but unexecuted query describes"
        " what would be asked, not data that was returned."
    )
    filters: list[QueryFilter] = Field(
        default_factory=list, description="The filters applied, one per filtered dimension."
    )
    requested_period: RequestedPeriod | None = Field(
        default=None, description="The time period the query asked for."
    )
    invalidity: QueryInvalidity | None = Field(
        default=None, description="Why the query cannot run, for an invalid constructed query."
    )
    factual_period: PeriodRange | None = Field(
        default=None, description="The time period the returned data actually covers."
    )
    series_count: int | None = Field(
        default=None, description="Number of data series returned, when the query returned data."
    )
    execution: QueryExecution | None = Field(
        default=None, description="How the execution went, for an executed query."
    )
    data_explorer_url: str | None = Field(
        default=None, description="Link to the query's data in the data explorer."
    )


class MissingDimensionsRecord(BaseYamlModel):
    """Why a query is incomplete: which dimensions still need a value."""

    model_config = ConfigDict(serialize_by_alias=True)

    dataset_urn: str | None = Field(
        default=None, description="URN of the dataset the missing dimensions belong to."
    )
    dimensions: list[MissingDimensionRecord] = Field(
        default_factory=list, description="The missing required dimensions."
    )


class CandidateDatasetRecord(BaseYamlModel):
    """A dataset the query could be narrowed to."""

    model_config = ConfigDict(serialize_by_alias=True)

    id: str = Field(description="Dataset URN. Name it in a follow-up query to pick this dataset.")
    name: str = Field(description="Human-readable dataset name.")
    is_official: bool | None = Field(
        default=None, description="Whether the dataset is official, when the channel marks it."
    )
    query: QueryRecord | None = Field(
        default=None, description="The query that would run against this dataset."
    )


class DataQueryStructuredContent(BaseYamlModel):
    """MCP structured content for the data query tool: the outcome, the queries the pipeline
    produced, and what a follow-up query would need when it produced none.

    Written for the calling model. What a client needs instead - the SDMX wiring, python code,
    resource URIs - is carried in the result's `_meta`.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    status: DataQueryStatus = Field(description="Which outcome the pipeline reached.")
    message: str | None = Field(
        default=None, description="What to know about the outcome, and how to proceed."
    )
    executed_at: str | None = Field(
        default=None, description="When the queries were executed (ISO 8601)."
    )
    queries: list[QueryRecord] = Field(
        default_factory=list, description="The queries, one per dataset."
    )
    missing_dimensions: MissingDimensionsRecord | None = Field(
        default=None,
        description="Dimensions the query must still specify, when it is incomplete.",
    )
    candidate_datasets: list[CandidateDatasetRecord] = Field(
        default_factory=list,
        description="Datasets to narrow the query to, when it matched several.",
    )


class McpAppQuery(AppJsonQueryWithMetadata):
    """One query in the MCP-App payload: the SDMX query model plus its `queryId`."""

    model_config = ConfigDict(serialize_by_alias=True)

    query_id: str = Field(description="Id of this query within the response.")

    @classmethod
    def from_app_query(cls, query: AppJsonQueryWithMetadata, query_id: str) -> Self:
        return cls(**query.model_dump(), query_id=query_id)


class DataQueryMcpAppMeta(BaseYamlModel):
    """The `{namespace}/mcp-app` payload: everything the UI widget renders and edits.

    Null fields are kept rather than omitted, so the payload's shape does not change with the
    outcome.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    status: DataQueryStatus = Field(
        description="Outcome of the data query pipeline (which branch produced the response)."
    )
    message: str | None = Field(
        default=None, description="Human-readable message, e.g. why no data is available."
    )
    queries: list[McpAppQuery] = Field(
        default_factory=list,
        description="The queries, one per dataset. Present for the data_available,"
        " executed_no_data, failed and not_executed outcomes.",
    )
    candidate_datasets: list[DataSetChoice] = Field(
        default_factory=list,
        description="Datasets to choose from for the dataset_selection_required outcome.",
    )
    missing_dimensions: MissingDimensionsInfo | None = Field(
        default=None,
        description="Required dimensions to specify for the missing_dimensions outcome, with every"
        " available value.",
    )
    python_code: str | None = Field(
        default=None,
        description="A self-contained sdmx1 snippet that reproduces the queries, when available.",
    )
    tools: DataQueryToolsInfo = Field(description="Companion MCP tools for these queries.")
    version: int = Field(
        default=DATA_QUERY_RESPONSE_VERSION, description="Schema version of this response."
    )


class ClientQueryRecord(BaseYamlModel):
    """One query as a programmatic client sees it: where to look at it, and what the result carries
    for it."""

    model_config = ConfigDict(serialize_by_alias=True)

    query_id: str = Field(description="Id of this query within the response.")
    urn: str = Field(description="URN of the queried dataset.")
    dataset_name: str | None = Field(default=None, description="Dataset name, when known.")
    data_explorer_url: str | None = Field(
        default=None, description="Deep link that opens this query in the data explorer."
    )
    dataset_url: str | None = Field(default=None, description="Link to the dataset itself.")
    resource_uris: list[str] = Field(
        default_factory=list,
        description="URIs of the resources this result carries for the query, as `content` has"
        " them.",
    )
    series_count: int | None = Field(
        default=None, description="Number of data series returned, when the query returned data."
    )


class DataQueryClientMeta(BaseYamlModel):
    """The `{namespace}/client` payload: what a programmatic client (e.g. Deep Research) needs to
    present and navigate the result."""

    model_config = ConfigDict(serialize_by_alias=True)

    status: DataQueryStatus = Field(
        description="Outcome of the data query pipeline (which branch produced the response)."
    )
    queries: list[ClientQueryRecord] = Field(
        default_factory=list, description="The queries, one per dataset."
    )
    version: int = Field(
        default=DATA_QUERY_RESPONSE_VERSION, description="Schema version of this response."
    )


class SdmxProxyStructuredContent(BaseYamlModel):
    """MCP structured content for the SDMX-proxy passthrough tool.

    Surfaces the upstream HTTP metadata so the MCP-App can distinguish success from
    error responses and know the body's media type.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    status_code: int = Field(description="HTTP status code returned by the upstream request.")
    content_type: str | None = Field(
        default=None, description="Content type of the upstream response body, if known."
    )


class GlossaryTermRecord(BaseYamlModel):
    """One available glossary term. Optional fields are omitted when the glossary has no value for
    them or the tool is not configured to expose them."""

    model_config = ConfigDict(serialize_by_alias=True)

    term: str = Field(
        description="The glossary term. Use this exact value as the id when requesting its "
        "definition via the term-definitions tool."
    )
    domain: str | None = Field(default=None, description="Domain the term belongs to, if exposed.")
    source: str | None = Field(default=None, description="Source of the term, if exposed.")


class AvailableTermsStructuredContent(BaseYamlModel):
    """MCP structured content for the available-glossary-terms tool: the terms as records so a
    caller can pick exact ids to request definitions for."""

    model_config = ConfigDict(serialize_by_alias=True)

    terms: list[GlossaryTermRecord] = Field(
        default_factory=list, description="The available glossary terms."
    )
    count: int = Field(description="Number of available glossary terms.")


class GlossaryDefinitionRecord(BaseYamlModel):
    """One glossary term's definition. Optional fields are omitted when the glossary has no value
    for them."""

    model_config = ConfigDict(serialize_by_alias=True)

    term: str = Field(description="The glossary term, exactly as stored in the glossary.")
    definition: str = Field(description="The term's definition.")
    domain: str | None = Field(default=None, description="Domain the term belongs to, if known.")
    source: str | None = Field(default=None, description="Source of the term, if known.")


class TermDefinitionsStructuredContent(BaseYamlModel):
    """MCP structured content for the term-definitions tool: one record per requested term that the
    glossary knows, with the unknown ones listed separately."""

    model_config = ConfigDict(serialize_by_alias=True)

    definitions: list[GlossaryDefinitionRecord] = Field(
        default_factory=list,
        description="One entry per requested term that was found in the glossary.",
    )
    not_found: list[str] | None = Field(
        default=None,
        description="The requested terms that are not in the glossary, as requested. Omitted when "
        "every requested term was found.",
    )


class ProviderRecord(BaseYamlModel):
    """One data provider, with how many of the channel's datasets it contributes."""

    model_config = ConfigDict(serialize_by_alias=True)

    name: str = Field(description="Provider name, e.g. 'IMF'.")
    dataset_count: int = Field(description="Number of datasets contributed by this provider.")


class DatasetRecord(BaseYamlModel):
    """One dataset the channel exposes. Optional fields are omitted when unknown."""

    model_config = ConfigDict(serialize_by_alias=True)

    id: str = Field(
        description="Dataset URN (source id), e.g. 'IMF:CPI(1.0.0)'. Stable identifier to pass to "
        "the dataset-structure and data-query tools."
    )
    name: str = Field(description="Human-readable dataset name.")
    description: str | None = Field(default=None, description="Dataset description, if available.")
    provider: str | None = Field(default=None, description="Provider name, if known.")
    last_updated: str | None = Field(
        default=None, description="Date the dataset was last updated (ISO 8601), if known."
    )
    url: str | None = Field(default=None, description="Link to the dataset, if available.")
    number_of_indicators: int | None = Field(
        default=None, description="Number of indicators in the dataset, if computed."
    )


class AvailableDatasetsStructuredContent(BaseYamlModel):
    """MCP structured content for the available-datasets tool: every dataset the channel exposes as
    a record with its stable URN, the distinct providers with their dataset counts, and
    channel-wide totals."""

    model_config = ConfigDict(serialize_by_alias=True)

    providers: list[ProviderRecord] = Field(
        default_factory=list,
        description="Distinct providers across the datasets, each with its dataset count.",
    )
    datasets: list[DatasetRecord] = Field(
        default_factory=list, description="The datasets, one record each."
    )
    total_datasets: int = Field(description="Total number of datasets.")
    total_indicators: int | None = Field(
        default=None,
        description="Total number of indicators across the datasets, if indicator counts were "
        "computed.",
    )
    total_agencies: int = Field(
        description="Number of distinct provider agencies across the datasets."
    )


class DatasetValueRecord(BaseYamlModel):
    """One value (code) of a dataset dimension."""

    model_config = ConfigDict(serialize_by_alias=True)

    id: str = Field(description="The value's query id (the code used in queries).")
    name: str = Field(description="Human-readable value name.")


class DatasetComponentRecord(BaseYamlModel):
    """A dimension or attribute of a dataset. Optional fields are omitted when unknown."""

    model_config = ConfigDict(serialize_by_alias=True)

    id: str = Field(description="The component's entity id (e.g. 'REF_AREA').")
    name: str = Field(description="Human-readable component name.")
    type: str = Field(description="The component's data type, e.g. 'category' or 'datetime'.")
    description: str | None = Field(
        default=None, description="The component's description, if available."
    )
    total_values: int | None = Field(
        default=None,
        description="Total number of available values, for a categorical dimension.",
    )
    sample_values: list[DatasetValueRecord] | None = Field(
        default=None,
        description="The dimension's available values, for a categorical dimension: all of them "
        "when there are at most 10, otherwise a random sample of 10 (so when total_values exceeds "
        "the sample size this is not the full list).",
    )
    sample_values_count: int | None = Field(
        default=None,
        description="Number of values listed in `sampleValues`. Lower than `totalValues` when the"
        " list is a sample.",
    )

    @model_validator(mode="after")
    def _derive_sample_values_count(self) -> Self:
        # Derived rather than computed: a computed field is always required by the serialization
        # schema, while this one is omitted along with `sampleValues` for a non-categorical
        # component. Overwritten unconditionally, so it cannot drift from the list it describes.
        self.sample_values_count = (
            len(self.sample_values) if self.sample_values is not None else None
        )
        return self


class DatasetStructureStructuredContent(BaseYamlModel):
    """MCP structured content for the dataset-structure tool: the dataset's identity plus its
    dimensions and attributes, with a bounded sample of each dimension's values. A dataset that
    does not exist is reported as a tool error, so this content always describes a found dataset.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    dataset_id: str = Field(description="The dataset URN (source id).")
    name: str = Field(description="Human-readable dataset name.")
    last_updated: str | None = Field(
        default=None, description="Date the dataset was last updated (ISO 8601), if known."
    )
    dimensions: list[DatasetComponentRecord] = Field(
        default_factory=list, description="The dataset's dimensions."
    )
    attributes: list[DatasetComponentRecord] = Field(
        default_factory=list, description="The dataset's attributes."
    )


class AvailabilityValueRecord(BaseYamlModel):
    """One available code of a dimension under the availability query."""

    model_config = ConfigDict(serialize_by_alias=True)

    id: str = Field(description="The code's query id (the value used in queries).")
    name: str | None = Field(
        default=None, description="Human-readable code name, for a categorical dimension."
    )


class AvailabilityDimensionRecord(BaseYamlModel):
    """Availability coverage for one dimension: how many codes are available under the query and a
    bounded sample of them."""

    model_config = ConfigDict(serialize_by_alias=True)

    id: str = Field(description="The dimension's entity id (e.g. 'REF_AREA').")
    name: str = Field(description="Human-readable dimension name.")
    total_available: int = Field(
        description="Total number of codes available for this dimension under the query."
    )
    returned: int = Field(
        description="Number of codes returned in `values` (bounded by the per-call and hard limits)."
    )
    truncated: bool = Field(
        description="Whether `values` is a truncated subset of the available codes."
    )
    values: list[AvailabilityValueRecord] = Field(
        default_factory=list, description="The available codes, up to the effective limit."
    )


class TimeCoverageRecord(BaseYamlModel):
    """The time dimension's available range under the availability query."""

    model_config = ConfigDict(serialize_by_alias=True)

    dimension_id: str = Field(description="The time dimension's entity id (e.g. 'TIME_PERIOD').")
    name: str = Field(description="Human-readable time dimension name.")
    start: str | None = Field(default=None, description="Earliest available time period, if known.")
    end: str | None = Field(default=None, description="Latest available time period, if known.")


class AvailabilityStructuredContent(BaseYamlModel):
    """MCP structured content for the availability-query tool: per-dimension coverage for the
    (possibly partial) query, with a bounded sample of each dimension's available codes and the
    time range when present. Optional fields are omitted when unknown."""

    model_config = ConfigDict(serialize_by_alias=True)

    dataset_id: str = Field(description="The requested dataset URN (source id).")
    found: bool = Field(description="Whether a dataset with that URN was found.")
    dimensions: list[AvailabilityDimensionRecord] = Field(
        default_factory=list, description="Per-dimension availability coverage."
    )
    time_coverage: TimeCoverageRecord | None = Field(
        default=None,
        description="The time dimension's available range, when the query yields one.",
    )
