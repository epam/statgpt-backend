from pydantic import ConfigDict, Field

from statgpt.app.schemas.data_query_outcome import (
    DataQueryStatus,
    DataSetChoice,
    MissingDimensionsInfo,
)
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.common.schemas.base import BaseYamlModel


class DataQueryToolsInfo(BaseYamlModel):
    """Names of companion MCP tools the caller can use to act on the queries."""

    model_config = ConfigDict(serialize_by_alias=True)

    sdmx_proxy: str | None = Field(
        default=None,
        description="Name of the SDMX-proxy MCP tool, if configured on the channel.",
    )


class DataQueryStructuredContent(BaseYamlModel):
    """MCP structured content for the data query tool.

    Always carries a ``status`` tagging the pipeline outcome; the remaining fields are
    populated per outcome. Serialized with camelCase aliases to match the DIAL attachment shape.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    status: DataQueryStatus = Field(
        description="Outcome of the data query pipeline (which branch produced the response)."
    )
    queries: list[AppJsonQueryWithMetadata] = Field(
        default_factory=list,
        description="The queries, one per dataset. Present for the data_available, "
        "executed_no_data, failed and not_executed outcomes.",
    )
    python_code: str | None = Field(
        default=None,
        description="A self-contained sdmx1 snippet that reproduces the queries, when available.",
    )
    candidate_datasets: list[DataSetChoice] = Field(
        default_factory=list,
        description="Datasets to choose from for the dataset_selection_required outcome.",
    )
    missing_dimensions: MissingDimensionsInfo | None = Field(
        default=None,
        description="Required dimensions to specify for the missing_dimensions outcome.",
    )
    message: str | None = Field(
        default=None,
        description="Human-readable message, e.g. explaining why no data is available.",
    )
    tools: DataQueryToolsInfo = Field(description="Companion MCP tools for these queries.")
    version: int = Field(default=2, description="Schema version of this structured content.")


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
    """One available glossary term."""

    term: str = Field(
        description="The glossary term. Use this exact value as the id when requesting its "
        "definition via the term-definitions tool."
    )
    domain: str | None = Field(default=None, description="Domain the term belongs to, if exposed.")
    source: str | None = Field(default=None, description="Source of the term, if exposed.")


class AvailableTermsStructuredContent(BaseYamlModel):
    """MCP structured content for the available-glossary-terms tool: the terms as records so a
    caller can pick exact ids to request definitions for."""

    terms: list[GlossaryTermRecord] = Field(
        default_factory=list, description="The available glossary terms."
    )
    count: int = Field(description="Number of available glossary terms.")


class GlossaryDefinitionRecord(BaseYamlModel):
    """A definition lookup result for a single requested term."""

    term: str = Field(description="The requested term.")
    found: bool = Field(description="Whether the term was found in the glossary.")
    domain: str | None = Field(default=None, description="Domain of the term, when found.")
    source: str | None = Field(default=None, description="Source of the term, when found.")
    definition: str | None = Field(default=None, description="The term's definition, when found.")


class TermDefinitionsStructuredContent(BaseYamlModel):
    """MCP structured content for the term-definitions tool: one record per requested term, each
    flagging whether it was found so a caller need not parse the prose."""

    definitions: list[GlossaryDefinitionRecord] = Field(
        default_factory=list, description="One entry per requested term."
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
    type: str | None = Field(default=None, description="The component's data type, if known.")
    description: str | None = Field(
        default=None, description="The component's description, if available."
    )
    total_values: int | None = Field(
        default=None,
        description="Total number of available values, for a categorical dimension.",
    )
    sample_values: list[DatasetValueRecord] | None = Field(
        default=None,
        description="A sample of the dimension's available values (up to 10), for a categorical "
        "dimension. When total_values exceeds the sample size this is not the full list.",
    )


class ProviderAgencyRecord(BaseYamlModel):
    """One agency behind a dataset's provider."""

    model_config = ConfigDict(serialize_by_alias=True)

    id: str = Field(description="Agency id.")
    name: str = Field(description="Agency name.")


class DatasetStructureStructuredContent(BaseYamlModel):
    """MCP structured content for the dataset-structure tool: the dataset's metadata plus its
    dimensions and attributes, with a bounded sample of each dimension's values. Optional fields
    are omitted when unknown."""

    model_config = ConfigDict(serialize_by_alias=True)

    dataset_id: str = Field(description="The requested dataset URN (source id).")
    found: bool = Field(description="Whether a dataset with that URN was found.")
    name: str | None = Field(default=None, description="Dataset name, when found.")
    description: str | None = Field(default=None, description="Dataset description, if available.")
    provider: str | None = Field(default=None, description="Provider name, if known.")
    last_updated: str | None = Field(
        default=None, description="Date the dataset was last updated (ISO 8601), if known."
    )
    url: str | None = Field(default=None, description="Link to the dataset, when available.")
    provider_agencies: list[ProviderAgencyRecord] | None = Field(
        default=None,
        description="Agencies behind the provider, when the dataset aggregates several.",
    )
    dimensions: list[DatasetComponentRecord] = Field(
        default_factory=list, description="The dataset's dimensions."
    )
    attributes: list[DatasetComponentRecord] = Field(
        default_factory=list, description="The dataset's attributes."
    )
