from pydantic import ConfigDict, Field

from statgpt.app.schemas.data_query_outcome import (
    DataQueryStatus,
    DataSetChoice,
    MissingDimensionsInfo,
)
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.common.schemas.base import BaseYamlModel


class TextToolStructuredContent(BaseYamlModel):
    """MCP structured content for tools whose result is a single text/Markdown rendering.

    These tools (glossary, publications, web search, plain content, ...) have no richer
    machine-readable shape than their conversational answer, so the structured content mirrors
    that text. This still gives the platform a typed, inspectable payload — a declared field with
    a known type, not a bare ``object`` — satisfying the output-schema contract without inventing
    structure the tool does not actually produce. It is the default output model for a tool that
    does not override ``get_mcp_output_model``.
    """

    text: str = Field(
        description="The tool's text/Markdown response, identical to the text content block."
    )


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


class PublicationTypeRecord(BaseYamlModel):
    """One publication type the channel exposes."""

    name: str = Field(
        description="The publication type name. Use this exact value when querying publications."
    )
    description: str | None = Field(default=None, description="What this publication type covers.")


class AvailablePublicationsStructuredContent(BaseYamlModel):
    """MCP structured content for the available-publications tool: the publication types as
    records so a caller can pick exact names to query."""

    publication_types: list[PublicationTypeRecord] = Field(
        default_factory=list, description="The available publication types."
    )
    count: int = Field(description="Number of available publication types.")


class DatasetRecord(BaseYamlModel):
    """One dataset the channel exposes."""

    id: str = Field(
        description="Dataset URN (source id), e.g. 'IMF:CPI(1.0.0)'. Stable identifier to pass to "
        "the dataset-structure and data-query tools."
    )
    name: str = Field(description="Human-readable dataset name.")
    url: str | None = Field(default=None, description="Link to the dataset, if available.")


class AvailableDatasetsStructuredContent(BaseYamlModel):
    """MCP structured content for the available-datasets and datasets-metadata tools: the datasets
    as records with stable URNs so a caller can reference them in follow-up tool calls."""

    datasets: list[DatasetRecord] = Field(
        default_factory=list, description="The datasets, one record each."
    )
    count: int = Field(description="Number of datasets.")


class DatasetComponentRecord(BaseYamlModel):
    """A dimension or attribute of a dataset."""

    id: str = Field(description="The component's entity id (e.g. 'REF_AREA').")
    name: str = Field(description="Human-readable component name.")


class DatasetStructureStructuredContent(BaseYamlModel):
    """MCP structured content for the dataset-structure tool: the dataset's dimensions and
    attributes as records. Sample values are intentionally omitted (they live in the text and can
    be large); the text remains the source for values."""

    dataset_id: str = Field(description="The requested dataset URN (source id).")
    found: bool = Field(description="Whether a dataset with that URN was found.")
    name: str | None = Field(default=None, description="Dataset name, when found.")
    url: str | None = Field(default=None, description="Link to the dataset, when available.")
    dimensions: list[DatasetComponentRecord] = Field(
        default_factory=list, description="The dataset's dimensions."
    )
    attributes: list[DatasetComponentRecord] = Field(
        default_factory=list, description="The dataset's attributes."
    )
