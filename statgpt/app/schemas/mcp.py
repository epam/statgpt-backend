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
        "executed_no_data, failed and not_executed outcomes. Each query carries a stable, "
        "opaque `recordId` (and the dataflow `urn`) that a follow-up call can pass back "
        "verbatim to reference the exact same record.",
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
    version: int = Field(default=3, description="Schema version of this structured content.")


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
