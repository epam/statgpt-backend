"""Outcome contract of the data query pipeline.

Kept as a leaf module: both the MCP wire schema (``schemas/mcp.py``) and the query builder
state (``schemas/query_builder.py``) import from here, so the wire contract doesn't drag in
the query builder runtime.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.common.schemas.base import BaseYamlModel


class DataQueryStatus(StrEnum):
    """Outcome of the data query pipeline, tagging which branch produced the response.

    Surfaced in the Data Query tool's MCP structured content so callers can act on the
    result programmatically instead of parsing the human-readable text.

    ``FAILED`` is the default: every branch is expected to override it, so a response that
    reports it either hit an error while fetching the data or never reached a branch at all.
    """

    FAILED = "failed"
    DATA_AVAILABLE = "data_available"
    NO_DATA = "no_data"
    EXECUTED_NO_DATA = "executed_no_data"
    DATASET_SELECTION_REQUIRED = "dataset_selection_required"
    INVALID_TIME_PERIOD = "invalid_time_period"
    MISSING_DIMENSIONS = "missing_dimensions"
    NOT_EXECUTED = "not_executed"


class DataSetChoice(BaseYamlModel):
    """
    Represent a dataset choice available for selection by either agent or user.
    """

    id: str = Field(description="The unique identifier of the dataset, used for selection.")
    name: str = Field(description="The human-readable name of the dataset, used for display.")
    description: str | None = Field(
        default=None,
        description="A brief description of the dataset, providing context and details.",
    )
    is_official: bool = Field(
        default=False,
        description="Indicates whether the dataset is official or not.",
    )


class DimensionValueInfo(BaseYamlModel):
    """An available value of a dimension the user can pick from."""

    id: str = Field(description="The dimension value id used in queries.")
    name: str = Field(description="The human-readable name of the value.")
    description: str | None = Field(
        default=None, description="An optional description of the value."
    )


class MissingDimensionInfo(BaseYamlModel):
    """A required dimension not yet specified, with the values available to choose from."""

    dimension_id: str = Field(description="The entity id of the missing dimension.")
    name: str = Field(description="The human-readable name of the missing dimension.")
    available_values: list[DimensionValueInfo] = Field(
        default_factory=list,
        description="Values available for this dimension given the rest of the query.",
    )


class MissingDimensionsInfo(BaseYamlModel):
    """Describes why a query is incomplete: which dimensions still need a value."""

    dataset_id: str = Field(description="The dataset the missing dimensions belong to.")
    dataset_urn: str | None = Field(
        default=None,
        description="Source id (URN) of that dataset, when known - the id a caller can query by.",
    )
    dimensions: list[MissingDimensionInfo] = Field(
        default_factory=list, description="The missing required dimensions."
    )


class InvalidPeriodInfo(BaseYamlModel):
    """Why a query's requested time period was rejected: one bound falls outside the dataset's
    available range."""

    rejected_bound: Literal["start", "end"] = Field(description="The rejected bound.")
    requested_value: str = Field(description="The value requested for the rejected bound.")
    available_start: str | None = Field(default=None, description="First available period.")
    available_end: str | None = Field(default=None, description="Last available period.")


class QueryDetails(BaseYamlModel):
    """What the MCP structured content reports about one dataset query beyond the query itself.

    Collected by the chain that renders the response, after the queries were summarized.
    """

    dataset_urn: str = Field(description="Source id (URN) of the queried dataset.")
    dataset_name: str = Field(description="Name of the queried dataset.")
    json_query: AppJsonQueryWithMetadata | None = Field(
        default=None,
        description="The constructed query, for outcomes that report queries that did not run.",
    )
    summary: str | None = Field(default=None, description="The query's short summary.")
    last_updated: str | None = Field(
        default=None, description="Date the dataset was last updated (ISO 8601), if known."
    )
    provider: str | None = Field(default=None, description="The dataset's provider, if known.")
    is_official: bool = Field(default=False, description="Whether the dataset is official.")
    dimension_names: dict[str, str] = Field(
        default_factory=dict, description="Dimension display names by entity id."
    )
    default_dimension_ids: list[str] = Field(
        default_factory=list,
        description="Dimensions filtered by a default rather than by the user's request.",
    )
    default_time_period: bool = Field(
        default=False, description="Whether the time period is a default one."
    )
    invalid_period: InvalidPeriodInfo | None = Field(
        default=None, description="Why the requested time period was rejected, if it was."
    )


class DataQueryMcpPayload(BaseModel):
    """MCP-response-only data captured for a single data query invocation.

    Lives on the in-memory ``DataQueryArtifact``, never on the persisted ``QueryBuilderAgentState``,
    so these potentially heavy payloads aren't serialized to the DIAL server or carried across turns
    (they are consumed only when building the MCP structured content).
    """

    constructed_queries: list[AppJsonQueryWithMetadata] = Field(
        default_factory=list,
        description="Queries constructed but not executed, surfaced for the not_executed outcome.",
    )
    candidate_datasets: list[DataSetChoice] = Field(
        default_factory=list,
        description="Datasets to choose from when the query matches multiple datasets.",
    )
    missing_dimensions: MissingDimensionsInfo | None = Field(
        default=None,
        description="Required dimensions the user must specify, when the query is incomplete.",
    )
    query_details: dict[str, QueryDetails] = Field(
        default_factory=dict,
        description="Per dataset id, what is reported about its query beyond the query itself.",
    )
    message: str | None = Field(
        default=None, description="Text for the calling model explaining the outcome."
    )
    executed_at: str | None = Field(
        default=None, description="When the queries were executed (ISO 8601)."
    )
