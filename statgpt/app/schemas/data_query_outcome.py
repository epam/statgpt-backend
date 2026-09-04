"""Outcome contract of the data query pipeline.

Kept as a leaf module: both the MCP wire schema (``schemas/mcp.py``) and the query builder
state (``schemas/query_builder.py``) import from here, so the wire contract doesn't drag in
the query builder runtime.
"""

from enum import StrEnum

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

    id: str = Field(
        description=(
            "Stable dataflow identifier of the dataset (SDMX short URN,"
            " 'AGENCY:RESOURCE(VERSION)'), used for selection. Opaque; pass back verbatim."
        )
    )
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
    dimensions: list[MissingDimensionInfo] = Field(
        default_factory=list, description="The missing required dimensions."
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
