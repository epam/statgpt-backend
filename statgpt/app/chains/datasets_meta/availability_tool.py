from pydantic import BaseModel, Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import StatGptTool, ToolArgs
from statgpt.app.chains.utils import dataset_utils
from statgpt.app.schemas import ToolArtifact, ToolMessageState
from statgpt.common.data.base import (
    CategoricalDimension,
    DataSet,
    DataSetAvailabilityQuery,
    DimensionQuery,
    QueryOperator,
)
from statgpt.common.schemas import AvailabilityQueryTool as AvailabilityQueryToolConfig
from statgpt.common.schemas import ToolTypes


class AvailabilityQueryArgs(ToolArgs):
    dataset_id: str = Field(
        description="Dataset ID (URN) in the format 'agency_id:resource_id(version)'."
    )
    partial_query: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Partial key mapping a dimension ID to the code IDs to fix, e.g."
            ' {"SERIES": ["51401"]}. Pass an empty object ({}) to get availability across the'
            " whole dataset."
        ),
    )
    codes_per_dimension: int | None = Field(
        default=10,
        ge=1,
        description=(
            "Maximum number of codes to return per dimension. Use null for no per-call limit"
            " (still bounded by the tool's hard limit)."
        ),
    )
    include_dimensions: list[str] | None = Field(
        default=None,
        description=(
            "Restrict the result to these dimension IDs. Use null (or omit) to include every"
            " dimension."
        ),
    )


def dataset_not_found_message(dataset_id: str) -> str:
    return (
        f"Dataset with ID '{dataset_id}' not found among available datasets. "
        f"Please check the ID and try again."
    )


def unknown_dimensions_message(unknown: list[str], valid: list[str]) -> str:
    return (
        f"Unknown dimension ID(s): {unknown}. "
        f"Valid dimension IDs for this dataset are: {valid}."
    )


class DimensionValue(BaseModel):
    id: str
    name: str | None = None


class DimensionCoverage(BaseModel):
    name: str
    total_available: int
    returned: int
    truncated: bool
    values: list[DimensionValue]


class TimeCoverage(BaseModel):
    dimension_id: str
    name: str
    start: str | None = None
    end: str | None = None


class AvailabilityPayload(BaseModel):
    dataset_id: str
    dimensions: dict[str, DimensionCoverage]
    time_coverage: TimeCoverage | None = None


def resolve_codes_limit(hard_limit: int, requested: int | None) -> int:
    """Clamp the caller's per-dimension limit to the configured hard ceiling."""
    if requested is None:
        return hard_limit
    return min(requested, hard_limit)


def valid_dimension_ids(dataset: DataSet) -> set[str]:
    return {dimension.entity_id for dimension in dataset.dimensions()}


def unknown_dimension_ids(
    valid_ids: set[str],
    partial_query: dict[str, list[str]],
    include_dimensions: list[str] | None,
) -> list[str]:
    """The requested dimension IDs (from the partial query and the include filter) that the dataset
    does not have, sorted for a stable message."""
    requested_ids = list(partial_query.keys()) + (include_dimensions or [])
    return sorted({dim_id for dim_id in requested_ids if dim_id not in valid_ids})


def build_availability_query(partial_query: dict[str, list[str]]) -> DataSetAvailabilityQuery:
    return DataSetAvailabilityQuery.from_dimension_queries_list(
        [
            DimensionQuery(dimension_id=dim_id, values=codes, operator=QueryOperator.IN)
            for dim_id, codes in partial_query.items()
        ]
    )


def _build_dimension_entry(
    dataset: DataSet, dimension_id: str, codes: list[str], limit: int
) -> DimensionCoverage:
    dimension = dataset.dimension(dimension_id)
    shown = codes[:limit]
    values = [
        DimensionValue(
            id=code,
            name=(
                dimension.name_by_query_id(code)
                if isinstance(dimension, CategoricalDimension)
                else None
            ),
        )
        for code in shown
    ]
    return DimensionCoverage(
        name=dimension.name,
        total_available=len(codes),
        returned=len(values),
        truncated=len(codes) > len(values),
        values=values,
    )


def build_availability_payload(
    dataset: DataSet,
    result: DataSetAvailabilityQuery,
    hard_limit: int,
    codes_per_dimension: int | None,
    include: set[str] | None,
) -> AvailabilityPayload:
    limit = resolve_codes_limit(hard_limit, codes_per_dimension)
    dimensions: dict[str, DimensionCoverage] = {}
    for dimension_id, dim_query in result.dimensions_queries_dict.items():
        if include is not None and dimension_id not in include:
            continue
        dimensions[dimension_id] = _build_dimension_entry(
            dataset, dimension_id, dim_query.values, limit
        )

    time_coverage: TimeCoverage | None = None
    if result.time_period_start or result.time_period_end:
        time_dimension = dataset.get_time_dimension()
        if include is None or time_dimension.entity_id in include:
            time_coverage = TimeCoverage(
                dimension_id=time_dimension.entity_id,
                name=time_dimension.name,
                start=result.time_period_start,
                end=result.time_period_end,
            )

    return AvailabilityPayload(
        dataset_id=dataset.source_id,
        dimensions=dimensions,
        time_coverage=time_coverage,
    )


class AvailabilityQueryTool(
    StatGptTool[AvailabilityQueryToolConfig], tool_type=ToolTypes.AVAILABILITY_QUERY
):
    @classmethod
    def get_args_schema(
        cls, tool_config: AvailabilityQueryToolConfig
    ) -> type[AvailabilityQueryArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return AvailabilityQueryArgs

    async def _arun(
        self,
        inputs: dict,
        dataset_id: str,
        partial_query: dict[str, list[str]] | None = None,
        codes_per_dimension: int | None = 10,
        include_dimensions: list[str] | None = None,
        **kwargs,
    ) -> tuple[str, ToolArtifact]:
        partial_query = partial_query or {}
        auth_context = ChainParameters.get_auth_context(inputs)
        target = ChainParameters.get_target(inputs)
        artifact = ToolArtifact(state=ToolMessageState(type=self.tool_type))

        dataset = await dataset_utils.get_dataset_by_source_id(inputs, dataset_id)
        if dataset is None:
            response = dataset_not_found_message(dataset_id)
            if target:
                target.append_content(response)
            return response, artifact

        valid_ids = valid_dimension_ids(dataset)
        unknown = unknown_dimension_ids(valid_ids, partial_query, include_dimensions)
        if unknown:
            response = unknown_dimensions_message(unknown, sorted(valid_ids))
            if target:
                target.append_content(response)
            return response, artifact

        query = build_availability_query(partial_query)
        result = await dataset.availability_query(query, auth_context)

        payload = build_availability_payload(
            dataset,
            result,
            self._tool_config.details.hard_limit,
            codes_per_dimension,
            set(include_dimensions) if include_dimensions else None,
        )
        exclude = {"time_coverage"} if payload.time_coverage is None else None
        response = payload.model_dump_json(exclude=exclude)

        if target:
            target.append_content(f"```json\n{response}\n```")

        return response, artifact
