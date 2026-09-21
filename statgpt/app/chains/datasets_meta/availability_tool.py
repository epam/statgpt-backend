import json
from typing import Any

from pydantic import Field

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


class AvailabilityQueryTool(
    StatGptTool[AvailabilityQueryToolConfig], tool_type=ToolTypes.AVAILABILITY_QUERY
):
    @classmethod
    def get_args_schema(
        cls, tool_config: AvailabilityQueryToolConfig
    ) -> type[AvailabilityQueryArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return AvailabilityQueryArgs

    def _resolve_limit(self, requested: int | None) -> int:
        """Clamp the caller's per-dimension limit to the configured hard ceiling."""
        hard_limit = self._tool_config.details.hard_limit
        if requested is None:
            return hard_limit
        return min(requested, hard_limit)

    def _build_dimension_entry(
        self, dataset: DataSet, dimension_id: str, codes: list[str], limit: int
    ) -> dict[str, Any]:
        dimension = dataset.dimension(dimension_id)
        shown = codes[:limit]
        values = [
            {
                "id": code,
                "name": (
                    dimension.name_by_query_id(code)
                    if isinstance(dimension, CategoricalDimension)
                    else None
                ),
            }
            for code in shown
        ]
        return {
            "name": dimension.name,
            "total_available": len(codes),
            "returned": len(values),
            "truncated": len(codes) > len(values),
            "values": values,
        }

    def _build_payload(
        self,
        dataset: DataSet,
        result: DataSetAvailabilityQuery,
        codes_per_dimension: int | None,
        include: set[str] | None,
    ) -> dict[str, Any]:
        limit = self._resolve_limit(codes_per_dimension)
        dimensions: dict[str, Any] = {}
        for dimension_id, dim_query in result.dimensions_queries_dict.items():
            if include is not None and dimension_id not in include:
                continue
            dimensions[dimension_id] = self._build_dimension_entry(
                dataset, dimension_id, dim_query.values, limit
            )

        payload: dict[str, Any] = {"dataset_id": dataset.source_id, "dimensions": dimensions}

        if result.time_period_start or result.time_period_end:
            time_dimension = dataset.get_time_dimension()
            if include is None or time_dimension.entity_id in include:
                payload["time_coverage"] = {
                    "dimension_id": time_dimension.entity_id,
                    "name": time_dimension.name,
                    "start": result.time_period_start,
                    "end": result.time_period_end,
                }
        return payload

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

        valid_ids = {dimension.entity_id for dimension in dataset.dimensions()}
        requested_ids = list(partial_query.keys()) + (include_dimensions or [])
        unknown = sorted({dim_id for dim_id in requested_ids if dim_id not in valid_ids})
        if unknown:
            response = unknown_dimensions_message(unknown, sorted(valid_ids))
            if target:
                target.append_content(response)
            return response, artifact

        query = DataSetAvailabilityQuery.from_dimension_queries_list(
            [
                DimensionQuery(dimension_id=dim_id, values=codes, operator=QueryOperator.IN)
                for dim_id, codes in partial_query.items()
            ]
        )
        result = await dataset.availability_query(query, auth_context)

        payload = self._build_payload(
            dataset,
            result,
            codes_per_dimension,
            set(include_dimensions) if include_dimensions else None,
        )
        response = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        if target:
            target.append_content(f"```json\n{response}\n```")

        return response, artifact
