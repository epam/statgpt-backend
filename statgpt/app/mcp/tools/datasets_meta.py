import asyncio
from collections import defaultdict
from typing import Any

from dateutil.parser import ParserError, parse
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import PrivateAttr

from statgpt.app.chains.datasets_meta.availability_tool import (
    AvailabilityPayload,
    AvailabilityQueryArgs,
    build_availability_payload,
    build_availability_query,
    unknown_dimension_ids,
    unknown_dimensions_message,
    valid_dimension_ids,
)
from statgpt.app.chains.datasets_meta.available_datasets_tool import AvailableDatasetsRunner
from statgpt.app.chains.datasets_meta.structure_tool import DatasetStructureArgs
from statgpt.app.chains.tools import ToolArgs
from statgpt.app.chains.utils import dataset_utils
from statgpt.app.schemas.mcp import (
    AvailabilityDimensionRecord,
    AvailabilityStructuredContent,
    AvailabilityValueRecord,
    AvailableDatasetsStructuredContent,
    DatasetComponentRecord,
    DatasetRecord,
    DatasetStructureStructuredContent,
    DatasetValueRecord,
    ProviderRecord,
    TimeCoverageRecord,
)
from statgpt.app.utils.formatters.dataset_detailed import sample_component_values
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import Attribute, CategoricalDimension, DataSet, Dimension
from statgpt.common.schemas import AvailabilityQueryTool as AvailabilityQueryToolConfig
from statgpt.common.schemas import AvailableDatasetsTool as AvailableDatasetsToolConfig
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas import DatasetStructureTool as DatasetStructureToolConfig
from statgpt.common.schemas import ToolTypes

from .base import StatGptMcpTool

# ~~~~~~~~~~~~~ structured content builders ~~~~~~~~~~~~~


async def _dataset_last_updated(dataset: DataSet, auth_context: AuthContext) -> str | None:
    """The dataset's last-updated date as an ISO 8601 date, from the source when known, otherwise
    parsed from the citation's free-text value (the way the SDMX dataset resolves `updated_at`).
    `None` when neither yields a date, so the field is never populated with unparsed text."""
    if updated_at := await dataset.updated_at(auth_context):
        return updated_at.date().isoformat()
    citation = dataset.config.citation
    if citation is None or not citation.last_updated:
        return None
    try:
        return parse(citation.last_updated).date().isoformat()
    except (ParserError, OverflowError):
        return None


async def datasets_to_structured_content(
    datasets: list[DataSet],
    auth_context: AuthContext,
    indicator_counts: dict[str, int] | None,
) -> AvailableDatasetsStructuredContent:
    """Build the MCP structured content for the available-datasets tool: each dataset as a record
    keyed by its stable URN (source id), the distinct providers with their dataset counts, and
    channel-wide totals."""
    last_updated = await asyncio.gather(
        *(_dataset_last_updated(ds, auth_context) for ds in datasets)
    )

    records: list[DatasetRecord] = []
    provider_counts: dict[str, int] = defaultdict(int)
    agencies: set[str] = set()
    for dataset, dataset_last_updated in zip(datasets, last_updated):
        citation = dataset.config.citation
        description = (
            citation.description if citation and citation.description else dataset.description
        )
        provider = citation.provider if citation else None
        if provider:
            provider_counts[provider] += 1
        if citation and (agency_names := citation.provider_agency_names_with_fallback_to_provider):
            agencies.update(agency_names)
        records.append(
            DatasetRecord(
                id=dataset.source_id,
                name=dataset.name,
                description=description,
                provider=provider,
                last_updated=dataset_last_updated,
                url=dataset.dataset_url,
                number_of_indicators=(
                    indicator_counts.get(dataset.entity_id) if indicator_counts else None
                ),
            )
        )

    providers = [
        ProviderRecord(name=name, dataset_count=count)
        for name, count in sorted(provider_counts.items())
    ]
    total_indicators = sum(indicator_counts.values()) if indicator_counts else None

    return AvailableDatasetsStructuredContent(
        providers=providers,
        datasets=records,
        total_datasets=len(datasets),
        total_indicators=total_indicators,
        total_agencies=len(agencies),
    )


async def dataset_structure_to_structured_content(
    dataset: DataSet, auth_context: AuthContext
) -> DatasetStructureStructuredContent:
    """Build the MCP structured content for the dataset-structure tool: the dataset's identity and
    its components. The provenance the text rendering carries (description, provider, link) is left
    out - a caller that needs it has it from the available-datasets tool."""
    return DatasetStructureStructuredContent(
        dataset_id=dataset.source_id,
        name=dataset.name,
        last_updated=await _dataset_last_updated(dataset, auth_context),
        dimensions=[_component_record(dim) for dim in dataset.dimensions()],
        attributes=[_component_record(attr) for attr in dataset.attributes()],
    )


def availability_payload_to_structured_content(
    payload: AvailabilityPayload,
) -> AvailabilityStructuredContent:
    """Adapt the availability tool's payload to MCP structured content: the dimensions dict (keyed
    by id) becomes a list of records carrying their id, matching the other datasets-meta tools."""
    dimensions = [
        AvailabilityDimensionRecord(
            id=dimension_id,
            name=coverage.name,
            total_available=coverage.total_available,
            returned=coverage.returned,
            truncated=coverage.truncated,
            values=[
                AvailabilityValueRecord(id=value.id, name=value.name) for value in coverage.values
            ],
        )
        for dimension_id, coverage in payload.dimensions.items()
    ]
    time_coverage = None
    if payload.time_coverage is not None:
        time_coverage = TimeCoverageRecord(
            dimension_id=payload.time_coverage.dimension_id,
            name=payload.time_coverage.name,
            start=payload.time_coverage.start,
            end=payload.time_coverage.end,
        )
    return AvailabilityStructuredContent(
        dataset_id=payload.dataset_id,
        found=True,
        dimensions=dimensions,
        time_coverage=time_coverage,
    )


def _dataset_not_found_error(dataset_id: str, channel_config: ChannelConfig) -> ToolError:
    """The error raised for an unknown dataset id, pointing at the tool that lists the valid ones
    when the channel exposes it."""
    available_datasets = channel_config.available_datasets
    if available_datasets is not None and available_datasets.enabled:
        tool_name = channel_config.mcp.tool_name_prefix + available_datasets.effective_mcp_name
        hint = f"Use the {tool_name} tool to list the available datasets and their exact ids."
    else:
        hint = "Ids are dataset URNs in the format 'agency_id:resource_id(version)'."
    return ToolError(f"Dataset with ID '{dataset_id}' not found. {hint}")


def _component_record(component: Dimension | Attribute) -> DatasetComponentRecord:
    if isinstance(component, Dimension):
        type_ = component.dimension_type.value
    else:
        type_ = component.attribute_type.value

    total_values: int | None = None
    sample_values: list[DatasetValueRecord] | None = None
    if isinstance(component, CategoricalDimension):
        values = component.available_values
        total_values = len(values)
        # Same sampling as the text rendering: everything when it fits, a random sample otherwise.
        sample_values = [
            DatasetValueRecord(id=value.query_id, name=value.name)
            for value in sample_component_values(values)
        ]

    return DatasetComponentRecord(
        id=component.entity_id,
        name=component.name,
        type=type_,
        description=component.description,
        total_values=total_values,
        sample_values=sample_values,
    )


# ~~~~~~~~~~~~~ MCP interfaces ~~~~~~~~~~~~~


class AvailableDatasetsMcpTool(
    StatGptMcpTool[AvailableDatasetsToolConfig, ToolArgs], tool_type=ToolTypes.AVAILABLE_DATASETS
):
    """Structured-only: the complete result lives in `structuredContent`, so no text block."""

    _runner: AvailableDatasetsRunner = PrivateAttr()

    def __init__(
        self,
        tool_config: AvailableDatasetsToolConfig,
        channel_config: ChannelConfig,
        inputs: dict[str, Any],
        auth_context: AuthContext,
        **kwargs: Any,
    ):
        super().__init__(tool_config, channel_config, inputs, auth_context, **kwargs)
        self._runner = AvailableDatasetsRunner(tool_config.details)

    @classmethod
    def get_output_model(cls) -> type[AvailableDatasetsStructuredContent]:
        return AvailableDatasetsStructuredContent

    async def _execute(self, args: ToolArgs) -> ToolResult:
        outcome = await self._runner.run(args.inputs)
        return self._structured_only(
            await datasets_to_structured_content(
                outcome.datasets, self._auth_context, outcome.indicator_counts
            )
        )


class DatasetStructureMcpTool(
    StatGptMcpTool[DatasetStructureToolConfig, DatasetStructureArgs],
    tool_type=ToolTypes.DATASET_STRUCTURE,
):
    """Structured-only: the complete result lives in `structuredContent`, so no text block."""

    @classmethod
    def get_args_schema(cls, tool_config: DatasetStructureToolConfig) -> type[DatasetStructureArgs]:
        return DatasetStructureArgs

    @classmethod
    def get_output_model(cls) -> type[DatasetStructureStructuredContent]:
        return DatasetStructureStructuredContent

    async def _execute(self, args: DatasetStructureArgs) -> ToolResult:
        dataset = await dataset_utils.get_dataset_by_source_id(args.inputs, args.dataset_id)
        if dataset is None:
            raise _dataset_not_found_error(args.dataset_id, self._channel_config)
        return self._structured_only(
            await dataset_structure_to_structured_content(dataset, self._auth_context)
        )


class AvailabilityQueryMcpTool(
    StatGptMcpTool[AvailabilityQueryToolConfig, AvailabilityQueryArgs],
    tool_type=ToolTypes.AVAILABILITY_QUERY,
):
    """Structured-only: the complete result lives in `structuredContent`, so no text block."""

    @classmethod
    def get_args_schema(
        cls, tool_config: AvailabilityQueryToolConfig
    ) -> type[AvailabilityQueryArgs]:
        return AvailabilityQueryArgs

    @classmethod
    def get_output_model(cls) -> type[AvailabilityStructuredContent]:
        return AvailabilityStructuredContent

    async def _execute(self, args: AvailabilityQueryArgs) -> ToolResult:
        dataset = await dataset_utils.get_dataset_by_source_id(args.inputs, args.dataset_id)
        if dataset is None:
            return self._structured_only(
                AvailabilityStructuredContent(dataset_id=args.dataset_id, found=False)
            )

        valid_ids = valid_dimension_ids(dataset)
        unknown = unknown_dimension_ids(valid_ids, args.partial_query, args.include_dimensions)
        if unknown:
            # A caller error the model can fix: surface the helpful message as a ToolError rather
            # than a found result, so the invalid ids are not silently ignored.
            raise ToolError(unknown_dimensions_message(unknown, sorted(valid_ids)))

        query = build_availability_query(args.partial_query)
        result = await dataset.availability_query(query, self._auth_context)

        payload = build_availability_payload(
            dataset,
            result,
            self._tool_config.details.hard_limit,
            args.codes_per_dimension,
            set(args.include_dimensions) if args.include_dimensions else None,
        )
        return self._structured_only(availability_payload_to_structured_content(payload))
