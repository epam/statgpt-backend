import asyncio
from collections import defaultdict
from typing import Any

from dateutil.parser import ParserError, parse
from fastmcp.tools import ToolResult
from pydantic import PrivateAttr

from statgpt.app.chains.datasets_meta.available_datasets_tool import AvailableDatasetsRunner
from statgpt.app.chains.datasets_meta.structure_tool import DatasetStructureArgs
from statgpt.app.chains.tools import ToolArgs
from statgpt.app.chains.utils import dataset_utils
from statgpt.app.schemas.mcp import (
    AvailableDatasetsStructuredContent,
    DatasetComponentRecord,
    DatasetRecord,
    DatasetStructureStructuredContent,
    DatasetValueRecord,
    ProviderAgencyRecord,
    ProviderRecord,
)
from statgpt.app.utils.formatters.dataset_detailed import sample_component_values
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import Attribute, CategoricalDimension, DataSet, Dimension
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
    dataset: DataSet, auth_context: AuthContext, *, include_provider_agencies: bool
) -> DatasetStructureStructuredContent:
    """Build the MCP structured content for the dataset-structure tool. `provider_agencies` is
    exposed only when the tool is configured to include it, mirroring the text rendering."""
    citation = dataset.config.citation
    description = citation.description if citation and citation.description else dataset.description
    provider_agencies = None
    if include_provider_agencies and citation and citation.provider_agencies:
        provider_agencies = [
            ProviderAgencyRecord(id=agency.id, name=agency.name)
            for agency in citation.provider_agencies
        ]
    return DatasetStructureStructuredContent(
        dataset_id=dataset.source_id,
        found=True,
        name=dataset.name,
        description=description,
        provider=citation.provider if citation else None,
        last_updated=await _dataset_last_updated(dataset, auth_context),
        url=dataset.dataset_url,
        provider_agencies=provider_agencies,
        dimensions=[_component_record(dim) for dim in dataset.dimensions()],
        attributes=[_component_record(attr) for attr in dataset.attributes()],
    )


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
            return self._structured_only(
                DatasetStructureStructuredContent(dataset_id=args.dataset_id, found=False)
            )
        return self._structured_only(
            await dataset_structure_to_structured_content(
                dataset,
                self._auth_context,
                include_provider_agencies=self._tool_config.details.include_provider_agencies,
            )
        )
