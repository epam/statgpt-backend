from mcp.types import ToolAnnotations
from pydantic import Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import StatGptTool, ToolArgs
from statgpt.app.chains.utils import dataset_utils
from statgpt.app.schemas import ToolArtifact, ToolMessageState
from statgpt.app.schemas.mcp import (
    DatasetComponentRecord,
    DatasetStructureStructuredContent,
    DatasetValueRecord,
    ProviderAgencyRecord,
)
from statgpt.app.utils.formatters import (
    CitationFormatterConfig,
    DatasetFormatterConfig,
    DetailedDatasetFormatter,
)
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import Attribute, CategoricalDimension, DataSet, Dimension
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas import DatasetStructureTool as DatasetStructureToolConfig
from statgpt.common.schemas import ToolTypes

_SAMPLE_VALUES_LIMIT = 10


class DatasetStructureArgs(ToolArgs):
    dataset_id: str = Field(
        description="Dataset ID (URN) in the format 'agency_id:resource_id(version)'."
    )


class DatasetStructureTool(
    StatGptTool[DatasetStructureToolConfig], tool_type=ToolTypes.DATASET_STRUCTURE
):
    @classmethod
    def get_mcp_annotations(cls) -> ToolAnnotations:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @classmethod
    def get_mcp_output_model(cls) -> type[DatasetStructureStructuredContent]:
        return DatasetStructureStructuredContent

    @classmethod
    def mcp_structured_only(cls) -> bool:
        return True

    def __init__(
        self, tool_config: DatasetStructureToolConfig, channel_config: ChannelConfig, **kwargs
    ):
        super().__init__(tool_config, channel_config, **kwargs)
        self._dataset_formatter_config = DatasetFormatterConfig(
            locale=channel_config.locale,
            add_source_id=True,
            add_entity_id=False,
            use_description=True,
            citation=CitationFormatterConfig(
                as_md_list=True,
                n_tabs=0,
                use_provider=True,
                use_last_updated=True,
                use_url=True,
                include_provider_agencies=tool_config.details.include_provider_agencies,
            ),
            highlight_name_in_bold=True,
            list_level=0,
        )

    @classmethod
    def get_args_schema(cls, tool_config: DatasetStructureToolConfig) -> type[DatasetStructureArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return DatasetStructureArgs

    async def _arun(self, inputs: dict, dataset_id: str, **kwargs) -> tuple[str, ToolArtifact]:
        dataset = await dataset_utils.get_dataset_by_source_id(inputs, dataset_id)
        target = ChainParameters.get_target(inputs)
        auth_context = ChainParameters.get_auth_context(inputs)

        if dataset is None:
            response = (
                f"Dataset with ID '{dataset_id}' not found among available datasets. "
                f"Please check the ID and try again."
            )
            if target:
                target.append_content(response)
            return response, ToolArtifact(
                state=ToolMessageState(type=self.tool_type),
                mcp_structured=DatasetStructureStructuredContent(
                    dataset_id=dataset_id, found=False
                ),
            )

        formatter = DetailedDatasetFormatter(
            self._dataset_formatter_config, auth_context=auth_context
        )
        response = await formatter.format(dataset)  # type: ignore[arg-type]

        if target:
            target.append_content(response)

        response += (
            "\n\nNote: Don't make any assumptions about the dataset beyond the provided structure information"
            ", especially regarding sample values of the datasets' dimensions."
        )

        return response, ToolArtifact(
            state=ToolMessageState(type=self.tool_type),
            mcp_structured=await self._to_structured_content(dataset, auth_context),
        )

    @classmethod
    async def _to_structured_content(
        cls, dataset: DataSet, auth_context: AuthContext
    ) -> DatasetStructureStructuredContent:
        citation = dataset.config.citation
        description = (
            citation.description if citation and citation.description else dataset.description
        )
        last_updated: str | None
        if updated_at := await dataset.updated_at(auth_context):
            last_updated = updated_at.date().isoformat()
        else:
            last_updated = citation.last_updated if citation else None
        provider_agencies = None
        if citation and citation.provider_agencies:
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
            last_updated=last_updated,
            url=dataset.dataset_url,
            provider_agencies=provider_agencies,
            dimensions=[cls._component_record(dim) for dim in dataset.dimensions()],
            attributes=[cls._component_record(attr) for attr in dataset.attributes()],
        )

    @staticmethod
    def _component_record(component: Dimension | Attribute) -> DatasetComponentRecord:
        type_ = getattr(component, "dimension_type", None) or getattr(
            component, "attribute_type", None
        )
        record = DatasetComponentRecord(
            id=component.entity_id,
            name=component.name,
            type=type_.value if type_ is not None else None,
            description=component.description,
        )
        if isinstance(component, CategoricalDimension):
            values = component.available_values
            record.total_values = len(values)
            record.sample_values = [
                DatasetValueRecord(id=value.query_id, name=value.name)
                for value in values[:_SAMPLE_VALUES_LIMIT]
            ]
        return record
