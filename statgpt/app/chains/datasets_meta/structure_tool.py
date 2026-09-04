from pydantic import Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import StatGptTool, ToolArgs
from statgpt.app.chains.utils import dataset_utils
from statgpt.app.schemas import ToolArtifact, ToolMessageState
from statgpt.app.utils.formatters import (
    CitationFormatterConfig,
    DatasetFormatterConfig,
    DetailedDatasetFormatter,
)
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas import DatasetStructureTool as DatasetStructureToolConfig
from statgpt.common.schemas import ToolTypes


class DatasetStructureArgs(ToolArgs):
    dataset_id: str = Field(
        description="Dataset ID (URN) in the format 'agency_id:resource_id(version)'."
    )


def dataset_not_found_message(dataset_id: str) -> str:
    return (
        f"Dataset with ID '{dataset_id}' not found among available datasets. "
        f"Please check the ID and try again."
    )


class DatasetStructureTool(
    StatGptTool[DatasetStructureToolConfig], tool_type=ToolTypes.DATASET_STRUCTURE
):
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
        artifact = ToolArtifact(state=ToolMessageState(type=self.tool_type))

        if dataset is None:
            response = dataset_not_found_message(dataset_id)
            if target:
                target.append_content(response)
            return response, artifact

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

        return response, artifact
