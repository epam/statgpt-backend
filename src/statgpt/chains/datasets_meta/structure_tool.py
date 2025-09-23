from pydantic import Field

from common.schemas import ChannelConfig
from common.schemas import DatasetStructureTool as DatasetStructureToolConfig
from common.schemas import ToolTypes
from common.schemas.enums import AvailableDatasetsVersion
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.tools import StatGptTool, ToolArgs
from statgpt.chains.utils import dataset_utils
from statgpt.schemas import ToolArtifact, ToolMessageState
from statgpt.utils.formatters import DetailedDatasetFormatter

from ._utils import _create_formatter_config


class DatasetStructureArgs(ToolArgs):
    dataset_id: str = Field(
        description="Dataset ID (URN) in the format 'agency_id:resource_id(version)'."
    )


class DatasetStructureTool(
    StatGptTool[DatasetStructureToolConfig], tool_type=ToolTypes.DATASET_STRUCTURE
):
    def __init__(
        self, tool_config: DatasetStructureToolConfig, channel_config: ChannelConfig, **kwargs
    ):
        super().__init__(tool_config, channel_config, **kwargs)
        self._dataset_formatter_config = _create_formatter_config(
            AvailableDatasetsVersion.full, channel_config.locale
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
            target.append_content(response)
            return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))

        formatter = DetailedDatasetFormatter(
            self._dataset_formatter_config, auth_context=auth_context
        )
        response = await formatter.format(dataset)
        target.append_content(response)

        return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))
