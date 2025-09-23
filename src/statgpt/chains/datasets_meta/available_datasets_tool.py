from common.schemas import AvailableDatasetsTool as AvailableDatasetsToolConfig
from common.schemas import ChannelConfig, ToolTypes
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.tools import StatGptTool
from statgpt.schemas import ToolArtifact, ToolMessageState
from statgpt.utils.formatters import DatasetsListFormatter

from ._utils import _create_formatter_config


class AvailableDatasetsTool(
    StatGptTool[AvailableDatasetsToolConfig], tool_type=ToolTypes.AVAILABLE_DATASETS
):

    def __init__(
        self, tool_config: AvailableDatasetsToolConfig, channel_config: ChannelConfig, **kwargs
    ):
        super().__init__(tool_config, channel_config, **kwargs)
        self._dataset_formatter_config = _create_formatter_config(
            tool_config.details.version, channel_config.locale
        )

    async def _arun(self, inputs: dict) -> tuple[str, ToolArtifact]:
        data_service = ChainParameters.get_data_service(inputs)
        auth_context = ChainParameters.get_auth_context(inputs)
        datasets = await data_service.list_available_datasets(auth_context)

        response = await DatasetsListFormatter(
            self._dataset_formatter_config,
            auth_context=auth_context,
        ).format(datasets, sort_by_name=True, add_stats=True, group_by_provider=True)

        target = ChainParameters.get_target(inputs)
        target.append_content(response)

        return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))
