from dataclasses import dataclass

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import StatGptTool
from statgpt.app.schemas import ToolArtifact, ToolMessageState
from statgpt.app.utils.formatters import DatasetsListFormatter
from statgpt.common.data.base import DataSet
from statgpt.common.schemas import AvailableDatasetsTool as AvailableDatasetsToolConfig
from statgpt.common.schemas import ChannelConfig, ToolTypes
from statgpt.common.schemas.tool_details import AvailableDatasetsDetails

from ._utils import _create_formatter_config


@dataclass(frozen=True)
class AvailableDatasetsOutcome:
    datasets: list[DataSet]
    indicator_counts: dict[str, int] | None
    """Indicator count per dataset entity id, when the tool is configured to compute it."""


class AvailableDatasetsRunner:
    """Lists the datasets available to the caller; shared by the LangChain and MCP interfaces."""

    def __init__(self, details: AvailableDatasetsDetails):
        self._details = details

    async def run(self, inputs: dict) -> AvailableDatasetsOutcome:
        data_service = ChainParameters.get_data_service(inputs)
        auth_context = ChainParameters.get_auth_context(inputs)

        versioned_datasets = await data_service.list_available_datasets(auth_context)
        datasets = [ds.data for ds in versioned_datasets]

        indicator_counts: dict[str, int] | None = None
        if self._details.include_indicator_count:
            indicator_counts = await data_service.get_indicator_counts(
                auth_context, versioned_datasets
            )
        return AvailableDatasetsOutcome(datasets=datasets, indicator_counts=indicator_counts)


class AvailableDatasetsTool(
    StatGptTool[AvailableDatasetsToolConfig], tool_type=ToolTypes.AVAILABLE_DATASETS
):

    def __init__(
        self, tool_config: AvailableDatasetsToolConfig, channel_config: ChannelConfig, **kwargs
    ):
        super().__init__(tool_config, channel_config, **kwargs)
        self._runner = AvailableDatasetsRunner(tool_config.details)
        self._dataset_formatter_config = _create_formatter_config(
            version=tool_config.details.version,
            locale=channel_config.locale,
            stats_header_format=tool_config.details.stats_header_format,
        )

    async def _arun(self, inputs: dict) -> tuple[str, ToolArtifact]:
        auth_context = ChainParameters.get_auth_context(inputs)
        outcome = await self._runner.run(inputs)

        formatter = DatasetsListFormatter(self._dataset_formatter_config, auth_context=auth_context)
        response = await formatter.format(
            outcome.datasets,
            sort_by_name=True,
            add_stats=True,
            group_by_provider=True,
            indicator_counts=outcome.indicator_counts,
        )

        target = ChainParameters.get_target(inputs)
        if target:
            target.append_content(response)

        return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))
