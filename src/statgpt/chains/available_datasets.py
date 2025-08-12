from common.auth.auth_context import AuthContext
from common.schemas import AvailableDatasetsTool as AvailableDatasetsToolConfig
from common.schemas import ToolTypes
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.tools import StatGptTool
from statgpt.schemas import ToolArtifact, ToolMessageState
from statgpt.utils.dataset_formatter import (
    CitationFormatterConfig,
    DatasetFormatterConfig,
    DatasetListFormatter,
)


class AvailableDatasetsTool(
    StatGptTool[AvailableDatasetsToolConfig], tool_type=ToolTypes.AVAILABLE_DATASETS
):
    async def _arun(self, inputs: dict) -> tuple[str, ToolArtifact]:
        data_service = ChainParameters.get_data_service(inputs)
        auth_context = ChainParameters.get_auth_context(inputs)
        datasets = await data_service.list_available_datasets(auth_context)
        response = await self._get_all_datasets_as_md_str(
            datasets, add_entity_id=True, auth_context=auth_context
        )

        target = ChainParameters.get_target(inputs)
        target.append_content(response)

        return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))

    @staticmethod
    async def _get_all_datasets_as_md_str(
        datasets: list,
        add_entity_id: bool,
        auth_context: AuthContext,
        official_dataset_label: str | None = None,
    ) -> str:
        """Not sure if this method is still needed."""
        return await DatasetListFormatter(
            DatasetFormatterConfig(
                source_id_name='Source ID' if add_entity_id else 'ID',
                add_entity_id=add_entity_id,
                entity_id_name='ID',
                use_description=True,
                citation=CitationFormatterConfig(
                    as_md_list=True,
                    n_tabs=1,
                    use_provider=True,
                    use_last_updated=True,
                    use_url=True,
                ),
                highlight_name_in_bold=False,
                official_dataset_label=official_dataset_label,
            ),
            auth_context,
        ).format(datasets, sort_by_id=False)
