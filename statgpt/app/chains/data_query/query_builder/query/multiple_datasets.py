from langchain_core.runnables import Runnable, RunnablePassthrough

from statgpt.app.chains.data_query.parameters import DataQueryParameters
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.schemas.data_query_outcome import DataSetChoice
from statgpt.app.schemas.query_builder import ChainState
from statgpt.app.services.chat_facade import VersionedDataSet
from statgpt.app.utils.formatters import DatasetQueryFormatter, DatasetQueryFormatterConfig
from statgpt.common.data.base import DataSetQuery
from statgpt.common.schemas.data_query_tool import DataQueryMessages


class MultipleDatasetsChain:

    def __init__(self, messages: DataQueryMessages):
        self._messages = messages

    async def _get_datasets_list(self, inputs: dict) -> str:
        chain_state = ChainState.model_validate(inputs)
        auth_context = chain_state.auth_context
        datasets_dict = chain_state.datasets_dict
        dataset_queries = chain_state.dataset_queries

        query_formatter = DatasetQueryFormatter(
            config=DatasetQueryFormatterConfig(
                locale=chain_state.data_service.channel_config.locale,
                include_missing_dimensions=False,
                include_default_queries=False,
                include_auto_selects=False,
                include_is_official=True,
            ),
            auth_context=auth_context,
        )

        return await query_formatter.format_queries(
            dataset_queries=dataset_queries,
            datasets_dict=datasets_dict,
        )

    async def _get_response_content(self, inputs: dict) -> str:
        datasets_list = await self._get_datasets_list(inputs)
        content = f"Relevant data can be pulled from the following datasets:\n{datasets_list}"
        target = ChainParameters.get_target(inputs)
        target.append_content(content)
        content += (
            "\n\n**Important**: at that point **no data is provided either to you or to user**, only query info. "
            "You may select one of the datasets without user's input, whenever you think it's possible, "
            "or ask user to select one of the datasets to proceed with query execution. When user selected something, "
            "call the same tool mentioning the dataset name or id in the tool call arguments."
        )
        agent_only_message = self._messages.get_multiple_datasets(
            ChainParameters.get_invocation_source(inputs)
        )
        if agent_only_message:
            content += f"\n\n{agent_only_message}"
        return content

    @staticmethod
    def build_dataset_choices(
        datasets_dict: dict[str, VersionedDataSet],
        dataset_queries: dict[str, DataSetQuery],
    ) -> list[DataSetChoice]:
        """Build the list of datasets the user/agent can choose from to disambiguate a query."""
        dataset_choices = []
        for dataset_id, versioned_dataset in datasets_dict.items():
            if dataset_id not in dataset_queries:
                continue
            dataset = versioned_dataset.data
            citation = dataset.config.citation
            description = (
                citation.description if citation and citation.description else dataset.description
            )
            dataset_choices.append(
                DataSetChoice(
                    id=dataset.source_id,
                    name=dataset.name,
                    description=description,
                    is_official=dataset.config.is_official,
                )
            )
        return dataset_choices

    async def create_chain(self) -> Runnable:
        return RunnablePassthrough.assign(
            **{
                DataQueryParameters.RESPONSE_FIELD: self._get_response_content,
            }
        )
