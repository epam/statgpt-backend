from langchain_core.runnables import Runnable, RunnablePassthrough

from common.auth.auth_context import AuthContext
from common.data.base import DataSet, DataSetQuery
from statgpt.chains.data_query.parameters import DataQueryParameters
from statgpt.chains.parameters import ChainParameters
from statgpt.schemas.query_builder import DataSetChoice

from .utils import format_dataset_queries


class MultipleDatasetsChain:

    def __init__(self, agent_only_message: str | None = None):
        self._agent_only_message = agent_only_message

    async def _get_datasets_list(self, inputs: dict) -> str:
        auth_context: AuthContext = inputs["auth_context"]
        datasets_dict: dict[str, DataSet] = inputs["datasets_dict"]
        dataset_queries: dict[str, DataSetQuery] = inputs["dataset_queries"]

        return await format_dataset_queries(
            auth_context,
            dataset_queries,
            datasets_dict,
            include_missing_dimensions=False,
            include_default_queries=False,
            include_auto_selects=False,
            print_is_official=True,
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
        if self._agent_only_message:
            content += f"\n\n{self._agent_only_message}"
        return content

    def _get_dataset_choices(self, inputs: dict) -> list[DataSetChoice]:
        datasets_dict: dict[str, DataSet] = inputs["datasets_dict"]
        dataset_queries: dict[str, DataSetQuery] = inputs["dataset_queries"]

        dataset_choices = []
        for dataset_id, dataset in datasets_dict.items():
            if dataset_id not in dataset_queries:
                continue
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
                # DataQueryParameters.DATASET_CHOICES: self._get_dataset_choices,
            }
        )
