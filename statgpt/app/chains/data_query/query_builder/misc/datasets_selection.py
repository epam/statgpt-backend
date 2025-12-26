from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.schemas.query_builder import (
    DataSetsSelectionChainResponse,
    DataSetsSelectionLLMResponse,
)
from statgpt.app.services.chat_facade import VersionedDataSet
from statgpt.app.utils.formatters import DatasetFormatterConfig, IndexedDatasetsListFormatter
from statgpt.common.config import logger
from statgpt.common.schemas import LLMModelConfig
from statgpt.common.schemas.base import SystemUserPrompt
from statgpt.common.schemas.enums import LocaleEnum
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils.models import get_chat_model


class DataSetsSelectionChain:
    def __init__(
        self,
        llm_model_config: LLMModelConfig,
        system_user_prompt: SystemUserPrompt,
        llm_api_base: str | None = None,
    ):
        self._system_prompt = system_user_prompt.system_message
        self._user_prompt = system_user_prompt.user_message
        self._llm_api_base = llm_api_base or dial_settings.url
        self._llm_model_config = llm_model_config

    async def create_chain(self, inputs: dict) -> Runnable:
        versioned_datasets_dict: dict[str, VersionedDataSet] = inputs["versioned_datasets_dict"]
        auth_context = ChainParameters.get_auth_context(inputs)

        ordered_datasets = {
            i: d.data for i, d in enumerate(versioned_datasets_dict.values(), start=1)
        }

        formatter = IndexedDatasetsListFormatter(
            DatasetFormatterConfig(
                # here we want to pass both entity_id and source_id to LLM prompt.
                # LLM should treat entity_id as the main ID and source_id as the secondary ID.
                locale=LocaleEnum.EN,
                add_entity_id=True,
                entity_id_name='ID',
                add_source_id=True,
                source_id_name='Source ID',
                highlight_name_in_bold=False,
            ),
            auth_context=auth_context,
        )
        datasets_list = await formatter.format(ordered_datasets)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("human", self._user_prompt),
            ],
        ).partial(datasets_list=datasets_list)

        llm = get_chat_model(
            api_key=auth_context.api_key,
            azure_endpoint=self._llm_api_base,
            model_config=self._llm_model_config,
        ).with_structured_output(DataSetsSelectionLLMResponse, method='json_schema')
        logger.info(
            f"{self.__class__.__name__} using LLM model: {self._llm_model_config.deployment.deployment_id}"
        )

        def _postprocess_llm_response(
            llm_response: DataSetsSelectionLLMResponse,
        ) -> DataSetsSelectionChainResponse:
            """Convert LLM response to chain response for backward compatibility."""
            dataset_ids = set()
            for idx in llm_response.dataset_indexes:
                if dataset := ordered_datasets.get(idx):
                    dataset_ids.add(dataset.entity_id)

            return DataSetsSelectionChainResponse(
                dataset_ids=list(dataset_ids),
                rewritten_query=llm_response.rewritten_query,
            )

        chain = prompt_template | llm | _postprocess_llm_response
        return chain
