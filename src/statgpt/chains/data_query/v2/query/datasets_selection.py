from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from common.config import logger
from common.data.base import DataSet
from common.schemas import LLMModelConfig
from common.schemas.base import SystemUserPrompt
from common.schemas.enums import LocaleEnum
from common.settings.dial import dial_settings
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters
from statgpt.schemas.query_builder import DataSetsSelectionResponse
from statgpt.utils.formatters import DatasetFormatterConfig, DatasetsListFormatter


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
        datasets_dict_indexed: dict[str, DataSet] = inputs["datasets_dict_indexed"]
        auth_context = ChainParameters.get_auth_context(inputs)

        datasets_list = await DatasetsListFormatter(
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
        ).format(list(datasets_dict_indexed.values()), sort_by_id=True)

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
        ).with_structured_output(DataSetsSelectionResponse, method='json_schema')
        logger.info(
            f"{self.__class__.__name__} using LLM model: {self._llm_model_config.deployment.deployment_id}"
        )

        chain = prompt_template | llm
        return chain
