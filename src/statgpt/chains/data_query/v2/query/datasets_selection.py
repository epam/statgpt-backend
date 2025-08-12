from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from common.config import DialConfig, LLMModelsConfig
from common.data.base import DataSet
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters
from statgpt.schemas.query_builder import DataSetsSelectionResponse
from statgpt.utils.dataset_formatter import DatasetFormatterConfig, DatasetListFormatter


class DataSetsSelectionChain:
    def __init__(
        self,
        system_prompt: str,
        llm_api_base: str | None = None,
        llm_model_name: str | None = None,
        llm_temperature: float = 0.0,
    ):
        self._system_prompt = system_prompt
        self._llm_api_base = llm_api_base or DialConfig.get_url()
        self._llm_model_name = llm_model_name or LLMModelsConfig.GPT_4_TURBO_2024_04_09
        self._llm_temperature = llm_temperature

    async def create_chain(self, inputs: dict) -> Runnable:
        datasets_dict_indexed: dict[str, DataSet] = inputs["datasets_dict_indexed"]
        auth_context = ChainParameters.get_auth_context(inputs)

        datasets_list = await DatasetListFormatter(
            DatasetFormatterConfig.model_validate(
                dict(
                    # here we want to pass both entity_id and source_id to LLM prompt.
                    # LLM should treat entity_id as the main ID and source_id as the secondary ID.
                    add_entity_id=True,
                    entity_id_name='ID',
                    add_source_id=True,
                    source_id_name='Source ID',
                    highlight_name_in_bold=False,
                    sort_by_id=True,
                )
            ),
            auth_context=auth_context,
        ).format(list(datasets_dict_indexed.values()), sort_by_id=True)

        parser = PydanticOutputParser(pydantic_object=DataSetsSelectionResponse)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("human", "{normalized_query}"),
            ],
        )
        prompt_template = prompt_template.partial(
            datasets=datasets_list, format_instructions=parser.get_format_instructions()
        )

        llm = get_chat_model(
            api_key=auth_context.api_key,
            model=self._llm_model_name,
            temperature=self._llm_temperature,
            azure_endpoint=self._llm_api_base,
        ).with_structured_output(DataSetsSelectionResponse, method='json_mode')

        chain = prompt_template | llm
        return chain
