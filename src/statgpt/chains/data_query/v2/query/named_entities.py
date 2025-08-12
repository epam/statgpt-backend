from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from common.config import DialConfig, LLMModelsConfig
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters
from statgpt.schemas.query_builder import NamedEntitiesResponse


class NamedEntitiesChain:
    _system_prompt: str

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

    @classmethod
    async def get_entity_types(cls, inputs: dict) -> str:
        data_service = ChainParameters.get_data_service(inputs)
        named_entity_types = data_service.get_named_entity_types()
        return ", ".join(named_entity_types)

    def create_chain(self, api_key: str) -> Runnable:
        parser = PydanticOutputParser(pydantic_object=NamedEntitiesResponse)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("human", "{normalized_query}"),
            ],
        ).partial(format_instructions=parser.get_format_instructions())

        return (
            RunnablePassthrough.assign(entity_types=self.get_entity_types)
            | prompt_template
            | get_chat_model(
                api_key=api_key,
                model=self._llm_model_name,
                temperature=self._llm_temperature,
                azure_endpoint=self._llm_api_base,
            )
            | parser
        )
