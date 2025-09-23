from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from common.config import logger
from common.schemas import LLMModelConfig
from common.settings.dial import dial_settings
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters
from statgpt.schemas.query_builder import NamedEntitiesResponse


class NamedEntitiesChain:
    _system_prompt: str

    def __init__(
        self,
        llm_model_config: LLMModelConfig,
        system_prompt: str,
        llm_api_base: str | None = None,
    ):
        self._system_prompt = system_prompt
        self._llm_api_base = llm_api_base or dial_settings.url
        self._llm_model_config = llm_model_config

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

        chain = (
            RunnablePassthrough.assign(entity_types=self.get_entity_types)
            | prompt_template
            | get_chat_model(
                api_key=api_key,
                azure_endpoint=self._llm_api_base,
                model_config=self._llm_model_config,
            )
            | parser
        )
        logger.info(
            f"{self.__class__.__name__} using LLM model: {self._llm_model_config.deployment.deployment_id}"
        )
        return chain
