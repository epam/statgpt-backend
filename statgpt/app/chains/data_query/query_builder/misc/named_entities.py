from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.schemas.query_builder import NamedEntitiesResponse
from statgpt.common.config import logger
from statgpt.common.schemas import LLMModelConfig
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils.models import get_chat_model


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

    def create_chain(self, inputs: dict) -> Runnable:
        auth_context = ChainParameters.get_auth_context(inputs)

        # ``format_instructions`` is partialed to an empty string so channel-custom
        # prompts that still reference the placeholder keep rendering; json_schema
        # structured output now enforces the shape, so the instructions are redundant.
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("human", "{normalized_query}"),
            ],
        ).partial(format_instructions="")

        llm = get_chat_model(
            api_key=auth_context.api_key,
            azure_endpoint=self._llm_api_base,
            model_config=self._llm_model_config,
        ).with_structured_output(NamedEntitiesResponse, method='json_schema')

        chain = (
            RunnablePassthrough.assign(entity_types=self.get_entity_types) | prompt_template | llm
        )
        logger.info(
            f"{self.__class__.__name__} using LLM model: {self._llm_model_config.deployment.deployment_id}"
        )
        return chain
