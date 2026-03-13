from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.schemas.query_builder import DateTimeQueryResponse
from statgpt.common.config import multiline_logger as logger
from statgpt.common.schemas import LLMModelConfig
from statgpt.common.utils.models import get_chat_model


class DateTimeDimensionChain:
    _system_prompt: str

    def __init__(self, llm_model_config: LLMModelConfig, system_prompt: str):
        self._llm_model_config = llm_model_config
        self._system_prompt = system_prompt

    def create_chain(self, inputs: dict) -> Runnable:
        auth_context = ChainParameters.get_auth_context(inputs)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("human", "{query}"),
            ],
        )

        llm = get_chat_model(
            api_key=auth_context.api_key,
            model_config=self._llm_model_config,
            streaming=False,
        ).with_structured_output(schema=DateTimeQueryResponse, method='json_schema')
        logger.info(
            f"{self.__class__.__name__} using LLM model: {self._llm_model_config.deployment.deployment_id}"
        )

        # TODO: add grounding to ensure format

        return (
            RunnablePassthrough.assign(
                current_date=lambda d: ChainParameters.get_configuration(d).get_current_date()
            )
            | prompt_template
            | llm
        )
