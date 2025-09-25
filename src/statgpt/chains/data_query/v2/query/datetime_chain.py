from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from common.config import multiline_logger as logger
from common.schemas import LLMModelConfig
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters
from statgpt.schemas.query_builder import DateTimeQueryResponse


class DateTimeDimensionChain:
    _system_prompt: str

    def __init__(self, llm_model_config: LLMModelConfig, system_prompt: str):
        self._llm_model_config = llm_model_config
        self._system_prompt = system_prompt

    def create_chain(self, api_key: str) -> Runnable:
        # used only to generate format instructions
        tmp_parser = PydanticOutputParser(pydantic_object=DateTimeQueryResponse)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("human", "{query}"),
            ],
        ).partial(format_instructions=tmp_parser.get_format_instructions())

        llm = get_chat_model(
            api_key=api_key,
            model_config=self._llm_model_config,
        ).with_structured_output(schema=DateTimeQueryResponse, method='json_mode')
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
