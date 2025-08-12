from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from common import utils
from common.config import LLMModelsConfig
from common.utils.models import get_chat_model
from statgpt.schemas.query_builder import DateTimeQueryResponse


class DateTimeDimensionChain:
    _system_prompt: str

    def __init__(self, system_prompt: str):
        self._system_prompt = system_prompt

    def create_chain(self, api_key: str) -> Runnable:
        # used only to generate format instructions
        tmp_parser = PydanticOutputParser(pydantic_object=DateTimeQueryResponse)
        today_str = utils.get_ts_now_str(ts_format='%Y-%m-%d')
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("human", "{query}"),
            ],
        ).partial(format_instructions=tmp_parser.get_format_instructions(), current_date=today_str)

        llm = get_chat_model(
            api_key=api_key,
            # model=LLMModelsConfig.GPT_4_TURBO_2024_04_09,
            model=LLMModelsConfig.GPT_4_O_2024_05_13,  # seems to work better than other models
            # model=LLMModelsConfig.GPT_4_O_2024_08_06,
            temperature=0.0,
        ).with_structured_output(schema=DateTimeQueryResponse, method='json_mode')

        # TODO: add grounding to ensure format

        return prompt_template | llm
