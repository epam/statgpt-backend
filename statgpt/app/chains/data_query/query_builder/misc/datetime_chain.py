from enum import StrEnum

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import BaseModel, Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.schemas.query_builder import DateTimeQueryResponse
from statgpt.common.schemas import LLMModelConfig
from statgpt.common.utils.models import get_chat_model


class Intent(StrEnum):
    HISTORICAL = "historical"
    FORECAST = "forecast"
    SPANNING = "spanning"
    UNSPECIFIED = "unspecified"


class DateTimeLLMResponse(BaseModel):
    intent: Intent = Field(
        default=Intent.UNSPECIFIED,
        description=(
            'Temporal intent: "historical" (past tense, last N, dates before today), '
            '"forecast" (forecast keywords, future tense, next N, dates after today), '
            '"spanning" (present tense + time ref, range crossing today, all available data, '
            'no tense with open-ended past reference), '
            '"unspecified" (no temporal reference at all).'
        ),
    )
    start: str | None = Field(
        default=None,
        description=(
            "Start date in YYYY-MM-DD format. "
            "Only set if explicitly stated in the query. Do NOT infer missing bounds."
        ),
    )
    end: str | None = Field(
        default=None,
        description=(
            "End date in YYYY-MM-DD format. "
            "Only set if explicitly stated in the query. Do NOT infer missing bounds."
        ),
    )
    current_period: bool = Field(
        default=False,
        description=(
            'Set to true when the query references the current period using phrases like '
            '"this year", "this month", "this quarter", "current year", etc. '
            'Do NOT set for explicit dates like "in 2025". '
            'When true, start and end should be the full period boundaries.'
        ),
    )


class DateTimeDimensionChain:
    _system_prompt: str

    def __init__(self, llm_model_config: LLMModelConfig, system_prompt: str):
        self._llm_model_config = llm_model_config
        self._system_prompt = system_prompt

    @staticmethod
    def _post_process(data: dict) -> DateTimeQueryResponse:
        response: DateTimeLLMResponse = data["llm_response"]
        current_date: str = data["current_date"]

        start = response.start
        end = response.end
        intent = response.intent

        # Step 1: Current-period clamping (only when current_period=True)
        if response.current_period:
            if intent == Intent.HISTORICAL and end is not None and end > current_date:
                end = current_date
            elif intent == Intent.FORECAST and start is not None and start < current_date:
                start = current_date

        # Step 2: Fill missing bounds
        if intent == Intent.HISTORICAL and end is None:
            end = current_date
        if intent == Intent.FORECAST and start is None:
            start = current_date

        # Step 3: Derive flags
        time_period_specified = (
            (intent != Intent.UNSPECIFIED) or (start is not None) or (end is not None)
        )

        return DateTimeQueryResponse(
            start=start,
            end=end,
            time_period_specified=time_period_specified,
        )

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
        ).with_structured_output(schema=DateTimeLLMResponse, method='json_schema')

        return (
            RunnablePassthrough.assign(
                current_date=lambda d: ChainParameters.get_configuration(d).get_current_date()
            )
            | RunnablePassthrough.assign(llm_response=prompt_template | llm)
            | self._post_process
        )
