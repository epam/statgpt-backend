import logging

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import BaseModel, Field

from common.schemas import LLMModelConfig
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters
from statgpt.utils.formatters import DatasetQueryFormatter, DatasetQueryFormatterConfig

_log = logging.getLogger(__name__)


class QuerySummary(BaseModel):
    uuid: str = Field(description="The UUID of the query.")
    short_summary: str = Field(description="A brief summary of the query.")


class QuerySummaries(BaseModel):
    summaries: list[QuerySummary] = Field(description="A list of query summaries.")


class SummarizeQueriesChain:

    _system_prompt: str
    _llm_model_config: LLMModelConfig

    def __init__(self, llm_model_config: LLMModelConfig, system_prompt: str):
        self._system_prompt = system_prompt
        self._llm_model_config = llm_model_config

    @staticmethod
    async def _format_queries(inputs: dict) -> str:
        auth_context = ChainParameters.get_auth_context(inputs)

        data_service = ChainParameters.get_data_service(inputs)
        queries = ChainParameters.get_dataset_queries(inputs)
        datasets = ChainParameters.get_datasets_dict(inputs)
        formatter = DatasetQueryFormatter(
            config=DatasetQueryFormatterConfig(
                locale=data_service.channel_config.locale,
                include_missing_dimensions=False,
                include_default_queries=True,
                include_auto_selects=True,
                include_query_uuid=True,
            ),
            auth_context=auth_context,
        )
        data_responses = ChainParameters.get_data_responses(inputs)

        formatted_queries = await formatter.format_queries(
            dataset_queries=queries,
            datasets_dict=datasets,
            data_responses=data_responses,
        )
        return formatted_queries

    @staticmethod
    async def _enrich_queries(inputs: dict) -> dict:
        queries = ChainParameters.get_dataset_queries(inputs)
        query_summaries = inputs["query_summaries"]
        summary_dict = {summary.uuid: summary for summary in query_summaries.summaries}

        for query in queries.values():
            if query.uuid in summary_dict:
                summary = summary_dict.get(query.uuid)
                if not summary:
                    _log.warning(f"Missing summary for query in SummarizeQueryChain: {query}")
                    continue
                query.short_summary = summary.short_summary
        return inputs

    def create_chain(self, inputs: dict) -> Runnable:
        auth_context = ChainParameters.get_auth_context(inputs)
        locale = ChainParameters.get_data_service(inputs).channel_config.locale

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self._system_prompt),
                HumanMessagePromptTemplate.from_template("{formatted_queries}"),
            ],
        ).partial(language=locale.get_language_name())
        chat_model = get_chat_model(
            api_key=auth_context.api_key, model_config=self._llm_model_config
        ).with_structured_output(schema=QuerySummaries, method="json_schema")

        return (
            RunnablePassthrough.assign(
                query_summaries=(
                    RunnablePassthrough.assign(formatted_queries=self._format_queries)
                    | prompt_template
                    | chat_model
                ),
            )
            | self._enrich_queries
        )
