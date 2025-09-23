import typing as t

from aidial_sdk.chat_completion import Stage
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from common.auth.auth_context import AuthContext
from common.data.base import DataSet, DataSetQuery
from common.schemas import StagesConfig
from statgpt.chains.data_query.parameters import DataQueryParameters
from statgpt.chains.parameters import ChainParameters
from statgpt.config import ChainParametersConfig
from statgpt.utils.callbacks import StageCallback

from .utils import format_dataset_queries


class DataQueryChain:
    def __init__(self, stages_config: StagesConfig, executed_message_agent_only: str | None):
        self._stages_config = stages_config
        self._executed_message_agent_only = executed_message_agent_only

    async def summarize_dataset_queries(self, inputs: dict) -> dict:
        auth_context = ChainParameters.get_auth_context(inputs)
        datasets_dict = ChainParameters.get_datasets_dict(inputs)
        data_responses = ChainParameters.get_data_responses(inputs)
        dataset_queries = ChainParameters.get_dataset_queries(inputs)
        configuration = ChainParameters.get_configuration(inputs)

        target = ChainParameters.get_target(inputs)

        formatted_queries = await format_dataset_queries(
            auth_context,
            dataset_queries,
            datasets_dict,
            include_missing_dimensions=False,
            include_default_queries=True,
            include_auto_selects=True,
            data_responses=data_responses,
        )

        response_content = "The following queries were executed:\n\n" + formatted_queries
        target.append_content(response_content)
        # append message to be shown to agent only (not to user) if it's configured
        if self._executed_message_agent_only:
            response_content += f"\n\n{self._executed_message_agent_only}"

        timestamp = configuration.get_current_timestamp()
        response_content += f"\n[Data Query executed at {timestamp}]"

        inputs[DataQueryParameters.RESPONSE_FIELD] = response_content
        return inputs

    @classmethod
    def _get_data_query_chain(
        cls, dataset: DataSet, query: DataSetQuery, auth_context: AuthContext
    ) -> Runnable:
        async def _query(d: dict):
            return await d["dataset"].query(d["query"], auth_context)

        return RunnablePassthrough.assign(
            dataset=lambda _: dataset, query=lambda _: query
        ) | RunnableLambda(_query)

    @classmethod
    def _get_data_query_chains(cls, inputs: dict) -> t.Dict[str, Runnable]:
        datasets_dict = inputs["datasets_dict"]
        dataset_queries = inputs["dataset_queries"]
        auth_context = ChainParameters.get_auth_context(inputs)
        return {
            dataset_id: cls._get_data_query_chain(datasets_dict[dataset_id], query, auth_context)
            for dataset_id, query in dataset_queries.items()
        }

    @classmethod
    def _get_data_queries_chain(cls, inputs: dict) -> RunnableParallel:
        return RunnableParallel(cls._get_data_query_chains(inputs))

    @staticmethod
    async def append_data_responses_stage(stage: Stage, d: dict):
        pass

    def create_chain(self) -> Runnable:
        chain = (
            RunnablePassthrough.assign(
                **{ChainParametersConfig.DATA_RESPONSES: self._get_data_queries_chain},
            )
            | self.summarize_dataset_queries
        )

        stage_name = "Executing Data Query"
        callback = StageCallback(
            stage_name=stage_name,
            content_appender=self.append_data_responses_stage,
            debug_only=self._stages_config.is_stage_debug(stage_name),
        )

        chain = chain.with_config(RunnableConfig(callbacks=[callback]))
        return chain
