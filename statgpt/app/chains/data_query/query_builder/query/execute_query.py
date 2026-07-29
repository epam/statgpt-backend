from aidial_sdk.chat_completion import Stage
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from statgpt.app.chains.data_query.parameters import DataQueryParameters
from statgpt.app.chains.data_query.query_builder import utils as query_utils
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.config import ChainParametersConfig
from statgpt.app.schemas.data_query_outcome import DataQueryStatus
from statgpt.app.schemas.query_builder import ChainState
from statgpt.app.services.chat_facade import VersionedDataSet
from statgpt.app.utils.callbacks import StageCallback
from statgpt.app.utils.formatters import DatasetQueryFormatter, DatasetQueryFormatterConfig
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import DataResponse, DataSetQuery
from statgpt.common.schemas import StagesConfig
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus
from statgpt.common.schemas.tool_details import StageDescriptor

from .summarize_query import SummarizeQueriesChain


class ExecuteQueryChain:
    def __init__(
        self,
        stages_config: StagesConfig,
        stage: StageDescriptor,
        executed_message_agent_only: str | None,
        summarize_queries_chain: SummarizeQueriesChain,
    ):
        self._stages_config = stages_config
        self._stage = stage
        self._executed_message_agent_only = executed_message_agent_only
        self._summarize_queries_chain = summarize_queries_chain

    async def summarize_dataset_queries(self, inputs: dict) -> dict:
        chain_state = ChainState.model_validate(inputs)
        auth_context = chain_state.auth_context
        datasets_dict = chain_state.datasets_dict
        dataset_queries = chain_state.dataset_queries

        data_responses = ChainParameters.get_data_responses(inputs)
        configuration = ChainParameters.get_configuration(inputs)

        target = ChainParameters.get_target(inputs)

        query_formatter = DatasetQueryFormatter(
            config=DatasetQueryFormatterConfig(
                locale=chain_state.data_service.channel_config.locale,
                include_missing_dimensions=False,
                include_default_queries=True,
                include_auto_selects=True,
            ),
            auth_context=auth_context,
        )

        formatted_queries = await query_formatter.format_queries(
            dataset_queries=dataset_queries,
            datasets_dict=datasets_dict,
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

        query_utils.set_data_query_status(inputs, self._execution_status(data_responses))
        return inputs

    @staticmethod
    def _execution_status(data_responses: dict[str, DataResponse | None] | None) -> DataQueryStatus:
        """Classify the outcome of executed queries.

        Rows win: a partially failed fan-out that still returned data is reported as
        ``data_available``. Otherwise a fetch or parse failure is reported as ``failed`` rather
        than as an empty result — ``Sdmx21DataSet.query`` swallows those errors and returns an
        empty response, so row count alone cannot tell them apart.
        """
        responses = [r for r in (data_responses or {}).values() if r is not None]
        if any(not response.is_empty for response in responses):
            return DataQueryStatus.DATA_AVAILABLE
        if any(
            response.status.request_status == DataRequestStatus.FAILED
            or response.status.parsing_status == DataParsingStatus.FAILED
            for response in responses
        ):
            return DataQueryStatus.FAILED
        return DataQueryStatus.EXECUTED_NO_DATA

    @classmethod
    def _get_data_query_chain(
        cls, dataset: VersionedDataSet, query: DataSetQuery, auth_context: AuthContext
    ) -> Runnable:
        async def _query(d: dict):
            return await d["dataset"].data.query(d["query"], auth_context)

        return RunnablePassthrough.assign(
            dataset=lambda _: dataset, query=lambda _: query
        ) | RunnableLambda(_query)

    @classmethod
    def _get_data_query_chains(cls, inputs: dict) -> dict[str, Runnable]:
        chain_state = ChainState.model_validate(inputs)
        auth_context = chain_state.auth_context
        datasets_dict = chain_state.datasets_dict
        dataset_queries = chain_state.dataset_queries
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
            | self._summarize_queries_chain.create_chain
            | self.summarize_dataset_queries
        )

        callback = StageCallback(
            stage_name=self._stage.name,
            content_appender=self.append_data_responses_stage,
            debug_only=self._stage.is_debug(self._stages_config),
        )

        chain = chain.with_config(RunnableConfig(callbacks=[callback]))  # type: ignore[assignment]
        return chain
