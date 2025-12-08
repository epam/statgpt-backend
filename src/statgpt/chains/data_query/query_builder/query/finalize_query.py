from aidial_sdk.chat_completion import Stage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnablePassthrough

from common.config import multiline_logger as logger
from common.data.base import DataSetAvailabilityQuery, DataSetQuery, DimensionQuery, Query
from common.data.base.enums import InvalidDataSetQueryReasonType
from common.data.base.query import InvalidDataSetQueryReason
from common.data.sdmx.v21.dataset import Sdmx21DataSet
from common.schemas import DataQueryDetails
from common.schemas.data_query_tool import DataQueryMessages, DataQueryPrompts
from common.schemas.enums import TimePeriodStrategy
from common.utils import async_utils
from statgpt.chains.data_query.parameters import DataQueryParameters
from statgpt.chains.data_query.query_builder import utils as query_utils
from statgpt.chains.data_query.query_constructor import QueryConstructorFactory
from statgpt.chains.parameters import ChainParameters
from statgpt.config import ChainParametersConfig, StateVarsConfig
from statgpt.default_prompts import DataQueryDefaultPrompts
from statgpt.schemas.query_builder import (
    ChainState,
    DatasetAvailabilityQueriesType,
    DatasetDimensionTermNameType,
)
from statgpt.utils.callbacks import StageCallback
from statgpt.utils.datetime_adjuster import expand_time_range
from statgpt.utils.formatters import (
    DatasetAvailabilityQueryFormatter,
    DatasetAvailabilityQueryFormatterConfig,
    DatasetQueryFormatter,
    DatasetQueryFormatterConfig,
)

from .execute_query import ExecuteQueryChain
from .incomplete_queries import IncompleteQueriesChain
from .invalid_selected_time_period import InvalidSelectedTimePeriodChain
from .multiple_datasets import MultipleDatasetsChain
from .nodata import NoDataChain
from .summarize_query import SummarizeQueriesChain


class FinalizeQueryChainFactory:
    def __init__(self, config: DataQueryDetails):
        self._config = config

        prompts: DataQueryPrompts = self._config.prompts
        messages: DataQueryMessages = self._config.messages

        self._summarize_queries_chain = SummarizeQueriesChain(
            llm_model_config=self._config.llm_models.summarize_queries_model_config,
            system_prompt=prompts.summarize_queries_prompt
            or DataQueryDefaultPrompts.SUMMARIZE_QUERIES_PROMPT,
        )
        self._execute_query_chain = ExecuteQueryChain(
            stages_config=self._config.stages_config,
            executed_message_agent_only=messages.data_query_executed_agent_only,
            summarize_queries_chain=self._summarize_queries_chain,
        )
        self._no_data_chain = NoDataChain(
            message=messages.no_data,
        )
        self._multiple_datasets_chain = MultipleDatasetsChain(
            agent_only_message=messages.multiple_datasets_agent_only,
        )
        self._incomplete_queries_chain = IncompleteQueriesChain(
            llm_model_config=self._config.llm_models.incomplete_queries_model_config,
            system_prompt=prompts.incomplete_queries_prompt
            or DataQueryDefaultPrompts.INCOMPLETE_QUERIES_PROMPT,
        )
        self._invalid_time_period_chain = InvalidSelectedTimePeriodChain(
            message=messages.invalid_time_period
        )

    async def _format_dataset_queries(self, inputs: dict) -> str:
        chain_state = ChainState(**inputs)
        auth_context = chain_state.auth_context
        datasets_dict = chain_state.datasets_dict
        dataset_queries = chain_state.dataset_queries

        if dataset_queries:

            query_formatter = DatasetQueryFormatter(
                config=DatasetQueryFormatterConfig(
                    locale=chain_state.data_service.channel_config.locale,
                    include_missing_dimensions=False,
                    include_default_queries=True,
                    include_auto_selects=True,
                ),
                auth_context=auth_context,
            )
            return await query_formatter.format_queries(
                dataset_queries=dataset_queries,
                datasets_dict=datasets_dict,
            )
        else:
            # NOTE: we failed to build valid dataset queries.
            # we need to show user our best attempt and explain the reason why the query is invalid.
            logger.warning(
                'There are no dataset queries to format. '
                'Will show user the best attempt to build non-empty strong queries.'
            )
            msg = (
                '### No Data\n\n'
                'There is no data for the built query. '
                'The most likely reasons are:\n'
                '- incompatible combination of selected dimension values\n'
                '- absent indicator specifications in the query\n\n'
                'Please try to change the query.'
            )

            strong_queries_best_attempt = chain_state.strong_queries_best_nonempty_attempt
            if strong_queries_best_attempt:
                msg += '\n\n### Best Attempt to Build Query'
                msg += '\n\nHere is the best attempt to build query:'

                query_formatter = DatasetAvailabilityQueryFormatter(  # type: ignore[assignment]
                    config=DatasetAvailabilityQueryFormatterConfig(
                        locale=chain_state.data_service.channel_config.locale,
                        header_level=4,
                    ),
                    auth_context=auth_context,
                )

                formatted_queries = await query_formatter.format_queries(
                    dataset_queries=strong_queries_best_attempt,  # type: ignore[arg-type]
                    datasets_dict=datasets_dict,
                )
                msg += f"\n\n{formatted_queries}"
            return msg

    async def _populate_dataset_queries(self, stage: Stage, inputs: dict):
        chain_state = ChainState(**inputs)
        stage.append_content(chain_state.dataset_queries_formatted_str)

    async def _get_dataset_queries(self, inputs: dict) -> dict[str, DataSetQuery]:
        """Convert strong queries to dataset queries. Set missing dimensions if possible."""
        chain_state = ChainState(**inputs)
        datasets_dict = chain_state.datasets_dict
        strong_queries = chain_state.strong_queries
        strong_availability = chain_state.strong_availability

        if not strong_queries:
            return {}
        strong_queries_nonempty = query_utils.filter_empty_dataset_availability_queries(
            queries=strong_queries
        )
        if not strong_queries_nonempty:
            return {}

        dataset_queries: dict[str, DataSetQuery] = {}

        query_constructor = QueryConstructorFactory.create(self._config)
        for dataset_id, ds_strong_query in strong_queries_nonempty.items():
            dataset_queries[dataset_id] = await query_constructor.construct_query(
                ds_query=ds_strong_query,
                ds_availability_query=strong_availability.get(
                    dataset_id, DataSetAvailabilityQuery()
                ),
                dataset=datasets_dict[dataset_id].data,
                chain_state=chain_state,
            )

        return dataset_queries

    @staticmethod
    async def _get_availability_for_dataset_queries(inputs: dict) -> DatasetAvailabilityQueriesType:
        chain_state = ChainState.model_validate(inputs)
        queries = {k: v for k, v in chain_state.dataset_queries.items() if v.is_valid}
        if not queries:
            return {}

        tasks = []
        for dataset_id, query in queries.items():
            dataset = chain_state.datasets_dict[dataset_id].data
            tasks.append(
                dataset.availability_query(query.to_availability_query(), chain_state.auth_context)
            )

        logger.debug(f"Running {len(tasks)} availability queries concurrently")
        task_results = await async_utils.gather_with_concurrency(10, *tasks)
        result = {dataset_id: query for dataset_id, query in zip(queries.keys(), task_results)}
        logger.debug(f"Completed {len(result)} availability queries")
        return result

    @staticmethod
    def _time_query_to_values(query: Query) -> tuple[str | None, str | None]:
        if query.operator == query_utils.QueryOperator.BETWEEN and len(query.values) == 2:
            return query.values[0], query.values[1]
        if (
            query.operator == query_utils.QueryOperator.GREATER_THAN_OR_EQUALS
            and len(query.values) == 1
        ):
            return query.values[0], None
        if (
            query.operator == query_utils.QueryOperator.LESS_THAN_OR_EQUALS
            and len(query.values) == 1
        ):
            return None, query.values[0]
        raise ValueError(f"Unsupported time query format: {query!r}")

    def _apply_default_time_period_if_possible(
        self, availability_queries: dict[str, DataSetAvailabilityQuery], chain_state: ChainState
    ) -> dict[str, DataSetQuery]:

        result = {}

        for dataset_id, ava_query in availability_queries.items():
            dataset_query = chain_state.dataset_queries[dataset_id]
            dataset: Sdmx21DataSet = chain_state.datasets_dict[dataset_id].data  # type: ignore[assignment]

            time_dimension = dataset.get_time_dimension()
            if not time_dimension:  # dataset has no time dimension
                result[dataset_id] = dataset_query
                continue

            time_period_default = dataset.config.dimension_default_queries.get("TIME_PERIOD")
            if not time_period_default:
                result[dataset_id] = dataset_query
                continue

            default_start, default_end = self._time_query_to_values(time_period_default[0])
            if not default_start and not default_end:
                result[dataset_id] = dataset_query
                continue

            default_time_period_query = DimensionQuery.from_default_query(
                time_period_default[0], dimension_id="TIME_PERIOD"
            )

            available_start, available_end = ava_query.time_period_start, ava_query.time_period_end
            if available_start and default_end and default_end < available_start:
                # default end is before available start, do not apply default
                result[dataset_id] = dataset_query
            elif available_end and default_start and default_start > available_end:
                # default start is after available end, do not apply default
                result[dataset_id] = dataset_query
            else:
                # apply default time period
                dataset_query.dimensions_queries.append(default_time_period_query)
                result[dataset_id] = dataset_query

        return result

    def _apply_selected_time_period_to_query(
        self, availability_queries: dict[str, DataSetAvailabilityQuery], chain_state: ChainState
    ) -> dict[str, DataSetQuery]:
        date_time_query_response = chain_state.date_time_query_response
        date_time_query = date_time_query_response.to_query()
        if not date_time_query:  # data was requested for the entire available time period
            return chain_state.dataset_queries

        selected_start, selected_end = date_time_query_response.start, date_time_query_response.end
        valid_queries = {}
        invalid_queries = {}

        for dataset_id, ava_query in availability_queries.items():
            dataset_query = chain_state.dataset_queries[dataset_id]
            dataset: Sdmx21DataSet = chain_state.datasets_dict[dataset_id].data  # type: ignore[assignment]

            time_dimension = dataset.get_time_dimension()
            if not time_dimension:  # dataset has no time dimension
                valid_queries[dataset_id] = dataset_query
                continue

            available_start, available_end = ava_query.time_period_start, ava_query.time_period_end
            if available_start and selected_end and selected_end < available_start:
                # selected end is before available start, invalid query
                dataset_query.is_valid = False
                dataset_query.invalidity_reason = InvalidDataSetQueryReason(
                    type=InvalidDataSetQueryReasonType.INVALID_TIME_PERIOD,
                    details={
                        'field': 'selected_end',
                        'value': selected_end,
                        'available_start': available_start,
                        'available_end': available_end,
                    },
                )
                invalid_queries[dataset_id] = dataset_query
            elif available_end and selected_start and selected_start > available_end:
                # selected start is after available end, invalid query
                dataset_query.is_valid = False
                dataset_query.invalidity_reason = InvalidDataSetQueryReason(
                    type=InvalidDataSetQueryReasonType.INVALID_TIME_PERIOD,
                    details={
                        'field': 'selected_start',
                        'value': selected_start,
                        'available_start': available_start,
                        'available_end': available_end,
                    },
                )
                invalid_queries[dataset_id] = dataset_query
            else:
                # apply selected time period
                dataset_query.dimensions_queries.append(
                    DimensionQuery.from_query(date_time_query, dimension_id="TIME_PERIOD")
                )
                valid_queries[dataset_id] = dataset_query

        if valid_queries:
            return valid_queries  # prefer valid queries
        else:
            return invalid_queries  # all queries are invalid, let user know

    async def _post_time_period_filter(self, inputs: dict) -> dict[str, DataSetQuery]:
        """Apply selected time period if not done already."""

        if self._config.time_period_strategy is not TimePeriodStrategy.AFTER:
            # time period already applied
            return ChainParameters.get_dataset_queries(inputs)

        ava_queries = await self._get_availability_for_dataset_queries(inputs)

        if not ava_queries:
            # no valid dataset queries
            return ChainParameters.get_dataset_queries(inputs)

        chain_state = ChainState.model_validate(inputs)

        if chain_state.date_time_query_response.time_period_specified:
            return self._apply_selected_time_period_to_query(ava_queries, chain_state)
        else:
            return self._apply_default_time_period_if_possible(ava_queries, chain_state)

    def _map_dimension_ids_to_names(self, inputs: dict) -> DatasetDimensionTermNameType:
        dataset_to_dimension_id_to_name = {}
        chain_state = ChainState(**inputs)
        datasets = chain_state.datasets_dict
        dataset_queries = chain_state.dataset_queries
        for dataset_id, dataset_query in dataset_queries.items():
            dataset: Sdmx21DataSet = datasets[dataset_id].data  # type: ignore[assignment]
            dataset_dimension_id_to_name = {}
            for dimension, dimension_query in dataset_query.dimensions_queries_dict.items():
                id2name_mapping = dataset.map_component_values_id_2_name(
                    value_ids=dimension_query.values, component_id=dimension
                )
                # `None` is returned if the dimension has no corresponding code list, e.g.,
                # when it's time period dimension.
                if id2name_mapping is None:
                    continue
                dataset_dimension_id_to_name[dimension] = id2name_mapping
            dataset_to_dimension_id_to_name[dataset_id] = dataset_dimension_id_to_name
        return dataset_to_dimension_id_to_name

    async def _route_based_on_data_query_status(self, inputs: dict) -> Runnable:
        state = ChainParameters.get_state(inputs)
        skip = state.get(StateVarsConfig.CMD_SKIP_DATA_QUERY_SUMMARIZATION, False)
        if skip:
            query = ChainParameters.get_query(inputs)
            response = f"<call to Query_Data was skipped for debug purposes>\n\n {query!r}"
            return RunnablePassthrough.assign(
                **{DataQueryParameters.RESPONSE_FIELD: lambda _: response}
            )

        chain_state = ChainState(**inputs)
        dataset_queries = chain_state.dataset_queries

        auth_context = ChainParameters.get_auth_context(inputs)

        if not dataset_queries:  # todo: use missing dimensions to ask question to user
            # TODO: there are at least 3 possibile cases:
            #
            # 1. all queries were filtered by avaialability,
            # due to invalid dimension values combination.
            # i.e. there is no data for the query
            #
            # 2. user did not specify at least 1 required indicator dimension.
            #
            # 3. search failed to build query for at least 1 required indicator dimension.
            #
            # Currently we don't differentiate between these cases,
            # and the message shown to user is misleading.
            return await self._no_data_chain.create_chain(inputs)

        valid_queries = {ds_id: dq for ds_id, dq in dataset_queries.items() if dq.is_valid}

        if self._config.clarify_if_multiple_datasets and len(valid_queries) > 1:
            return (
                self._summarize_queries_chain.create_chain
                | await self._multiple_datasets_chain.create_chain()
            )

        if len(valid_queries) >= 1:
            return (
                RunnablePassthrough.assign(
                    **{ChainParametersConfig.DATASET_QUERIES: lambda _: valid_queries}
                )
                | self._execute_query_chain.create_chain()
            )

        if any(q.invalidity_reason is not None for q in dataset_queries.values()):
            return (
                self._summarize_queries_chain.create_chain
                | self._invalid_time_period_chain.create_chain()
            )

        # all queries are invalid: have some dimensions are missing
        # ToDo: adjust to process multiple datasets
        dataset_id = next(iter(dataset_queries))

        query_formatter = DatasetQueryFormatter(
            config=DatasetQueryFormatterConfig(
                locale=chain_state.data_service.channel_config.locale,
                include_missing_dimensions=True,
                include_default_queries=True,
                include_auto_selects=True,
            ),
            auth_context=auth_context,
        )

        incomplete_queries_chain_inputs = dict(
            **{k: v for k, v in inputs.items() if k != 'dataset_queries'},
            formatted_query_with_missing_dimensions=await query_formatter.format_queries(
                dataset_queries=dataset_queries,
                datasets_dict=chain_state.datasets_dict,
                availability_queries=chain_state.strong_availability,  # type: ignore[arg-type]
            ),
            dataset_queries={dataset_id: dataset_queries[dataset_id]},
        )
        return (
            self._summarize_queries_chain.create_chain
            | await self._incomplete_queries_chain.create_chain(
                incomplete_queries_chain_inputs, auth_context.api_key
            )
        )

    def _create_finalization_chain(self) -> Runnable:
        """Create the actual finalization chain."""
        construct_data_query_stage_name = "Constructing Data Query"

        return (
            (
                RunnablePassthrough.assign(dataset_queries=self._get_dataset_queries)
                | RunnablePassthrough.assign(dataset_queries=self._post_time_period_filter)
                | RunnablePassthrough.assign(dataset_queries=expand_time_range)
                | RunnablePassthrough.assign(
                    dimension_id_to_name=self._map_dimension_ids_to_names,
                    dataset_queries_formatted_str=self._format_dataset_queries,
                )
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            stage_name=construct_data_query_stage_name,
                            content_appender=self._populate_dataset_queries,
                            debug_only=self._config.stages_config.is_stage_debug(
                                construct_data_query_stage_name
                            ),
                        )
                    ]
                )
            )
            | query_utils.set_tool_state
            | self._route_based_on_data_query_status
        )

    async def _route_finalization(self, inputs: dict) -> Runnable:
        """Route to either skip or execute finalization based on skip_finalization flag."""
        if inputs.get('skip_finalization', False):
            return RunnablePassthrough()
        else:
            return self._create_finalization_chain()

    def create(self) -> Runnable:
        return RunnableLambda(self._route_finalization)
