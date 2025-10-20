import asyncio
import logging
from copy import deepcopy
from itertools import groupby
from operator import attrgetter

from aidial_sdk.chat_completion import Stage
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.data.base import DataSetAvailabilityQuery, DataSetQuery, QueryOperator
from common.data.sdmx.common import DimensionVirtualCodeCategory
from common.data.sdmx.v21.dataset import Sdmx21DataSet
from common.schemas import ChannelConfig, DataQueryDetails
from common.schemas.data_query_tool import (
    DataQueryMessages,
    DataQueryPrompts,
    SpecialDimensionsProcessor,
)
from common.schemas.enums import SpecialDimensionsProcessorType
from common.utils import async_utils
from common.utils.timer import debug_timer
from statgpt.chains import CandidatesSelectionSimpleChainFactory
from statgpt.chains.data_query.base import BaseDataQueryFactory
from statgpt.chains.data_query.parameters import DataQueryParameters
from statgpt.chains.data_query.query_constructor import QueryConstructorFactory
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.utils import dataset_utils
from statgpt.config import ChainParametersConfig, StateVarsConfig
from statgpt.default_prompts.v2 import DefaultPrompts
from statgpt.schemas.query_builder import (
    ChainState,
    DatasetAvailabilityQueriesType,
    DatasetDimensionTermNameType,
    LLMSelectionDimensionCandidate,
    MetaStateKeys,
    NamedEntitiesResponse,
    NamedEntity,
    QueryBuilderAgentState,
)
from statgpt.services.chat_facade import ScoredDimensionCandidate
from statgpt.utils.callbacks import StageCallback
from statgpt.utils.formatters import DatasetFormatterConfig, DatasetsListFormatter

from . import utils as query_utils
from .data_query import DataQueryChain
from .datasets_selection import DataSetsSelectionChain
from .datetime_adjuster import expand_time_range
from .datetime_chain import DateTimeDimensionChain
from .incomplete_queries import IncompleteQueriesChain
from .indicator_selection.factory import IndicatorSelectionFactory
from .multiple_datasets import MultipleDatasetsChain
from .named_entities import NamedEntitiesChain
from .nodata import NoDataChain
from .normalization import NormalizationChain
from .special_dimensions import LHCLChainFactory, SpecialDimensionChainFactoryBase


class QueryBuilderFactoryV2(BaseDataQueryFactory):
    def __init__(self, config: DataQueryDetails, channel_config: ChannelConfig):
        super().__init__(config, channel_config)

        prompts: DataQueryPrompts = self._config.prompts
        messages: DataQueryMessages = self._config.messages

        self._datetime_chain = DateTimeDimensionChain(
            llm_model_config=self._config.llm_models.time_period_model_config,
            system_prompt=prompts.datetime_prompt or DefaultPrompts.DATETIME_PROMPT,
        )
        # self._group_expander_chain = GroupExpanderChain(
        #     llm_model_config=self._config.llm_models.group_expander_model_config,
        #     system_prompt=prompts.group_expander_prompt or DefaultPrompts.GROUP_EXPANDER_PROMPT,
        #     fallback_prompt=prompts.group_expander_fallback_prompt
        #     or DefaultPrompts.GROUP_EXPANDER_FALLBACK_PROMPT,
        # )
        self._normalization_chain = NormalizationChain(
            llm_model_config=self._config.llm_models.query_normalization_model_config,
            system_prompt=prompts.normalization_prompt or DefaultPrompts.NORMALIZATION_PROMPT,
        )
        self._named_entities_chain = NamedEntitiesChain(
            llm_model_config=self._config.llm_models.named_entities_model_config,
            system_prompt=prompts.named_entities_prompt or DefaultPrompts.NAMED_ENTITIES_PROMPT,
        )
        self._datasets_selection_chain = DataSetsSelectionChain(
            llm_model_config=self._config.llm_models.datasets_selection_model_config,
            system_user_prompt=prompts.dataset_selection_prompts
            or DefaultPrompts.DATASET_SELECTION_PROMPTS,
        )
        self._dimensions_selection_chain_factory = CandidatesSelectionSimpleChainFactory(
            llm_model_config=self._config.llm_models.dimensions_selection_model_config,
            system_prompt=prompts.validation_system_prompt
            or DefaultPrompts.VALIDATION_SYSTEM_PROMPT,
            user_prompt=prompts.validation_user_prompt or DefaultPrompts.VALIDATION_USER_PROMPT,
            candidates_key="dimension_candidates_for_llm_selection",
        )

        self._data_query_chain = DataQueryChain(
            stages_config=self._config.stages_config,
            executed_message_agent_only=messages.data_query_executed_agent_only,
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
            or DefaultPrompts.INCOMPLETE_QUERIES_PROMPT,
        )

    @staticmethod
    def _apply_dataset_selection_response(inputs: dict):
        """
        1. Filter datasets by selected IDs
        2. Update normalized query
        """

        chain_state = ChainState(**inputs)
        datasets_selection_response = chain_state.datasets_selection_response
        datasets_dict_indexed = chain_state.datasets_dict_indexed

        # set 'datasets_dict'
        if not (selected_dataset_ids := datasets_selection_response.dataset_ids):
            logger.info('LLM selected no datasets. Using all available datasets.')
            inputs['datasets_dict'] = datasets_dict_indexed
        else:
            selected_dataset_ids = set(selected_dataset_ids)
            datasets_dict = {
                ds_id: ds
                for ds_id, ds in datasets_dict_indexed.items()
                if ds_id in selected_dataset_ids
            }
            inputs['datasets_dict'] = datasets_dict
        # update 'normalized_query'
        inputs['normalized_query'] = datasets_selection_response.rewritten_query

        return inputs

    @classmethod
    def _set_tool_state(cls, inputs: dict) -> dict:
        chain_state = ChainState(**inputs)

        indexed_datasets_id_map = {
            entity_id: ds.source_id for entity_id, ds in chain_state.datasets_dict_indexed.items()
        }

        query = ChainParameters.get_query(inputs)

        agent_state = QueryBuilderAgentState(
            query=query,
            query_with_expanded_groups=chain_state.query_with_expanded_groups,
            normalized_query_raw=chain_state.normalized_query_raw,
            datasets_selection_response=chain_state.datasets_selection_response,
            normalized_query=chain_state.normalized_query,
            date_time_query_response=chain_state.date_time_query_response,
            named_entities_response=chain_state.named_entities_response,
            indexed_datasets_id_map=indexed_datasets_id_map,
            weak_queries=chain_state.weak_queries,
            strong_queries=chain_state.strong_queries,
            dataset_queries=chain_state.dataset_queries,
            retrieval_results=chain_state.retrieval_results,
            dimension_id_to_name=chain_state.dimension_id_to_name,
            special_dims_outputs=chain_state.special_dims_outputs,
        )

        # cast to dict, since it will be serialized
        agent_state_dict = agent_state.model_dump(mode='json')
        # update state inplace
        inputs[DataQueryParameters.STATE] = agent_state_dict

        return inputs

    @classmethod
    async def _get_dimension_candidates_from_named_entities(
        cls, inputs: dict
    ) -> list[ScoredDimensionCandidate]:
        with debug_timer("_get_dimension_candidates_from_named_entities"):
            chain_state = ChainState(**inputs)
            datasets_dict = chain_state.datasets_dict
            named_entities_response = chain_state.named_entities_response
            data_service = chain_state.data_service

            filtered_named_entities = [
                ne for ne in named_entities_response.entities if ne.entity_type.lower() != "dataset"
            ]
            tasks = []
            for entity in filtered_named_entities:
                tasks.append(
                    data_service.search_dimensions_scored(
                        entity.to_query(),
                        auth_context=chain_state.auth_context,
                        k=30,  # TODO: make configurable
                        datasets=set(datasets_dict.keys()),
                    )
                )
            with debug_timer(f"non_indicator_dimension.candidates_search[{len(tasks)}]"):
                results = await asyncio.gather(*tasks)

            candidates_all: list[ScoredDimensionCandidate] = []
            for result in results:
                candidates_all.extend(result)

            candidates_dedup = list(set(candidates_all))
            candidates_dedup = sorted(candidates_dedup, key=lambda x: x.score, reverse=True)

        return candidates_dedup

    @staticmethod
    async def _get_availability(inputs: dict, queries_key: str) -> DatasetAvailabilityQueriesType:
        chain_state = ChainState(**inputs)
        auth_context = ChainParameters.get_auth_context(inputs)
        queries: DatasetAvailabilityQueriesType = inputs[queries_key]
        if len(queries) == 0:
            return {}

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Starting availability queries for {len(queries)} datasets from {queries_key}"
            )
            # Format queries for detailed logging
            formatted_queries = await query_utils.format_availability_queries(
                auth_context,
                queries,
                chain_state.datasets_dict,
                add_value_ids=True,
                header_level=4,
                add_citation=False,
            )
            logger.debug(f"Availability queries to run:\n{formatted_queries}")

        tasks = []
        for dataset_id, query in queries.items():
            dataset = chain_state.datasets_dict[dataset_id]
            tasks.append(dataset.availability_query(query, auth_context))

        logger.debug(f"Running {len(tasks)} availability queries concurrently")
        task_results = await async_utils.gather_with_concurrency(10, *tasks)
        result = {dataset_id: query for dataset_id, query in zip(queries.keys(), task_results)}
        logger.debug(f"Completed {len(result)} availability queries")
        return result

    async def _get_availability_from_strong_queries(
        self, inputs: dict
    ) -> DatasetAvailabilityQueriesType:
        return await self._get_availability(inputs, "strong_queries")

    @staticmethod
    def _filter_queries_by_availability(
        inputs: dict, queries_key: str, availability_key: str
    ) -> dict:
        queries: DatasetAvailabilityQueriesType = inputs[queries_key]
        availability: DatasetAvailabilityQueriesType = inputs[availability_key]
        result = {}
        for dataset_id, query in queries.items():
            availability_query = availability.get(dataset_id)
            if not availability_query:
                continue
            result[dataset_id] = query.filter(availability_query)
        return result

    @classmethod
    def _filter_strong_queries_by_strong_availability(
        cls, inputs: dict
    ) -> DatasetAvailabilityQueriesType:
        return cls._filter_queries_by_availability(
            inputs, queries_key="strong_queries", availability_key="strong_availability"
        )

    @staticmethod
    def _get_country_named_entities(inputs: dict) -> list[NamedEntity]:
        chain_state = ChainState(**inputs)
        country_named_entity_type = chain_state.data_service.get_country_named_entity_type()
        named_entities_response = chain_state.named_entities_response.entities
        country_entities = [
            ne
            for ne in named_entities_response
            if country_named_entity_type.lower().startswith(ne.entity_type.lower())
            # ToDo: remove this temporary workaround by allowing to define hints or descriptions for named entity types
        ]
        logger.info(
            f'Found {len(country_entities)} {country_named_entity_type} named entities: {country_entities}'
        )
        return country_entities

    def _filter_strong_queries_by_countries(self, inputs: dict) -> DatasetAvailabilityQueriesType:
        """
        1. check if country dimension is filled in at least one dataset query.
        if so, remove all dataset queries without country query and exit.

        2. else, check if there are any country named entities detected.
        if so, remove all dataset queries and exit.

        Original example:
        STATCAN (Statistics of Canada) does not have data for France.
        query "What is the GDP of France?" will not fill country dim for STATCAN,
        but it will for datasets containing France.
        we filter out STATCAN dataset query in this method, as it's irrelevant.

        See issue 75 for reference.
        """
        logger.info('_filter_strong_queries_by_countries()')
        chain_state = ChainState(**inputs)
        queries = chain_state.strong_queries

        at_least_one_country_selected: bool = False
        datasets_entity_ids_without_country_query = set()
        datasets_source_ids_without_country_query = set()

        for dataset_id, query in queries.items():
            dataset = chain_state.datasets_dict[dataset_id]
            if not isinstance(dataset, Sdmx21DataSet):
                raise ValueError(
                    f'Dataset "{dataset.source_id}" is not an instance of InMemorySdmx21DataSet'
                )
            country_dim = dataset.country_dimension()
            if not country_dim:
                # no country dimension - don't filter this dataset out
                continue

            country_query = query.dimensions_queries_dict.get(country_dim.entity_id)

            if country_query and not country_query.is_empty():
                at_least_one_country_selected = True
            else:
                datasets_entity_ids_without_country_query.add(dataset_id)
                datasets_source_ids_without_country_query.add(dataset.source_id)

        if at_least_one_country_selected:
            logger.info('filter by selected country terms')

            if datasets_entity_ids_without_country_query:
                logger.info(
                    'at least one dataset has country query. '
                    f'removing {len(datasets_source_ids_without_country_query)} '
                    'dataset queries without country query: '
                    f'{datasets_source_ids_without_country_query}'
                )
                queries = {
                    dataset_id: query
                    for dataset_id, query in queries.items()
                    if dataset_id not in datasets_entity_ids_without_country_query
                }
            else:
                logger.info(
                    'keeping all dataset queries: '
                    'either all of them have country query '
                    'or there is no country dim for them'
                )
        else:
            if self._config.filter_by_country_entities is False:
                logger.info(
                    'no selected country terms. '
                    'filter by country named entities is disabled in config. '
                    'return queries as they are.'
                )
                return queries

            logger.info('no selected country terms. fallback to filter by country named entities')

            country_entities = chain_state.country_named_entities
            if country_entities:
                logger.info(
                    'no country queries found, but country named entity was found. '
                    'clearing all dataset queries'
                )
                queries = {}
            else:
                logger.info(
                    'no country queries found, and no country named entity found. '
                    'keeping all dataset queries'
                )
        return queries

    @staticmethod
    def _add_all_values_to_nonindicator_candidates(
        inputs: dict,
    ) -> list[LLMSelectionDimensionCandidate]:
        """
        Append 'All values' candidates for non-indicator dimensions.
        This is used to allow LLM to select all values for non-indicator dimensions.
        """
        chain_state = ChainState(**inputs)
        dimension_candidates = chain_state.dimension_candidates_for_llm_selection
        datasets_dict = chain_state.datasets_dict
        index = len(dimension_candidates)
        for ds in datasets_dict.values():
            if not isinstance(ds, Sdmx21DataSet):
                raise ValueError(
                    f'Dataset "{ds.source_id}" is not an instance of InMemorySdmx21DataSet'
                )
            dimensions = {dim.entity_id: dim for dim in ds.non_indicator_dimensions()}
            for dim_id, fixed_item in ds.config.dimension_all_values.items():
                if dim_id not in dimensions:
                    # skip indicator dimensions
                    continue
                dimension = dimensions[dim_id]
                # NOTE: we assume there are no such terms already present in dimension_candidates
                dimension_candidates.append(
                    LLMSelectionDimensionCandidate(
                        score=1.0,
                        dataset_id=ds.entity_id,
                        dimension_category=DimensionVirtualCodeCategory(
                            fixed_item=fixed_item,
                            dimension_id=dimension.entity_id,
                            dimension_name=dimension.name,
                            dimension_alias=dimension.alias,
                        ),
                        index=index,
                    )
                )
                index += 1
        return dimension_candidates

    async def _populate_normalization(self, stage: Stage, inputs: dict):
        normalized_query = inputs.get("normalized_query", "")
        if normalized_query:
            stage.append_content(f"Normalized Query: `{normalized_query}`\n")

    async def _populate_datetime(self, stage: Stage, inputs: dict):
        chain_state = ChainState(**inputs)
        datetime_json = chain_state.date_time_query_response.model_dump_json(indent=2)
        stage.append_content(f"Date Time Query:\n```json\n{datetime_json}\n```\n")

    async def _populate_named_entities(self, stage: Stage, inputs: dict):
        named_entities_response = inputs.get("named_entities_response", NamedEntitiesResponse())
        if not named_entities_response:
            return

        entities = sorted(named_entities_response.entities, key=attrgetter("entity_type", "entity"))
        for k, g in groupby(entities, key=attrgetter("entity_type")):
            entities_str = ", ".join(f"**{entity.entity}**" for entity in g)
            stage.append_content(f"* _{k}_: " + entities_str + '\n')

    async def _populate_datasets_dict(self, stage: Stage, inputs: dict):
        chain_state = ChainState(**inputs)
        channel_config = chain_state.data_service.channel_config

        content = await DatasetsListFormatter(
            DatasetFormatterConfig(
                locale=channel_config.locale,
                citation=None,
                use_description=False,
            ),
            chain_state.auth_context,
        ).format(chain_state.datasets_dict.values())

        stage.append_content(content)

    @classmethod
    async def _populate_strong_queries_stage(cls, stage: Stage, inputs: dict):
        with debug_timer("_populate_strong_queries_stage"):
            chain_state = ChainState(**inputs)
            await query_utils.populate_queries_stage(
                stage=stage,
                queries=chain_state.strong_queries,
                auth_context=chain_state.auth_context,
                datasets_dict=chain_state.datasets_dict,
            )

    def _update_strong_queries_best_attempt_if_possible(self, inputs: dict):
        """
        Here we save a copy of current version of non-empty strong queries to a separate field.
        This field is used as our best attempt to build non-empty strong queries.
        This function should be called before every update to strong queries.
        """
        chain_state = ChainState(**inputs)
        strong_queries = chain_state.strong_queries
        if not strong_queries:
            return inputs
        strong_filtered = query_utils.filter_empty_dataset_availability_queries(
            queries=strong_queries
        )
        if strong_filtered:
            inputs['strong_queries_best_nonempty_attempt'] = deepcopy(strong_filtered)
        return inputs

    async def _format_dataset_queries(self, inputs: dict) -> str:
        chain_state = ChainState(**inputs)
        auth_context = chain_state.auth_context
        datasets_dict = chain_state.datasets_dict
        dataset_queries = chain_state.dataset_queries

        if dataset_queries:
            return await query_utils.format_dataset_queries(
                auth_context,
                dataset_queries,
                datasets_dict,
                include_missing_dimensions=False,
                include_default_queries=True,
                include_auto_selects=True,
            )
        else:
            # NOTE: we failed to build valid dataset queries.
            # we need to show user our best attempt and explain the reason why the query is invalid.
            logger.warning(
                'There are no dataset queries to format. '
                'Will show user the best attemp to build non-empty strong queries.'
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
                formatted_queries = await query_utils.format_availability_queries(
                    auth_context, strong_queries_best_attempt, datasets_dict, header_level=4
                )
                msg += f"\n\n{formatted_queries}"
            return msg

    async def _populate_dataset_queries(self, stage: Stage, inputs: dict):
        chain_state = ChainState(**inputs)
        stage.append_content(chain_state.dataset_queries_formatted_str)

    def _filter_queries_by_required_indicator_dims(self, inputs: dict) -> dict[str, DataSetQuery]:
        chain_state = ChainState(**inputs)
        datasets_dict = chain_state.datasets_dict
        strong_queries = chain_state.strong_queries

        if not strong_queries:
            return {}

        filtered_queries = {}
        for dataset_id, dataset_query in strong_queries.items():
            # check if query contains at least 1 required indicator dimension.
            # if it does not, we remove this query,
            # without asking user to fill in missing dimensions (by marking query as invalid).
            # reason is we want to filter non-informative False Positive queries,
            # like selecting unit of measure without the actual indicator (measure).
            required_ind_dims = datasets_dict[dataset_id].indicator_dimensions_required_for_query()
            dim_queries = dataset_query.dimensions_queries_dict
            if required_ind_dims and all(
                indicator_id not in dim_queries or not dim_queries[indicator_id].values
                for indicator_id in required_ind_dims
            ):
                logger.info(
                    f'will remove "{dataset_id}" dataset query, since it does not contain '
                    'at least 1 required indicator dim: '
                    f'{required_ind_dims}. query: {dataset_query}'
                )
                continue

            filtered_queries[dataset_id] = dataset_query

        return filtered_queries

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

        for dataset_id, ds_strong_query in strong_queries_nonempty.items():
            dataset = datasets_dict[dataset_id]
            query_constructor = QueryConstructorFactory.create(dataset)
            dataset_queries[dataset_id] = await query_constructor.construct_query(
                ds_query=ds_strong_query,
                ds_availability_query=strong_availability.get(
                    dataset_id, DataSetAvailabilityQuery()
                ),
                dataset=datasets_dict[dataset_id],
                chain_state=chain_state,
            )

        return dataset_queries

    def _map_dimension_ids_to_names(self, inputs: dict) -> DatasetDimensionTermNameType:
        dataset_to_dimension_id_to_name = {}
        chain_state = ChainState(**inputs)
        datasets = chain_state.datasets_dict
        dataset_queries = chain_state.dataset_queries
        for dataset_no, dataset_query in dataset_queries.items():
            dataset: Sdmx21DataSet = datasets[dataset_no]
            dataset_dimension_id_to_name = {}
            for dimension, dimension_query in dataset_query.dimensions_queries_dict.items():
                id2name_mapping = dataset.map_dim_values_id_2_name(
                    value_ids=dimension_query.values, dimension_name=dimension
                )
                # `None` is returned if the dimension has no corresponding code list, e.g.,
                # when it's time period dimension.
                if id2name_mapping is None:
                    continue
                dataset_dimension_id_to_name[dimension] = id2name_mapping
            dataset_to_dimension_id_to_name[dataset_no] = dataset_dimension_id_to_name
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

        if self._config.clarify_if_multiple_datasets and len(dataset_queries) > 1:
            return await self._multiple_datasets_chain.create_chain()

        valid_queries = {ds_id: dq for ds_id, dq in dataset_queries.items() if dq.is_valid}
        if len(valid_queries) >= 1:
            return (
                RunnablePassthrough.assign(
                    **{ChainParametersConfig.DATASET_QUERIES: lambda _: valid_queries}
                )
                | self._data_query_chain.create_chain()
            )

        # all queries are invalid: have some dimensions are missing
        # ToDo: adjust to process multiple datasets
        dataset_id = next(iter(dataset_queries))

        incomplete_queries_chain_inputs = dict(
            **{k: v for k, v in inputs.items() if k != 'dataset_queries'},
            formatted_query_with_missing_dimensions=await query_utils.format_dataset_queries(
                auth_context=auth_context,
                dataset_queries=dataset_queries,
                datasets_dict=chain_state.datasets_dict,
                include_missing_dimensions=True,
                include_default_queries=True,
                include_auto_selects=True,
                availability=chain_state.strong_availability[dataset_id],
            ),
            dataset_queries={dataset_id: dataset_queries[dataset_id]},
        )
        return await self._incomplete_queries_chain.create_chain(
            incomplete_queries_chain_inputs, auth_context.api_key
        )

    async def _run_indicators_selection(self, inputs: dict) -> Runnable:
        service = ChainParameters.get_data_service(inputs)
        auth_context = ChainParameters.get_auth_context(inputs)
        list_datasets = list(ChainParameters.get_datasets_dict(inputs).values())

        meta_factory = IndicatorSelectionFactory(
            config=self._config,
            models_api_key=auth_context.api_key,
            vector_store=await service._get_indicators_vector_store(auth_context),
            matching_index_name=service.channel.matching_index_name,
            indicators_index_name=service.channel.indicators_index_name,
            list_datasets=list_datasets,
        )

        chain_factory = await meta_factory.get_indicator_selection(
            indicator_selection_version=self._config.indicator_selection_version
        )
        chain = chain_factory.create_chain()
        return chain

    def create_preparation_chain(self, auth_context: AuthContext) -> Runnable:
        normalizing_stage_name = "Normalizing Query"
        normalizing_query_stage_callback = StageCallback(
            stage_name=normalizing_stage_name,
            content_appender=self._populate_normalization,
            debug_only=self._config.stages_config.is_stage_debug(normalizing_stage_name),
        )

        named_entities_stage_name = "Extracting Named Entities"
        named_entities_stage_callback = StageCallback(
            stage_name=named_entities_stage_name,
            content_appender=self._populate_named_entities,
            debug_only=self._config.stages_config.is_stage_debug(named_entities_stage_name),
        )

        chain = (
            RunnablePassthrough.assign(
                datasets_dict_indexed=dataset_utils.get_available_datasets,
            )
            # # unpack country groups in the user prompt
            # | RunnablePassthrough.assign(
            #     query_with_expanded_groups=self._group_expander_chain.create_chain,
            # )
            # normalize (summarize) conversation
            | RunnablePassthrough.assign(
                normalized_query=self._normalization_chain.create_chain,
            ).with_config(config=RunnableConfig(callbacks=[normalizing_query_stage_callback]))
            # save 'normalized_query' to separate variable, since it will be overwritten later
            | RunnablePassthrough.assign(normalized_query_raw=lambda d: d["normalized_query"])
            # detect specified datasets and remove them from normalized query
            | (
                RunnablePassthrough.assign(
                    datasets_selection_response=self._datasets_selection_chain.create_chain
                )
                # NOTE: here we overwrite "normalized_query" field
                | self._apply_dataset_selection_response
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Selecting Datasets", self._populate_datasets_dict, debug_only=True
                        ),
                        StageCallback(
                            "Normalized Query with Datasets Removed",
                            self._populate_normalization,
                            debug_only=True,
                        ),
                    ]
                )
            )
            # extract named entities and time range
            | RunnablePassthrough.assign(
                named_entities_response=self._named_entities_chain.create_chain(
                    auth_context.api_key
                ),
                date_time_query_response=self._datetime_chain.create_chain(auth_context.api_key),
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        named_entities_stage_callback,
                        StageCallback(
                            "Extracting Time Range", self._populate_datetime, debug_only=True
                        ),
                    ]
                )
            )
            | RunnablePassthrough.assign(
                country_named_entities=self._get_country_named_entities,
            )
        )

        return chain

    def _create_nonindicators_chain(self) -> Runnable:
        """Chain for selecting non-indicator dimensions"""

        def _prepare_dimension_candidates_for_llm(
            inputs: dict,
        ) -> list[LLMSelectionDimensionCandidate]:
            chain_state = ChainState(**inputs)
            dimension_candidates = chain_state.dimension_candidates
            res = [
                LLMSelectionDimensionCandidate.from_scored_dimension_candidate(
                    candidate=c, index=ix
                )
                for ix, c in enumerate(dimension_candidates)
            ]
            return res

        def _filter_dimension_candidates_by_llm_response(
            inputs: dict,
        ) -> list[ScoredDimensionCandidate]:
            # NOTE: Candidates with same (dimension_id, query_id) may belong to different datasets.
            # However, duplicates by (dataset, dimension, query_id) are NOT expected.
            chain_state = ChainState(**inputs)
            llm_candidates = chain_state.dimension_candidates_for_llm_selection

            selected_ids = chain_state.dimension_values_llm_selection_output.get_selected_ids()
            selected_ids_expanded = (
                LLMSelectionDimensionCandidate.propagate_selection_status_to_duplicates(
                    candidates=llm_candidates, selected_ids=selected_ids
                )
            )

            filtered = [c for c in llm_candidates if c._id in selected_ids_expanded]
            filtered_casted = [c.to_scored_dimension_candidate() for c in filtered]

            return filtered_casted

        async def _populate_nonindicator_weak_queries_stage(stage: Stage, inputs: dict):
            with debug_timer("_populate_nonindicator_weak_queries_stage"):
                chain_state = ChainState(**inputs)
                await query_utils.populate_queries_stage(
                    stage=stage,
                    queries=chain_state.weak_queries_nonindicators,
                    auth_context=chain_state.auth_context,
                    datasets_dict=chain_state.datasets_dict,
                )

        def _candidates_to_queries(inputs: dict) -> DatasetAvailabilityQueriesType:
            with debug_timer("_candidates_to_queries"):
                chain_state = ChainState(**inputs)

                date_time_query_response = chain_state.date_time_query_response
                date_time_query = date_time_query_response.to_query()

                dataset_ids = list(chain_state.datasets_dict.keys())

                dataset_2_dim_2_all_values_term = {
                    ds_id: ds.config.dimension_all_values
                    for ds_id, ds in chain_state.datasets_dict.items()
                }

                return query_utils.dimension_candidates_to_queries(
                    candidates=chain_state.dimension_candidates,
                    date_time_query=date_time_query,
                    dataset_2_dim_2_all_values_term=dataset_2_dim_2_all_values_term,
                    dataset_ids_to_be_present=dataset_ids,
                )

        def _copy_strong_nonindicators_to_strong_queries(inputs: dict) -> dict:
            chain_state = ChainState(**inputs)
            inputs['strong_queries'] = deepcopy(chain_state.strong_queries_nonindicators)
            return inputs

        return (
            # ---------------------------
            # Retrieve candidate from vector store
            # ---------------------------
            (
                RunnablePassthrough.assign(
                    dimension_candidates=self._get_dimension_candidates_from_named_entities,
                )
                # NOTE: we display weak queries to show mapping of candidates to their datasets.
                # candidates, as passed to LLM prompt, will be shown in another stage.
                | RunnablePassthrough.assign(weak_queries_nonindicators=_candidates_to_queries)
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Non-indicator candidates, split by dataset",
                            _populate_nonindicator_weak_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
            )
            | (
                RunnablePassthrough.assign(
                    dimension_candidates_for_llm_selection=_prepare_dimension_candidates_for_llm
                )
                | RunnablePassthrough.assign(
                    dimension_candidates_for_llm_selection=self._add_all_values_to_nonindicator_candidates
                )
                # run LLM selection
                | RunnablePassthrough.assign(
                    dimension_values_llm_selection_output=self._dimensions_selection_chain_factory.create_chain()
                )
                | RunnablePassthrough.assign(
                    dimension_candidates=_filter_dimension_candidates_by_llm_response
                )
                | RunnablePassthrough.assign(strong_queries_nonindicators=_candidates_to_queries)
                | _copy_strong_nonindicators_to_strong_queries
                | self._update_strong_queries_best_attempt_if_possible
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Strong Non-Indicators",
                            self._populate_strong_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
            )
            | (
                # RunnablePassthrough.assign(
                #     strong_queries=self._filter_strong_queries_by_counterparties_presence
                # )
                RunnablePassthrough.assign(strong_queries=self._filter_strong_queries_by_countries)
                | RunnablePassthrough.assign(
                    strong_availability=self._get_availability_from_strong_queries
                )
                # save queries before any further processing
                | self._update_strong_queries_best_attempt_if_possible
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Strong Non-Indicators, filtered by named entities",
                            self._populate_strong_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
            )
        )

    def _create_indicators_chain(self) -> Runnable:
        selecting_indicators_stage_name = "Selecting Indicators"
        selecting_indicators_stage_callback = StageCallback(
            stage_name=selecting_indicators_stage_name,
            content_appender=None,
            debug_only=self._config.stages_config.is_stage_debug(selecting_indicators_stage_name),
        )

        return (
            # -------------------------------------------
            # Select indicators,
            # Filter dimension candidates by availability
            # -------------------------------------------
            #
            # NOTE: notes on availability queries
            #
            # NOTE: availability removes dimension values
            # that do not have valid combinations with other PRESENT dimensions:
            #   availability({'country': [US], 'indicator': ['IND_1', 'IND_2']}) ->
            #   {'country': [US], 'indicator': ['IND_1'], 'frequency': ['A']}
            # NOTE: availability lists available values for dimensions
            # that are NOT PRESENT in the query
            #   availability({'country': [US, FRA], 'indicator': ['IND_1', 'IND_2']}) ->
            #   {'country': [US, FRA], 'indicator': ['IND_1', 'IND_2'], 'frequency': ['A']}
            # NOTE: once we modify query, list of available values may change,
            # - data series present: (a1, b1, c1), (a1, b2, c2)
            # - availability({'a': ['a1']}) -> {'a': ['a1'], 'b': ['b1', 'b2'], 'c': ['c1', 'c2']}
            # - availability({'a': ['a1'], 'b': ['b1']}) -> {'a': ['a1'], 'b': ['b1'], 'c': ['c1']}
            #
            RunnablePassthrough.assign(
                indicators_selection_result=self._run_indicators_selection
            ).with_config(
                # TODO: pass this stage to indicators selection chain to populate
                config=RunnableConfig(callbacks=[selecting_indicators_stage_callback])
            )
            # unpack indicator selection outputs
            | RunnablePassthrough.assign(
                strong_queries=lambda d: d["indicators_selection_result"].queries,
                retrieval_results=lambda d: d["indicators_selection_result"].retrieval_results,
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Strong Queries, with indicators",
                            self._populate_strong_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
            )
            | self._update_strong_queries_best_attempt_if_possible
            | RunnablePassthrough.assign(
                strong_queries=self._filter_queries_by_required_indicator_dims
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Strong Queries, filter by required indicator dimensions",
                            self._populate_strong_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
            )
            | self._update_strong_queries_best_attempt_if_possible
            # filter queries by availability, so that queries we show to user in text
            # match the data we receive by executing the queries.
            | (
                RunnablePassthrough.assign(
                    # "strong_availability" will be used later, when building final dataset queries.
                    # so we cache it here.
                    strong_availability=self._get_availability_from_strong_queries,
                )
                | RunnablePassthrough.assign(
                    strong_queries=self._filter_strong_queries_by_strong_availability
                )
                | self._update_strong_queries_best_attempt_if_possible
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Strong Queries, filter by availability",
                            self._populate_strong_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
            )
        )

    @staticmethod
    def _get_special_dimension_factories(
        processor: SpecialDimensionsProcessor,
    ) -> dict[SpecialDimensionsProcessorType, SpecialDimensionChainFactoryBase]:
        return {
            SpecialDimensionsProcessorType.LHCL: LHCLChainFactory(processor=processor),
        }

    def _create_special_dimension_chain(self) -> Runnable:
        processors = self._config.special_dimensions_processors

        if len(processors) == 0:
            logger.info('no special dimension processors are present for data query tool')
            return RunnableLambda(lambda _: {})

        chains_dict = {}
        for processor in processors:
            factory_mapping = self._get_special_dimension_factories(processor=processor)
            factory = factory_mapping.get(processor.type)
            if not factory:
                raise NotImplementedError(
                    f'Unsupported special dimension processor type: {processor.type}'
                )
            chains_dict[processor.id] = factory.create_chain()

        chain = RunnableParallel(chains_dict)
        logger.info(
            f'created processors for {len(chains_dict)} following special dimensions: '
            f'{list(chains_dict.keys())}'
        )

        return chain.with_config(
            config=RunnableConfig(
                callbacks=[
                    StageCallback(
                        stage_name="Selecting Special Dimensions",
                        content_appender=None,
                        debug_only=True,
                    )
                ]
            )
        )

    def _combine_indicators_and_special_dimensions_chain_outputs(self, inputs: dict) -> dict:
        state = inputs[MetaStateKeys.CHAIN_STATE]
        state['special_dims_outputs'] = inputs[MetaStateKeys.SPECIAL_DIMENSIONS_OUTPUTS]
        return state

    @staticmethod
    def _add_special_dims_to_strong_queries(inputs: dict) -> DatasetAvailabilityQueriesType:
        state = ChainState(**inputs)
        strong_queries = state.strong_queries

        for _, sdim_out in state.special_dims_outputs.items():
            if sdim_out.no_queries():
                # no special dim queries at all - do not filter strong queries
                continue

            for ds_id, ds_strong in strong_queries.items():

                if (ds_sdim_query := sdim_out.dataset_queries.get(ds_id)) is None:
                    # no special dim query for this dataset - do not update it.
                    # NOTE: alternatively, we can remove this dataset query completely.
                    continue

                dim = ds_sdim_query.dimension_id
                if ds_sdim_query.operator != QueryOperator.IN:
                    raise ValueError(f'unexpected query operator: {ds_sdim_query.operator}')

                if (strong_dim_query := ds_strong.dimensions_queries_dict.get(dim)) is not None:
                    # we allow special dimension to be already present in strong queries.
                    # however, this should not happen (at least in the current code).
                    # in this case, add any missing terms
                    missing = set(ds_sdim_query.values).difference(strong_dim_query.values)
                    if missing:
                        strong_dim_query.values.extend(missing)
                else:
                    ds_strong.add_dimension_query(query=deepcopy(ds_sdim_query))

        return strong_queries

    def _create_nonindicators_ok_chain(self) -> Runnable:
        construct_data_query_stage_name = "Constructing Data Query"

        return (
            # --- select indicator dims and special dims ---
            RunnableParallel(
                {
                    MetaStateKeys.CHAIN_STATE: self._create_indicators_chain(),
                    MetaStateKeys.SPECIAL_DIMENSIONS_OUTPUTS: self._create_special_dimension_chain(),
                }
            )
            | self._combine_indicators_and_special_dimensions_chain_outputs
            # --- combine special dim queries with strong queries ---
            | (
                RunnablePassthrough.assign(
                    strong_queries=self._add_special_dims_to_strong_queries
                ).with_config(
                    config=RunnableConfig(
                        callbacks=[
                            StageCallback(
                                "Strong Queries, with special dimensions",
                                self._populate_strong_queries_stage,
                                debug_only=True,
                            )
                        ]
                    )
                )
                | self._update_strong_queries_best_attempt_if_possible
                # filter by availability
                | RunnablePassthrough.assign(
                    strong_availability=self._get_availability_from_strong_queries,
                )
                | RunnablePassthrough.assign(
                    strong_queries=self._filter_strong_queries_by_strong_availability
                )
                | self._update_strong_queries_best_attempt_if_possible
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Strong Queries, with special dimensions, filter by availability",
                            self._populate_strong_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
            )
            # --- create final dataset queries ---
            | (
                RunnablePassthrough.assign(dataset_queries=self._get_dataset_queries)
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
            | self._set_tool_state
            | self._route_based_on_data_query_status
        )

    async def _route_based_on_nonindicators_status(self, inputs: dict) -> Runnable:
        chain_state = ChainState(**inputs)
        country_entities = chain_state.country_named_entities
        strong_queries = chain_state.strong_queries

        # TODO: handle case when strong_queries are empty

        # if there are country entities, but no country dimensions values were found, we need to return "no data"
        if len(country_entities) > 0 and len(strong_queries) == 0:
            logger.info(
                'No country dimension values were found in the query, '
                'but country named entities were found. '
                'Returning "no data" message.'
            )

            country_names = [ne.entity for ne in country_entities]
            data_service = chain_state.data_service
            country_named_entity_type = data_service.get_country_named_entity_type()
            if self._config.messages.no_data_for_country:
                message = self._config.messages.no_data_for_country
            else:
                message = "No data was found for {country_details}. Try to change the query."
            try:
                country_details = f"{country_named_entity_type} {', '.join(country_names)}"
                message = message.format(country_details=country_details)
            except KeyError:
                pass  # key not found in message, keep the original message

            inputs[DataQueryParameters.RESPONSE_FIELD] = message
            target = ChainParameters.get_target(inputs)
            target.append_content(message)
            return RunnableLambda(self._set_tool_state)
        else:
            return self._create_nonindicators_ok_chain()

    async def create_chain(self, inputs: dict | None) -> Runnable:
        if inputs is None:
            raise ValueError("Request context inputs are required")
        auth_context = ChainParameters.get_auth_context(inputs)

        return (
            self.create_preparation_chain(auth_context)
            | self._create_nonindicators_chain()
            | self._route_based_on_nonindicators_status
        )
