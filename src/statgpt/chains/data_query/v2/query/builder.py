import asyncio
from copy import deepcopy
from itertools import groupby
from operator import attrgetter

from aidial_sdk.chat_completion import Stage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnablePassthrough

from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.data.base import (
    DataSet,
    DataSetAvailabilityQuery,
    DataSetQuery,
    DimensionQuery,
    DimensionType,
    Query,
    QueryOperator,
)
from common.data.sdmx.common import DimensionVirtualCodeCategory
from common.data.sdmx.v21.dataset import Sdmx21DataSet
from common.schemas import ChannelConfig, DataQueryDetails
from common.schemas.data_query_tool import DataQueryMessages, DataQueryPrompts
from statgpt.chains import CandidatesSelectionSimpleChainFactory
from statgpt.chains.data_query.base import BaseDataQueryFactory
from statgpt.chains.data_query.parameters import DataQueryParameters
from statgpt.chains.parameters import ChainParameters
from statgpt.default_prompts.v2 import DefaultPrompts
from statgpt.schemas.query_builder import (
    ChainState,
    DatasetAvailabilityQueriesType,
    DateTimeQueryResponse,
    LLMSelectionDimensionCandidate,
    NamedEntitiesResponse,
    NamedEntity,
    QueryBuilderAgentState,
)
from statgpt.services import ScoredDimensionCandidate
from statgpt.utils.callbacks import StageCallback
from statgpt.utils.dataset_formatter import DatasetFormatterConfig, DatasetListFormatter
from statgpt.utils.request_context import RequestContext

from . import utils as v2_query_utils
from .data_query import DataQueryChain
from .datasets_selection import DataSetsSelectionChain
from .datetime_chain import DateTimeDimensionChain
from .group_expander_chain import GroupExpanderChain
from .incomplete_queries import IncompleteQueriesChain
from .indicator_selection.factory import IndicatorSelectionFactory
from .multiple_datasets import MultipleDatasetsChain
from .named_entities import NamedEntitiesChain
from .nodata import NoDataChain
from .normalization import NormalizationChain


class QueryBuilderFactoryV2(BaseDataQueryFactory):
    def __init__(self, config: DataQueryDetails, channel_config: ChannelConfig):
        super().__init__(config, channel_config)

        prompts: DataQueryPrompts = self._config.prompts
        messages: DataQueryMessages = self._config.messages

        self._datetime_chain = DateTimeDimensionChain(
            prompts.datetime_prompt or DefaultPrompts.DATETIME_PROMPT
        )
        self._group_expander_chain = GroupExpanderChain(
            prompts.group_expander_prompt or DefaultPrompts.GROUP_EXPANDER_PROMPT,
            prompts.group_expander_fallback_prompt or DefaultPrompts.GROUP_EXPANDER_FALLBACK_PROMPT,
        )
        self._normalization_chain = NormalizationChain(
            prompts.normalization_prompt or DefaultPrompts.NORMALIZATION_PROMPT
        )
        self._named_entities_chain = NamedEntitiesChain(
            prompts.named_entities_prompt or DefaultPrompts.NAMED_ENTITIES_PROMPT
        )
        self._datasets_selection_chain = DataSetsSelectionChain(
            prompts.dataset_selection_prompt or DefaultPrompts.DATASET_SELECTION_PROMPT
        )
        self._dimensions_selection_chain_factory = CandidatesSelectionSimpleChainFactory(
            prompts.validation_system_prompt or DefaultPrompts.VALIDATION_SYSTEM_PROMPT,
            prompts.validation_user_prompt or DefaultPrompts.VALIDATION_USER_PROMPT,
            "dimension_candidates_for_llm_selection",
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
            prompts.incomplete_queries_prompt or DefaultPrompts.INCOMPLETE_QUERIES_PROMPT
        )

    @staticmethod
    async def _get_available_datasets(inputs: dict) -> dict[str, DataSet]:
        data_service = ChainParameters.get_data_service(inputs)
        auth_context = ChainParameters.get_auth_context(inputs)
        datasets = await data_service.list_available_datasets(auth_context)
        return {ds.entity_id: ds for ds in datasets}

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
        results = await asyncio.gather(*tasks)

        candidates_all: list[ScoredDimensionCandidate] = []
        for result in results:
            candidates_all.extend(result)

        candidates_dedup = list(set(candidates_all))
        candidates_dedup = sorted(candidates_dedup, key=lambda x: x.score, reverse=True)

        return candidates_dedup

    @classmethod
    def _nonindicator_candidates_to_availability_queries(
        cls, inputs: dict
    ) -> DatasetAvailabilityQueriesType:
        chain_state = ChainState(**inputs)
        datasets_dict = chain_state.datasets_dict
        date_time_query_response = chain_state.date_time_query_response
        dimension_candidates = chain_state.dimension_candidates

        # gather dataset dimension queries
        dataset_dimension_queries = {}
        for candidate in dimension_candidates:
            dataset_query = dataset_dimension_queries.setdefault(candidate.dataset_id, {})

            dataset = datasets_dict[candidate.dataset_id]
            dimension_all_values = dataset.config.dimension_all_values
            dimension_id = candidate.dimension_category.dimension_id
            all_value = dimension_all_values.get(dimension_id)
            if all_value is not None and candidate.query_id == all_value.id:
                dataset_query[dimension_id] = "*"  # all values selected
            else:
                dimension_query = dataset_query.setdefault(
                    candidate.dimension_category.dimension_id, set()
                )
                if dimension_query != "*":
                    # if we already have all values selected, do not add more values
                    # otherwise, add the candidate query_id to the dimension query
                    dimension_query.add(candidate.query_id)

        # cast to availability queries
        result = {
            dataset_id: cls._to_availability_query(dataset_query, date_time_query_response)
            for dataset_id, dataset_query in dataset_dimension_queries.items()
        }
        # ensure all datasets have queries, even if there are no non-indicator candidates.
        for dataset_id, dataset in datasets_dict.items():
            if dataset_id not in result:
                result[dataset_id] = cls._to_availability_query({}, date_time_query_response)

        return result

    @staticmethod
    def _to_availability_query(
        query: dict, date_time_query_response: DateTimeQueryResponse
    ) -> DataSetAvailabilityQuery:
        availability_query = DataSetAvailabilityQuery()
        for dimension_id, value in query.items():
            if value == "*":
                dimension_query = DimensionQuery(
                    dimension_id=dimension_id, values=[], operator=QueryOperator.ALL
                )
            else:
                dimension_query = DimensionQuery(
                    dimension_id=dimension_id, values=value, operator=QueryOperator.IN
                )
            availability_query.add_dimension_query(dimension_query)
        date_time_query = date_time_query_response.to_query()
        if date_time_query:
            availability_query.add_dimension_query(date_time_query)
        return availability_query

    @staticmethod
    async def _get_availability(inputs: dict, queries_key: str) -> DatasetAvailabilityQueriesType:
        chain_state = ChainState(**inputs)
        auth_context = ChainParameters.get_auth_context(inputs)
        queries: DatasetAvailabilityQueriesType = inputs[queries_key]
        if len(queries) == 0:
            return {}
        tasks = []
        for dataset_id, query in queries.items():
            dataset = chain_state.datasets_dict[dataset_id]
            tasks.append(dataset.availability_query(query, auth_context))
        task_results = await asyncio.gather(*tasks)
        result = {dataset_id: query for dataset_id, query in zip(queries.keys(), task_results)}
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

    @staticmethod
    def _filter_strong_queries_by_countries_presence(
        inputs: dict,
    ) -> DatasetAvailabilityQueriesType:
        """
        Idea is to remove all dataset queries without country query,
        if there is at least one dataset query with country query.

        Original reason is to remove datasets not having requested country data,
        for example STATCAN (Statistics of Canada) does not have data for France.
        We can't filter STATCAN dataset simply by availability of France dim value,
        since there is no France in the first place -
        it won't be selected by LLM and there would be nothing to use in availability data.

        See issue 75 for reference.
        """
        logger.info('_filter_strong_queries_by_countries_presence()')

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
                # we don't the country dimension for this dataset.
                # it's also possible that country dimension is not present at all.
                # thus we can't filter this dataset out.
                continue

            country_query = query.dimensions_queries_dict.get(country_dim.entity_id)

            if country_query and (
                country_query.operator == QueryOperator.ALL or country_query.values
            ):
                at_least_one_country_selected = True
            else:
                datasets_entity_ids_without_country_query.add(dataset_id)
                datasets_source_ids_without_country_query.add(dataset.source_id)

        if at_least_one_country_selected:
            # remove all datasets with country dimension and without country query
            if datasets_entity_ids_without_country_query:
                logger.info(
                    'found at least one country query. '
                    f'removing following {len(datasets_source_ids_without_country_query)} '
                    'dataset queries, since they do not contain country query: '
                    f'{datasets_source_ids_without_country_query}'
                )
                queries = {
                    dataset_id: query
                    for dataset_id, query in queries.items()
                    if dataset_id not in datasets_entity_ids_without_country_query
                }
            else:
                logger.info(
                    'found at least one country query. '
                    'keeping all dataset queries: they all either contain '
                    'a country query, or do not have country dimension specified'
                )
        else:
            # if the named entity of type country was found in the query, but no country dimensions were set,
            # we need to clear the strong queries
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
    def filter_candidates_by_queries(
        queries: DatasetAvailabilityQueriesType,
        dimension_candidates: list[ScoredDimensionCandidate],
    ) -> list[ScoredDimensionCandidate]:
        # TODO: not used

        result = []
        for candidate in dimension_candidates:
            dataset_query = queries.get(candidate.dataset_id)
            if not dataset_query:
                continue
            if (dimension_id := candidate.dimension_category.dimension_id) in dataset_query:
                dimension_query = dataset_query[dimension_id]
                if candidate.query_id in dimension_query.values:
                    result.append(candidate)
        return result

    @staticmethod
    def _convert_dimension_candidates_model_for_llm_selection(
        inputs: dict,
    ) -> list[LLMSelectionDimensionCandidate]:
        chain_state = ChainState(**inputs)
        dimension_candidates = chain_state.dimension_candidates
        dimension_candidates_for_llm_selection = []
        for ix, c in enumerate(dimension_candidates):
            dataset = chain_state.datasets_dict[c.dataset_id]
            if not isinstance(dataset, Sdmx21DataSet):
                raise ValueError(
                    f'Dataset "{dataset.source_id}" is not an instance of InMemorySdmx21DataSet'
                )
            c = LLMSelectionDimensionCandidate.from_scored_dimension_candidate(
                candidate=c, index=ix
            )
            dimension_candidates_for_llm_selection.append(c)

        return dimension_candidates_for_llm_selection

    @staticmethod
    def _append_non_indicator_all_values(
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

    @staticmethod
    def _filter_irrelevant_dimension_values_by_llm(inputs: dict) -> list[ScoredDimensionCandidate]:
        # NOTE: Candidates with same (dimension_id, query_id) may belong to different datasets.
        # However, duplicates by (dataset, dimension, query_id) are NOT expected.
        logger.info('_filter_irrelevant_dimension_values_by_llm()')
        chain_state = ChainState(**inputs)
        candidates = chain_state.dimension_candidates_for_llm_selection

        selected_ids = chain_state.dimension_values_llm_selection_output.get_selected_ids()
        selected_ids_expanded = (
            LLMSelectionDimensionCandidate.propagate_selection_status_to_duplicates(
                candidates=candidates, selected_ids=selected_ids
            )
        )
        logger.info(f'selected ids ({len(selected_ids)} items): {selected_ids}')
        logger.info(
            f'selected ids expanded ({len(selected_ids_expanded)} items): {selected_ids_expanded}'
        )
        filtered = [c for c in candidates if c._id in selected_ids_expanded]
        logger.info(f'selected candidates ({len(filtered)} items): {filtered}')

        filtered_casted = [c.to_scored_dimension_candidate() for c in filtered]

        return filtered_casted

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
        datasets_dict = chain_state.datasets_dict
        auth_context = ChainParameters.get_auth_context(inputs)

        content = await DatasetListFormatter(
            DatasetFormatterConfig.model_validate(
                dict(
                    citation=dict(
                        use_provider=False,
                        use_last_updated=False,
                        use_url=False,
                    ),
                    use_description=False,
                )
            ),
            auth_context,
        ).format(datasets_dict.values())

        stage.append_content(content)

    async def _populate_queries_stage(self, stage: Stage, inputs: dict, queries_key: str):
        queries: DatasetAvailabilityQueriesType = inputs.get(queries_key, {})
        if not queries:
            stage.append_content("No queries")
            return

        chain_state = ChainState(**inputs)
        content = await v2_query_utils.format_availability_queries(
            auth_context=chain_state.auth_context,
            dataset_queries=queries,
            datasets_dict=chain_state.datasets_dict,
            format_values_as_list=True,
            add_value_ids=True,
            add_citation=False,
        )
        stage.append_content(content)

    async def _populate_nonindicator_weak_queries_stage(self, stage: Stage, inputs: dict):
        await self._populate_queries_stage(stage, inputs, "weak_queries_no_indicators")

    async def _populate_strong_queries_stage(self, stage: Stage, inputs: dict):
        await self._populate_queries_stage(stage, inputs, "strong_queries")

    def _update_strong_queries_best_attempt_if_possible(self, inputs: dict):
        """
        Here we save a copy of current version of non-empty strong queries to a separate field.
        This field is used as our best attempt to build non-empty strong queries.
        Generally, we would want to call this function after every update to strong queries.
        """
        chain_state = ChainState(**inputs)
        strong_queries = chain_state.strong_queries
        if not strong_queries:
            return inputs
        strong_filtered = v2_query_utils.filter_empty_dataset_availability_queries(
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
            return await v2_query_utils.format_dataset_queries(
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
                formatted_queries = await v2_query_utils.format_availability_queries(
                    auth_context, strong_queries_best_attempt, datasets_dict, header_level=4
                )
                msg += f"\n\n{formatted_queries}"
            return msg

    async def _populate_dataset_queries(self, stage: Stage, inputs: dict):
        chain_state = ChainState(**inputs)
        stage.append_content(chain_state.dataset_queries_formatted_str)

    @staticmethod
    def _set_dimension_query_from_default_or_available_values(
        dim_id,
        dim_type: DimensionType,
        dataset_id,
        default_queries,
        availability: Query | None,
    ):
        """
        Try to set dimension query for dimension absent in strong queries.
        Use default queries if available, otherwise use available values.
        """

        if default_queries:
            default_query = default_queries[0]
            if not default_query.values:
                logger.info(
                    f'No default values for "{dim_id}" dimension ' f'in "{dataset_id}" dataset'
                )
                return

            if dim_type != DimensionType.CATEGORY:
                return DimensionQuery.from_default_query(default_query, dim_id)

            if availability is None or not availability.values:
                logger.info(
                    f'No available values extracted for "{dim_id}" dimension '
                    f'in "{dataset_id}" dataset'
                )
                return

            available_values = availability.values

            # filter default values by availability
            filtered_defaults = set(default_query.values).intersection(available_values)
            if not filtered_defaults:
                logger.info(
                    'No default values left after filtering by availability for '
                    f'"{dim_id}" dimension in "{dataset_id}" dataset'
                )
                return

            return DimensionQuery(
                values=list(filtered_defaults),
                operator=default_query.operator,
                dimension_id=dim_id,
                is_default=True,
            )

        logger.info(
            f'No default queries for "{dim_id}" dimension in "{dataset_id}" dataset. '
            'Will try to auto-set dimension queries using availability data.'
        )

        if dim_type != DimensionType.CATEGORY:
            logger.info(
                f'Can\'t auto-set query for "{dim_id}" dimension '
                f'in "{dataset_id}" dataset, since it\'s not a categorical dimension.'
            )
            return

        if availability is None or not availability.values:
            logger.info(
                f'No available values extracted for "{dim_id}" dimension '
                f'in "{dataset_id}" dataset'
            )
            return

        available_values = availability.values

        # TODO: make k_low and k_high configurable
        k_low = 10
        k_high = 40
        if len(available_values) > k_low:
            logger.info(
                f'Too many available values ({len(available_values)}) '
                f'for "{dim_id}" dimension without default queries '
                f'in "{dataset_id}" dataset. Can\'t auto-set dimension query. '
                f'Sample values: {available_values[:10]}'
            )

            # TODO: ask clarifications. propose to select from available values.
            # Need to:
            # 0. ask non-indicator dim clarifictions ONLY after
            # indicators clarifiactions are resolved.
            # 1. store them in some structure (field, object)
            # 2. ask LLM either to explicitly list them,
            # or to provide a template str to place them to.

            if len(available_values) > k_high:
                # TODO: do not list all available values in clarification question
                # samples = available_values[:k_high]
                pass

            return

        logger.info(
            f'Auto-setting dimension query for "{dim_id}" dimension '
            f'in "{dataset_id}" dataset to following '
            f'{len(available_values)} available values: {available_values}'
        )
        return DimensionQuery(
            values=list(available_values),  # shallow copy should be enough
            operator=QueryOperator.ALL,
            dimension_id=dim_id,
            is_default=False,
        )

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

    def _get_dataset_queries(self, inputs: dict) -> dict[str, DataSetQuery]:
        """Create final dataset queries. Set missing dimensions if possible."""
        chain_state = ChainState(**inputs)
        datasets_dict = chain_state.datasets_dict
        strong_queries = chain_state.strong_queries
        strong_availability = chain_state.strong_availability

        if not strong_queries:
            return {}
        queries_filtered = v2_query_utils.filter_empty_dataset_availability_queries(
            queries=strong_queries
        )
        if not queries_filtered:
            return {}

        dataset_queries: dict[str, DataSetQuery] = {}
        for dataset_id, dataset_query in queries_filtered.items():
            dataset = datasets_dict[dataset_id]
            ds_default_queries = dataset.config.dimension_default_queries
            ds_dimension_queries: dict[str, DimensionQuery] = {
                d.dimension_id: d for d in dataset_query.dimensions_queries
            }
            is_ds_query_valid = True
            ds_strong_availability_query = strong_availability.get(
                dataset_id, DataSetAvailabilityQuery()
            )

            # now we check if dataset query is valid:
            # whether a query for each dataset dimension is either present, or could be auto-set.
            for dimension in dataset.dimensions():
                dim_id = dimension.entity_id
                if dim_id in ds_dimension_queries:
                    # dimension is already present in strong queries.
                    # continue checking next dimensions.
                    continue

                default_queries = ds_default_queries.get(dim_id)
                availability = ds_strong_availability_query.dimensions_queries_dict.get(dim_id)

                if dimension.dimension_type == DimensionType.DATETIME:
                    chain_state = ChainState(**inputs)
                    dtqr = chain_state.date_time_query_response
                    if dtqr.time_period_specified:
                        logger.info(
                            f'there is an empty time period filter in dataset "{dataset_id}". '
                            'LLM detected that user specified time period filter to be empty. '
                            f'keeping empty time filter, not setting default'
                        )
                        ds_dimension_queries[dim_id] = DimensionQuery(
                            values=['', ''],
                            operator=QueryOperator.BETWEEN,
                            dimension_id=dim_id,
                            is_default=False,
                        )
                        continue
                    else:
                        logger.info(
                            f'there is an empty time period filter in dataset "{dataset_id}". '
                            'LLM detected that user did not specify time period. '
                            f'using default time period: {default_queries}'
                        )

                dim_query = self._set_dimension_query_from_default_or_available_values(
                    dim_id=dim_id,
                    dim_type=dimension.dimension_type,
                    dataset_id=dataset.source_id,
                    default_queries=default_queries,
                    availability=availability,
                )
                if dim_query is not None:
                    ds_dimension_queries[dim_id] = dim_query
                else:
                    # this dimension query is missing, hence the whole dataset query is invalid.
                    # statgpt will ask user to fill missing dimensions for invalid queries.
                    # do not break here, continue checking next dimensions,
                    # since we need to detect ALL missing dimensions in each data query.
                    is_ds_query_valid = False

            dataset_queries[dataset_id] = DataSetQuery(
                # indicator_query=dataset_query.indicator_query,
                dimensions_queries=list(ds_dimension_queries.values()),
                is_valid=is_ds_query_valid,
            )

        return dataset_queries

    def _map_dimension_ids_to_names(self, inputs: dict) -> dict[str, dict[str, dict[str, str]]]:
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

        if len(dataset_queries) > 1:
            return await self._multiple_datasets_chain.create_chain()

        # only one dataset query
        dataset_id = next(iter(dataset_queries))
        dataset_query = dataset_queries[dataset_id]

        if dataset_query.is_valid:
            return self._data_query_chain.create_chain()
        else:
            # some dimensions are missing
            incomplete_queries_chain_inputs = dict(
                formatted_query_with_missing_dimensions=await v2_query_utils.format_dataset_queries(
                    auth_context=auth_context,
                    dataset_queries=dataset_queries,
                    datasets_dict=chain_state.datasets_dict,
                    include_missing_dimensions=True,
                    include_default_queries=True,
                    include_auto_selects=True,
                    availability=chain_state.strong_availability[dataset_id],
                ),
                **inputs,
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
            vector_store=service._get_indicators_vector_store(auth_context),
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
                datasets_dict_indexed=self._get_available_datasets,
            )
            # unpack country groups in the user prompt
            | RunnablePassthrough.assign(
                query_with_expanded_groups=self._group_expander_chain.create_chain,
            )
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

    def _create_non_indicator_dimensions_chain(self) -> Runnable:
        return (
            # ---------------------------
            # Select dimension candidates
            # ---------------------------
            (
                RunnablePassthrough.assign(
                    dimension_candidates=self._get_dimension_candidates_from_named_entities,
                )
                | RunnablePassthrough.assign(
                    weak_queries_no_indicators=self._nonindicator_candidates_to_availability_queries
                )
                # NOTE: there is no reason to filter non-indicator candidates by availability,
                # since we assume that vector store contains only available values.
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Weak Queries, without indicators",
                            self._populate_nonindicator_weak_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
            )
            # create strong queries by validating by LLM (validation chain)
            | (
                RunnablePassthrough.assign(
                    dimension_candidates_for_llm_selection=self._convert_dimension_candidates_model_for_llm_selection
                )
                | RunnablePassthrough.assign(
                    dimension_candidates_for_llm_selection=self._append_non_indicator_all_values
                )
                | RunnablePassthrough.assign(
                    dimension_values_llm_selection_output=self._dimensions_selection_chain_factory.create_chain()
                )
                | RunnablePassthrough.assign(
                    dimension_candidates=self._filter_irrelevant_dimension_values_by_llm
                )
                | RunnablePassthrough.assign(
                    strong_queries=self._nonindicator_candidates_to_availability_queries
                )
                # | RunnablePassthrough.assign(
                #     strong_queries=self._filter_strong_queries_by_counterparties_presence
                # )
                | RunnablePassthrough.assign(
                    strong_queries=self._filter_strong_queries_by_countries_presence
                )
                | RunnablePassthrough.assign(
                    strong_availability=self._get_availability_from_strong_queries
                )
                | self._update_strong_queries_best_attempt_if_possible
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Strong Queries, without indicators",
                            self._populate_strong_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
                # TODO: currentlly we don't filter non-indicator candidates by availability,
                # which can lead to (near) bug.
                # For example, there are 0 series in IMF:CPI(3.0.0)
                # for "Ukraine" and "Harmonized index of consumer prices" Index Type,
                # however both dimension values are available in the dataset.
                # In this case, there is no need to run indicators search,
                # since there are no indicaotrs available.
                # NOTE: vector store contains only avialable values.
                # NOTE: thus problem could be only with incompatible combination
                # of non-indicator dimensions, which seems to be very rare.
            )
        )

    def _create_indicators_selection_chain(self) -> Runnable:
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
            # NOTE #1: availability:
            # 1. removes dimension values that do not have valid combinations with other PRESENT dimensions
            # 2. lists available values for dimensions that are NOT PRESENT in the query
            # Example:
            # - dataset: "IMF.RES:WORLD_ECONOMIC_OUTLOOK(5.0.0)"
            # - availability({'country': [111], 'indicator': ['LP', 'TTPCH']}) -> {'country': [111], 'indicator': ['LP'], 'frequency': ['A']}
            # - availability({'country': [110, 111], 'indicator': ['LP', 'TTPCH']}) -> {'country': [110, 111], 'indicator': ['LP', 'TTPCH'], 'frequency': ['A']}
            #
            # NOTE #2. once we modify query, list of available values may change,
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
                            "Strong Queries, with indicators, before post-processing",
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
                            "Strong Queries, with indicators, filtered by required indicator dimensions",
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
                    strong_availability=self._get_availability_from_strong_queries,
                    strong_queries_before_availability=lambda d: deepcopy(d["strong_queries"]),
                )
                | RunnablePassthrough.assign(
                    strong_queries=self._filter_strong_queries_by_strong_availability
                )
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Final Strong Queries, with indicators, filtered by availability",
                            self._populate_strong_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
            )
        )

    def _create_query_with_indicators_chain(self) -> Runnable:
        stage_name = "Constructing Data Query"
        constructing_stage_callback = StageCallback(
            stage_name=stage_name,
            content_appender=self._populate_dataset_queries,
            debug_only=self._config.stages_config.is_stage_debug(stage_name),
        )

        return (
            self._create_indicators_selection_chain()
            | (
                RunnablePassthrough.assign(dataset_queries=self._get_dataset_queries)
                | RunnablePassthrough.assign(
                    dimension_id_to_name=self._map_dimension_ids_to_names,
                    dataset_queries_formatted_str=self._format_dataset_queries,
                )
            ).with_config(config=RunnableConfig(callbacks=[constructing_stage_callback]))
            | self._set_tool_state
            | self._route_based_on_data_query_status
        )

    async def _route_based_on_non_indicator_data_query_status(self, inputs: dict) -> Runnable:
        chain_state = ChainState(**inputs)
        country_entities = chain_state.country_named_entities

        strong_queries = chain_state.strong_queries
        # if there are country entities, but no country dimensions values were found, we need to return "no data"
        if len(country_entities) > 0 and len(strong_queries) == 0:
            logger.info(
                'No country dimension values were found in the query, but country named entities were found. '
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
            return self._create_query_with_indicators_chain()

    async def create_chain(self, request_context: RequestContext) -> Runnable:
        if request_context.inputs is None:
            raise ValueError("Request context inputs are required")
        auth_context = ChainParameters.get_auth_context(request_context.inputs)

        return (
            self.create_preparation_chain(auth_context)
            | self._create_non_indicator_dimensions_chain()
            | self._route_based_on_non_indicator_data_query_status
        )
