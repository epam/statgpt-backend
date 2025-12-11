import asyncio
from copy import deepcopy

from aidial_sdk.chat_completion import Stage
from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough

from common.config import multiline_logger as logger
from common.data.sdmx.common import DimensionVirtualCodeCategory
from common.data.sdmx.v21.dataset import Sdmx21DataSet
from common.schemas.data_query_tool import DataQueryDetails, DataQueryPrompts
from common.schemas.enums import TimePeriodStrategy
from common.utils.timer import debug_timer
from statgpt.chains import CandidatesSelectionSimpleChainFactory
from statgpt.chains.data_query.query_builder import utils as query_utils
from statgpt.default_prompts import data_query_default_prompts
from statgpt.schemas.query_builder import (
    DatasetAvailabilityQueriesType,
    LLMSelectionDimensionCandidate,
)
from statgpt.services.chat_facade import ScoredDimensionCandidate
from statgpt.utils.callbacks import StageCallback
from statgpt.utils.formatters import DatasetAvailabilityQueryFormatter

from .base import DimensionSearchChainFactoryBase
from .schemas import SearchInput


class NonIndicatorsSearchChainFactory(DimensionSearchChainFactoryBase):
    def __init__(self, config: DataQueryDetails):
        super().__init__(config)

        prompts: DataQueryPrompts = self._config.prompts

        self._dimensions_selection_chain_factory = CandidatesSelectionSimpleChainFactory(
            llm_model_config=self._config.llm_models.dimensions_selection_model_config,
            system_prompt=prompts.validation_system_prompt
            or data_query_default_prompts.validation_system_prompt,
            user_prompt=prompts.validation_user_prompt
            or data_query_default_prompts.validation_user_prompt,
            candidates_key="dimension_candidates_for_llm_selection",
        )

    async def _get_dimension_candidates_from_named_entities(
        self, inputs: dict
    ) -> list[ScoredDimensionCandidate]:
        with debug_timer("_get_dimension_candidates_from_named_entities"):
            search_input = SearchInput(**inputs)

            filtered_named_entities = [
                ne
                for ne in search_input.named_entities_response.entities
                if ne.entity_type.lower() != "dataset"
            ]
            version_ids = set(
                (ds.version.version_data_id for ds in search_input.datasets_dict.values())
            )
            tasks = []
            for entity in filtered_named_entities:
                tasks.append(
                    search_input.data_service.search_dimensions_scored(
                        entity.to_query(),
                        auth_context=search_input.auth_context,
                        k=self._config.candidates_per_entity,
                        dataset_versions=version_ids,
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
        search_input = SearchInput(**inputs)
        queries = search_input.strong_queries

        at_least_one_country_selected: bool = False
        datasets_entity_ids_without_country_query = set()
        datasets_source_ids_without_country_query = set()

        for dataset_id, query in queries.items():
            dataset = search_input.datasets_dict[dataset_id].data
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

            country_entities = search_input.country_named_entities
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
        search_input = SearchInput(**inputs)
        dimension_candidates = search_input.dimension_candidates_for_llm_selection
        datasets_dict = search_input.datasets_dict
        index = len(dimension_candidates)
        for versioned_ds in datasets_dict.values():
            ds = versioned_ds.data
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

    @staticmethod
    def _prepare_dimension_candidates_for_llm(
        inputs: dict,
    ) -> list[LLMSelectionDimensionCandidate]:
        search_input = SearchInput(**inputs)
        dimension_candidates = search_input.dimension_candidates
        res = [
            LLMSelectionDimensionCandidate.from_scored_dimension_candidate(candidate=c, index=ix)
            for ix, c in enumerate(dimension_candidates)
        ]
        return res

    @staticmethod
    def _filter_dimension_candidates_by_llm_response(
        inputs: dict,
    ) -> list[ScoredDimensionCandidate]:
        search_input = SearchInput(**inputs)
        llm_candidates = search_input.dimension_candidates_for_llm_selection

        assert (
            search_input.dimension_values_llm_selection_output is not None
        ), "dimension_values_llm_selection_output must be set"
        selected_ids = search_input.dimension_values_llm_selection_output.get_selected_ids()
        selected_ids_expanded = (
            LLMSelectionDimensionCandidate.propagate_selection_status_to_duplicates(
                candidates=llm_candidates, selected_ids=selected_ids
            )
        )

        filtered = [c for c in llm_candidates if c._id in selected_ids_expanded]
        filtered_casted = [c.to_scored_dimension_candidate() for c in filtered]

        return filtered_casted

    @staticmethod
    async def _populate_nonindicator_weak_queries_stage(stage: Stage, inputs: dict):
        with debug_timer("_populate_nonindicator_weak_queries_stage"):
            search_input = SearchInput.model_validate(inputs)
            await DatasetAvailabilityQueryFormatter.populate_queries_stage(
                stage=stage,
                queries=search_input.weak_queries_nonindicators,
                auth_context=search_input.auth_context,
                datasets_dict=search_input.datasets_dict,
            )

    def _candidates_to_queries(self, inputs: dict) -> DatasetAvailabilityQueriesType:
        with debug_timer("_candidates_to_queries"):
            search_input = SearchInput(**inputs)

            if self._config.time_period_strategy is TimePeriodStrategy.BEFORE:
                date_time_query_response = search_input.date_time_query_response
                date_time_query = date_time_query_response.to_query()
            else:
                date_time_query = None

            dataset_ids = list(search_input.datasets_dict.keys())

            dataset_2_dim_2_all_values_term = {
                ds_id: ds.data.config.dimension_all_values
                for ds_id, ds in search_input.datasets_dict.items()
            }

            return query_utils.dimension_candidates_to_queries(
                candidates=search_input.dimension_candidates,
                date_time_query=date_time_query,
                dataset_2_dim_2_all_values_term=dataset_2_dim_2_all_values_term,
                dataset_ids_to_be_present=dataset_ids,
            )

    @staticmethod
    def _copy_strong_nonindicators_to_strong_queries(inputs: dict) -> dict:
        search_input = SearchInput(**inputs)
        inputs['strong_queries'] = deepcopy(search_input.strong_queries_nonindicators)
        return inputs

    def create(self) -> Runnable:
        return (
            (
                RunnablePassthrough.assign(
                    dimension_candidates=self._get_dimension_candidates_from_named_entities,
                )
                | RunnablePassthrough.assign(weak_queries_nonindicators=self._candidates_to_queries)
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Non-indicator candidates, split by dataset",
                            self._populate_nonindicator_weak_queries_stage,
                            debug_only=True,
                        )
                    ]
                )
            )
            | (
                RunnablePassthrough.assign(
                    dimension_candidates_for_llm_selection=self._prepare_dimension_candidates_for_llm
                )
                | RunnablePassthrough.assign(
                    dimension_candidates_for_llm_selection=self._add_all_values_to_nonindicator_candidates
                )
                | RunnablePassthrough.assign(
                    dimension_values_llm_selection_output=self._dimensions_selection_chain_factory.create_chain()
                )
                | RunnablePassthrough.assign(
                    dimension_candidates=self._filter_dimension_candidates_by_llm_response
                )
                | RunnablePassthrough.assign(
                    strong_queries_nonindicators=self._candidates_to_queries
                )
                | self._copy_strong_nonindicators_to_strong_queries
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
                RunnablePassthrough.assign(strong_queries=self._filter_strong_queries_by_countries)
                | RunnablePassthrough.assign(
                    strong_availability=self._get_availability_from_strong_queries
                )
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
