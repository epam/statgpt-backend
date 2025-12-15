from copy import deepcopy

from langchain_core.runnables import Runnable, RunnablePassthrough

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.schemas.query_builder import (
    ChainState,
    DatasetAvailabilityQueriesType,
    IndicatorsSearchResult,
    RetrievalStageDescription,
    RetrievalStagesResults,
)
from statgpt.app.services.chat_facade import VersionedDataSet
from statgpt.app.services.hybrid_searcher import HybridSearcher, HybridSearchResult
from statgpt.app.utils.dial_stages import optional_timed_stage
from statgpt.app.utils.formatters import DatasetAvailabilityQueryFormatter
from statgpt.common.data.base import DataSetAvailabilityQuery
from statgpt.common.schemas import DataQueryDetails
from statgpt.common.schemas.enums import TimePeriodStrategy

from .base import IndicatorSelectionBase
from .format_utils import (
    format_hybrid_final_queries,
    format_hybrid_llm_scored,
    format_hybrid_scored_dicts,
)


class IndicatorsSelectionHybrid(IndicatorSelectionBase):
    _RETRIEVAL_STAGES_ORDERED = [
        RetrievalStageDescription(
            field_name='lexical',
            short_name='lexical',
            description='lexical search results',
        ),
        RetrievalStageDescription(
            field_name='semantic',
            short_name='semantic',
            description='semantic search results',
        ),
        RetrievalStageDescription(
            field_name='llm_scored',
            short_name='llm_scored',
            description='LLM-scored indicators',
        ),
        RetrievalStageDescription(
            field_name='final',
            short_name='final',
            description='Final search results, filtered by availability',
        ),
    ]
    _HYBRID_SELECTION_STAGE_NAME = "Hybrid Indicators Selection"

    _search: HybridSearcher

    def __init__(self, config: DataQueryDetails, search: HybridSearcher):
        self._config = config
        self._search = search

    async def _get_search_result(self, inputs: dict) -> HybridSearchResult:
        chain_state = ChainState.model_validate(inputs)

        is_debug = self._config.stages_config.is_stage_debug(self._HYBRID_SELECTION_STAGE_NAME)
        enabled = (is_debug and chain_state.show_debug_stages) or not is_debug

        with optional_timed_stage(
            choice=chain_state.choice, name=self._HYBRID_SELECTION_STAGE_NAME, enabled=enabled
        ) as stage:
            return await self._search.search(
                stage=stage,
                query=chain_state.normalized_query,
                datasets=chain_state.datasets_dict,
                named_entities=chain_state.named_entities_response.entities,
                period=chain_state.date_time_query_response,
                availability_queries=chain_state.strong_availability,
            )

    async def _show_indicator_queries_stage(self, inputs: dict) -> dict:
        chain_state = ChainState.model_validate(inputs)

        datasets_dict: dict[str, VersionedDataSet] = ChainParameters.get_datasets_dict(inputs)
        search_result: HybridSearchResult = inputs['search_result']

        with optional_timed_stage(
            choice=chain_state.choice,
            name="[DEBUG] Indicator Queries",
            enabled=chain_state.show_debug_stages,
        ) as stage:
            if stage:
                indicator_queries: DatasetAvailabilityQueriesType = {
                    dataset_id: DataSetAvailabilityQuery.from_dimension_queries_list(dim_queries)
                    for dataset_id, dim_queries in search_result.final_queries.items()
                }

                await DatasetAvailabilityQueryFormatter.populate_queries_stage(
                    stage=stage,
                    queries=indicator_queries,
                    auth_context=chain_state.auth_context,
                    datasets_dict=datasets_dict,
                )

        return inputs

    def _get_final_queries(self, inputs: dict) -> dict[str, DataSetAvailabilityQuery]:
        """
        combine final indicator queries with strong nonindicators queries
        """
        chain_state = ChainState.model_validate(inputs)

        # I don't understand why time period is added here again, probably can be removed
        if self._config.time_period_strategy is TimePeriodStrategy.BEFORE:
            date_time_query = chain_state.date_time_query_response.to_query()
        else:
            date_time_query = None

        search_result: HybridSearchResult = inputs['search_result']
        final = search_result.final_queries

        result_queries = {}
        for dataset_id, query in chain_state.strong_queries.items():
            if dataset_id not in final:
                continue
            result_query = deepcopy(query)
            for dimension_query in final[dataset_id]:
                result_query.add_dimension_query(dimension_query)
            if date_time_query:
                result_query.add_dimension_query(date_time_query)
            result_queries[dataset_id] = result_query

        return result_queries

    def _get_retrieval_results(self, inputs: dict) -> RetrievalStagesResults:
        chain_state = ChainState.model_validate(inputs)

        if not chain_state.show_debug_stages:
            return RetrievalStagesResults()

        datasets_dict: dict[str, VersionedDataSet] = inputs["datasets_dict"]
        search_result: HybridSearchResult = inputs['search_result']

        return RetrievalStagesResults(
            indicators=dict(
                lexical=format_hybrid_scored_dicts(
                    dicts=search_result.lexical,
                ),
                semantic=format_hybrid_scored_dicts(
                    dicts=search_result.semantic,
                ),
                llm_scored=format_hybrid_llm_scored(
                    llm_scored=search_result.llm_scored, datasets_dict=datasets_dict
                ),
                final=format_hybrid_final_queries(
                    final_queries=search_result.final_queries,
                    datasets_dict=datasets_dict,
                ),
            ),
            stages_descriptions_ordered=self._RETRIEVAL_STAGES_ORDERED,
        )

    def _get_primary_queries_with_retrieval_results(self, inputs: dict) -> IndicatorsSearchResult:
        final_queries = self._get_final_queries(inputs)
        retrieval_results = self._get_retrieval_results(inputs)
        return IndicatorsSearchResult(queries=final_queries, retrieval_results=retrieval_results)

    def create_chain(self) -> Runnable:
        return (
            RunnablePassthrough.assign(search_result=self._get_search_result)
            | self._show_indicator_queries_stage
            | self._get_primary_queries_with_retrieval_results
        )
