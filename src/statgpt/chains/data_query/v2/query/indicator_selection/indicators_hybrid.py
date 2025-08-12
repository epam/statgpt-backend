from copy import deepcopy

from langchain_core.runnables import Runnable, RunnablePassthrough

from common.data.base import DataSet, DataSetAvailabilityQuery, DimensionQuery
from common.indexer.searcher import Search
from statgpt.chains.data_query.v2.query.utils import get_indicators_for_retrieval_results
from statgpt.chains.parameters import ChainParameters
from statgpt.schemas.query_builder import (
    DateTimeQueryResponse,
    IndicatorsSearchResult,
    RetrievalStageDescription,
    RetrievalStagesResults,
)

from .base import IndicatorSelectionBase


class IndicatorsSelectionHybrid(IndicatorSelectionBase):
    _STAGES_DESCRIPTIONS = [
        RetrievalStageDescription(
            field_name='hybrid_best_of',
            short_name='hybrid_best_of',
            description='Indicators returned by hybrid search',
        ),
    ]

    _search: Search

    def __init__(self, search: Search):
        self._search = search

    async def _get_search_result(self, inputs: dict):
        choice = ChainParameters.get_choice(inputs)
        normalized_query: str = inputs['normalized_query']
        datasets = inputs['datasets_dict']
        entities_response = inputs['named_entities_response']
        period_response = inputs['date_time_query_response']
        availability = inputs['strong_availability']
        entities = entities_response.entities if entities_response else None

        with choice.create_stage("Hybrid Indicators Selection") as stage:
            return await self._search.search(
                stage, normalized_query, datasets, entities, period_response, availability
            )

    def _get_hybrid_best_of(self, inputs: dict) -> dict[str, list[DimensionQuery]]:
        best_of: dict[str, list[DimensionQuery]] = inputs["search_result"]
        if not best_of:
            return {}
        return best_of

    def _get_primary_queries_with_retrieval_results(self, inputs: dict) -> IndicatorsSearchResult:
        datasets_dict: dict[str, DataSet] = inputs["datasets_dict"]
        # TODO: should be str, not int
        best_of: dict[str, list[DimensionQuery]] = inputs["hybrid_best_of"]
        strong_queries: dict[str, DataSetAvailabilityQuery] = inputs["strong_queries"]
        date_time_query_response: DateTimeQueryResponse = inputs["date_time_query_response"]
        date_time_query = date_time_query_response.to_query()
        result_queries = {}
        for dataset_id, query in strong_queries.items():
            if dataset_id not in best_of:
                continue
            result_query = deepcopy(query)
            for dimension_query in best_of[dataset_id]:
                result_query.add_dimension_query(dimension_query)
            if date_time_query:
                result_query.add_dimension_query(date_time_query)
            result_queries[dataset_id] = result_query
        return IndicatorsSearchResult(
            queries=result_queries,
            retrieval_results=RetrievalStagesResults(
                indicators=get_indicators_for_retrieval_results(
                    best_of=best_of,
                    datasets_dict=datasets_dict,
                    stage_name=IndicatorsSelectionHybrid._STAGES_DESCRIPTIONS[0].field_name,
                ),
                stages_descriptions_ordered=IndicatorsSelectionHybrid._STAGES_DESCRIPTIONS,
            ),
        )

    def create_chain(self) -> Runnable:
        return (
            RunnablePassthrough.assign(search_result=self._get_search_result)
            | RunnablePassthrough.assign(hybrid_best_of=self._get_hybrid_best_of)
            | self._get_primary_queries_with_retrieval_results
        )
