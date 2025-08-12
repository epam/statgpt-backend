from copy import deepcopy

from langchain_core.runnables import Runnable, RunnablePassthrough

from common.config import logger
from common.data.base import DimensionQuery, QueryOperator
from common.data.sdmx.common.indicator import ComplexIndicatorComponentDetails
from common.schemas import DataQueryDetails
from statgpt.chains.data_query.v2.query.indicator_selection.base import (
    SemanticIndicatorSelectionBase,
)
from statgpt.schemas.query_builder import (
    ChainState,
    DatasetAvailabilityQueriesType,
    DatasetDimQueries,
    IndicatorsSearchResult,
    RetrievalStagesResults,
)
from statgpt.services import ScoredIndicatorCandidate

from .packed_indicators_selection import PackedIndicatorsSelectionV1ChainFactory


class IndicatorSelectionSemanticV2ChainFactory(SemanticIndicatorSelectionBase):
    """
    Specialized version of indicators selection for complex indicators.

    """

    def __init__(
        self,
        config: DataQueryDetails,
        vector_search_top_k: int = 100,
    ):
        super().__init__(config)
        self._candidates_key = 'indicator_candidates'
        self._vector_search_top_k = vector_search_top_k  # TODO: move to config

    def get_indicator_selection_chain_factory(self):
        return PackedIndicatorsSelectionV1ChainFactory(candidates_key=self._candidates_key)

    async def _get_indicator_candidates_from_normalized_query(
        self, inputs: dict
    ) -> list[ScoredIndicatorCandidate]:
        chain_state = ChainState(**inputs)
        datasets_dict = chain_state.datasets_dict
        normalized_query = chain_state.normalized_query
        from statgpt.chains.parameters import ChainParameters  # TODO: resolve circular import

        data_service = ChainParameters.get_data_service(inputs)

        candidates = await data_service.search_indicators_scored(
            normalized_query,
            auth_context=chain_state.auth_context,
            k=self._vector_search_top_k,
            datasets=set(datasets_dict.keys()),
        )
        return candidates

    @classmethod
    def _filter_indicator_candidates_by_availability(
        cls,
        queries: DatasetAvailabilityQueriesType,
        indicator_candidates: list[ScoredIndicatorCandidate],
    ) -> list[ScoredIndicatorCandidate]:
        """
        Keep only those indicator candidates that are present in result of availability query
        """

        result = set()

        available_values = {
            dataset_id: {
                dim_id: set(dim_query.values)
                for dim_id, dim_query in dataset_query.dimensions_queries_dict.items()
            }
            for dataset_id, dataset_query in queries.items()
        }

        for ind_candidate in indicator_candidates:
            dataset_id = ind_candidate.dataset_id
            if dataset_id not in available_values:
                # TODO: possible bug:
                # queries with available values could have some dataset missed (? need to check ?)
                continue
            avail_dataset_values = available_values[dataset_id]

            # iterate over complex indicator components
            # and check if each of them is present in available values
            add_indicator = True
            components_details = ind_candidate.indicator.get_components_details()
            for cur_comp_details in components_details:
                cur_comp_details: ComplexIndicatorComponentDetails
                dim_id = cur_comp_details.dimension_id
                avail_dim_values = avail_dataset_values.get(dim_id, set())
                if cur_comp_details.query_id not in avail_dim_values:
                    add_indicator = False
                    break
            # add indicator if check passed
            if add_indicator:
                result.add(ind_candidate)

        result = list(result)

        if not result:
            logger.warning(
                f'{cls.__name__}._filter_indicator_candidates_by_availability():'
                f'there are no candidates left after filtering by availability: {result=}'
            )

        return result

    @classmethod
    def _add_llm_indicator_queries_to_strong_queries(
        cls, inputs: dict
    ) -> DatasetAvailabilityQueriesType:
        """
        Combine indicator queries built by LLM with strong queries.

        NOTE: queries selected by LLM are filtered by dataset.

        NOTE:
        - LLM could select non-existing (non-available) combinations of indicator dim values.
        - such combinations must be filtered out later by availability queries.
        - no filtration by availability is done here.
        """

        chain_state = ChainState(**inputs)
        llm_response: DatasetDimQueries = inputs['llm_indicator_queries']

        combined_queries = deepcopy(chain_state.strong_queries)

        for dataset_id, dim_queries in llm_response.queries.items():
            if dataset_id not in combined_queries:
                # NOTE: here we filter indicator candidates!
                #
                # NOTE: we could ensure "strong_queries" and "strong_availability"
                # contain same list of datasets (probably it's true already).
                # In this case we wouldn't need this filter here.
                #
                # there are 2 scenarios:
                # 1. dataset was filtered out,
                #    for example by user query on some other dimension like "country".
                #    in this case, we should skip this dataset indeed.
                # 2. there were no filters in strong queries at all.
                #    in this case we should add this indicator to create a strong query.
                #    BUG: it looks like a bug in this case
                #    TODO: fix bug for case 2.
                continue

            for dimension_id, dim_values in dim_queries.items():
                combined_queries[dataset_id].add_dimension_query(
                    DimensionQuery(
                        values=dim_values, operator=QueryOperator.IN, dimension_id=dimension_id
                    )
                )

        return combined_queries

    @staticmethod
    async def _pack_results(inputs: dict):
        strong_queries: DatasetAvailabilityQueriesType = inputs["strong_queries"]
        res = IndicatorsSearchResult(
            queries=strong_queries,
            retrieval_results=RetrievalStagesResults(),  # TODO: for now we leave it empty
        )
        return res

    def create_chain(self) -> Runnable:
        """
        NOTE: "strong queries" are queries validated by LLM.
        """

        indicator_selection_chain = self.get_indicator_selection_chain_factory().create_chain()

        chain = (
            # extract available indicator combinations relevant to user query.
            # NOTE: all extracted combinations are available (they have data).
            RunnablePassthrough.assign(
                **{self._candidates_key: self._get_indicator_candidates_from_normalized_query}
            )
            # filter extracted candidates by strong_availability
            | RunnablePassthrough.assign(
                **{
                    self._candidates_key: lambda d: self._filter_indicator_candidates_by_availability(
                        queries=d["strong_availability"],
                        indicator_candidates=d[self._candidates_key],
                    )
                }
            )
            | RunnablePassthrough.assign(llm_indicator_queries=indicator_selection_chain)
            | RunnablePassthrough.assign(
                strong_queries=self._add_llm_indicator_queries_to_strong_queries
            )
            | self._pack_results
            # NOTE: built queries will be filtered by availability after this chain
        )

        return chain
