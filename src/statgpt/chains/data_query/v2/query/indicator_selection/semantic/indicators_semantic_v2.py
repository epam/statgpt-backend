from copy import deepcopy

from langchain_core.runnables import Runnable, RunnablePassthrough

from common.config import logger
from common.data.base import DataSetAvailabilityQuery, DimensionQuery, QueryOperator
from common.data.sdmx.common.indicator import ComplexIndicatorComponentDetails
from common.schemas import DataQueryDetails
from statgpt.chains.data_query.v2.query import utils as query_utils
from statgpt.chains.data_query.v2.query.indicator_selection.base import (
    SemanticIndicatorSelectionBase,
)
from statgpt.chains.parameters import ChainParameters
from statgpt.config import StateVarsConfig
from statgpt.schemas.query_builder import (
    ChainState,
    DatasetAvailabilityQueriesType,
    DatasetDimQueries,
    IndicatorsSearchResult,
    RetrievalStagesResults,
)
from statgpt.services import ScoredIndicatorCandidate
from statgpt.utils.dial_stages import optional_timed_stage

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
        return PackedIndicatorsSelectionV1ChainFactory(
            candidates_key=self._candidates_key,
            llm_model_config=self._config.llm_models.indicators_selection_model_config,
        )

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

    @staticmethod
    async def _convert_llm_indicator_queries(inputs: dict) -> DatasetAvailabilityQueriesType:
        llm_response: DatasetDimQueries = inputs['llm_indicator_queries']
        return {
            dataset_id: DataSetAvailabilityQuery.from_dimension_queries_list(
                [
                    DimensionQuery(
                        values=dim_values, operator=QueryOperator.IN, dimension_id=dimension_id
                    )
                    for dimension_id, dim_values in dimension_queries.items()
                ]
            )
            for dataset_id, dimension_queries in llm_response.queries.items()
        }

    @staticmethod
    async def _show_indicator_queries_stage(inputs: dict) -> dict:
        indicator_queries: DatasetAvailabilityQueriesType = inputs['llm_indicator_queries']
        state = ChainParameters.get_state(inputs)
        show_debug_stages = state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False)
        with optional_timed_stage(
            choice=ChainParameters.get_choice(inputs),
            name="[DEBUG] Indicator Queries",
            enabled=show_debug_stages,
        ) as stage:
            if stage:
                await query_utils.populate_queries_stage(
                    stage=stage,
                    queries=indicator_queries,
                    auth_context=ChainParameters.get_auth_context(inputs),
                    datasets_dict=ChainParameters.get_datasets_dict(inputs),
                )
        return inputs

    @classmethod
    async def _add_llm_indicator_queries_to_strong_queries(
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
        indicator_queries: DatasetAvailabilityQueriesType = inputs['llm_indicator_queries']
        combined_queries = deepcopy(inputs["strong_queries"])

        for dataset_id, dim_queries in indicator_queries.items():
            if dataset_id not in combined_queries:
                # the dataset was removed in nonindicator chain for a reason.
                # don't add indicator query for this dataset.
                continue

            for dim_query in dim_queries.dimensions_queries:
                combined_queries[dataset_id].add_dimension_query(dim_query)

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
            | RunnablePassthrough.assign(llm_indicator_queries=self._convert_llm_indicator_queries)
            | self._show_indicator_queries_stage
            | RunnablePassthrough.assign(
                strong_queries=self._add_llm_indicator_queries_to_strong_queries
            )
            | self._pack_results
            # NOTE: built queries will be filtered by availability after this chain
        )

        return chain
