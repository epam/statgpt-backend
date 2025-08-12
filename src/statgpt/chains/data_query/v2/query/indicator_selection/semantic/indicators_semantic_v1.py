import asyncio
from copy import deepcopy

from langchain_core.runnables import Runnable, RunnablePassthrough

from common.data.base import DataSetAvailabilityQuery, DimensionQuery, QueryOperator
from statgpt.chains.data_query.v2.query.indicator_selection.base import (
    SemanticIndicatorSelectionBase,
)
from statgpt.default_prompts.v2 import DefaultPrompts
from statgpt.schemas.query_builder import (
    ChainState,
    IndicatorsSearchResult,
    RetrievalStageDescription,
    RetrievalStagesResults,
)
from statgpt.services import ScoredIndicatorCandidate


class IndicatorSelectionSemanticV1ChainFactory(SemanticIndicatorSelectionBase):
    """
    Map indicators from the provided list of candidates to binary relevancy (0/1).
    For complex indicators, dimension ids and names are combined into single strings.
    """

    # TODO: should be removed, not relevant with hybrid search
    _STAGES_DESCRIPTIONS = [
        RetrievalStageDescription(
            field_name='vector_search',
            short_name='vs',
            description='Indicators from Vector Search outputs over selected datasets',
        ),
        RetrievalStageDescription(
            field_name='availability',
            short_name='avail',
            description=(
                'Indicators from Vector Search outputs, filtered by selected '
                'non-indicator dimensions values availability'
            ),
        ),
        RetrievalStageDescription(
            field_name='llm_selects',
            short_name='llm',
            description='Indicators selected by LLM',
        ),
    ]

    def _get_system_prompt(self) -> str:
        return (
            self._config.prompts.indicators_selection_system_prompt
            or DefaultPrompts.INDICATORS_SELECTION_SYSTEM_PROMPT
        )

    def _get_validation_user_prompt(self) -> str:
        return self._config.prompts.validation_user_prompt or DefaultPrompts.VALIDATION_USER_PROMPT

    def get_indicator_selection_chain_factory(self):
        from statgpt.chains import (  # TODO: resolve circular import
            CandidatesSelectionBatchedChainFactory,
            CandidatesSelectionMappingChainFactory,
        )

        indicators_batch_chain_factory = CandidatesSelectionMappingChainFactory(
            system_prompt=self._get_system_prompt(),
            user_prompt=self._get_validation_user_prompt(),
            candidates_key="indicator_candidates",
        )
        return CandidatesSelectionBatchedChainFactory(
            inner_chain_factory=indicators_batch_chain_factory,
            candidates_key="indicator_candidates",
            batch_size=20,  # TODO: move to config
        )

    async def _get_indicator_candidates_from_normalized_query(
        self, inputs: dict
    ) -> list[ScoredIndicatorCandidate]:
        chain_state = ChainState(**inputs)
        datasets_dict = chain_state.datasets_dict
        normalized_query = chain_state.normalized_query
        from statgpt.chains.parameters import ChainParameters

        data_service = ChainParameters.get_data_service(inputs)

        candidates = await data_service.search_indicators_scored(
            normalized_query,
            auth_context=chain_state.auth_context,
            k=100,  # TODO: make configurable
            datasets=set(datasets_dict.keys()),
        )
        return candidates

    @staticmethod
    def _filter_indicator_candidates_by_queries(
        queries: dict[str, DataSetAvailabilityQuery],
        indicator_candidates: list[ScoredIndicatorCandidate],
    ):
        result = set()

        for candidate in indicator_candidates:
            dataset_query = queries.get(candidate.dataset_id)
            if not dataset_query:  # or not dataset_query.indicator_query:
                continue
            # if candidate.query_id in dataset_query.indicator_query.values:
            #     result.append(candidate)

            for indicator in candidate.indicator.indicators:
                dimension_id = indicator.dimension_id

                if (
                    dimension_id is None
                    or dimension_id not in dataset_query.dimensions_queries_dict
                ):
                    continue

                if indicator.query_id in dataset_query.dimensions_queries_dict[dimension_id].values:
                    result.add(candidate)
        result = list(result)
        return result

    @classmethod
    def _add_indicators_to_strong_availability_queries(
        cls, inputs: dict
    ) -> dict[str, DataSetAvailabilityQuery]:
        chain_state = ChainState(**inputs)
        old_strong_queries = chain_state.strong_queries
        indicator_candidates = chain_state.indicator_candidates

        for dataset_id, dataset_query in old_strong_queries.items():
            dataset = chain_state.datasets_dict[dataset_id]

            for dimension in dataset.indicator_dimensions():
                if query := dataset_query.dimensions_queries_dict.get(dimension.entity_id):
                    query.values = []  # TODO: ?

        result_strong_queries = {}
        for candidate in indicator_candidates:
            if candidate.dataset_id not in old_strong_queries:
                continue  # TODO: can cause retrieval to fail
            if candidate.dataset_id not in result_strong_queries:
                result_strong_queries[candidate.dataset_id] = old_strong_queries[
                    candidate.dataset_id
                ]
            dataset_query = result_strong_queries[candidate.dataset_id]

            for indicator in candidate.indicator.indicators:
                if query := dataset_query.dimensions_queries_dict.get(indicator.dimension_id):
                    if indicator.query_id not in query.values:
                        query.values.append(indicator.query_id)
                else:
                    dataset_query.add_dimension_query(
                        DimensionQuery(
                            values=[indicator.query_id],
                            operator=QueryOperator.IN,
                            dimension_id=indicator.dimension_id,
                        )
                    )
        return result_strong_queries

    @staticmethod
    def _filter_irrelevant_indicators_by_llm(inputs: dict) -> list[ScoredIndicatorCandidate]:
        chain_state = ChainState(**inputs)
        selected_ids = chain_state.indicators_llm_selection_output.get_selected_ids()
        filtered = [c for c in chain_state.indicator_candidates if c.query_id in selected_ids]
        return filtered

    @staticmethod
    def _limit_indicator_candidates_number(inputs: dict) -> list[ScoredIndicatorCandidate]:
        chain_state = ChainState(**inputs)
        indicator_candidates = chain_state.indicator_candidates
        # TODO: HACK!!! currently we limit number of indicators to present to user.
        # It's done AFTER LLM selection, meaning that we have already lost the concept
        # of indicators relevance order (LLM does not provide relevancy scores)!
        # It means, we CAN'T simly take top N indicators,
        # since they are not sorted by relevance anymore.
        # Once we have clarification questions implemented, we MUST
        # keep ALL selected indicators and remove this hack!!!!!!
        n = min(30, len(indicator_candidates))  # TODO: make number of items to keep configurable
        filtered = indicator_candidates[:n]
        return filtered

    @staticmethod
    async def _availability(inputs: dict, queries_key: str) -> dict[str, DataSetAvailabilityQuery]:
        chain_state = ChainState(**inputs)
        from statgpt.chains.parameters import ChainParameters  # TODO: resolve circular import

        auth_context = ChainParameters.get_auth_context(inputs)
        queries: dict[str, DataSetAvailabilityQuery] = inputs[queries_key]
        tasks = []
        for dataset_id, query in queries.items():
            dataset = chain_state.datasets_dict[dataset_id]
            tasks.append(dataset.availability_query(query, auth_context))
        task_results = await asyncio.gather(*tasks)
        result = {dataset_id: query for dataset_id, query in zip(queries.keys(), task_results)}
        return result

    async def _availability_by_strong_queries(
        self, inputs: dict
    ) -> dict[str, DataSetAvailabilityQuery]:
        return await self._availability(inputs, "strong_queries")

    @staticmethod
    async def _pack_results(inputs: dict):
        strong_queries: dict[str, DataSetAvailabilityQuery] = inputs["strong_queries"]
        res = IndicatorsSearchResult(
            queries=strong_queries,
            retrieval_results=RetrievalStagesResults(),  # leavy empty
        )
        return res

    def create_chain(self) -> Runnable:
        # NOTE !!!!! the chain below contains several bugs
        # that were fixed in the SemanticV2 chain, for example:
        # * unexpected in-place modfication of strong_queries
        # * incorrect/non-optimal logic to process Complex Indicators
        # * probably something else

        # TODO: use separate model for LLM prompt to use simple ids instead of complex ones

        indicator_selection_chain_factory = self.get_indicator_selection_chain_factory()

        return (
            RunnablePassthrough.assign(
                indicator_candidates=self._get_indicator_candidates_from_normalized_query
            )
            # save vector search outputs before applying further filtering
            | RunnablePassthrough.assign(
                indicator_candidates_vector_search_outputs=lambda inputs: deepcopy(
                    inputs["indicator_candidates"]
                )
            )
            | RunnablePassthrough.assign(
                indicator_candidates=lambda d: self._filter_indicator_candidates_by_queries(
                    queries=d["strong_availability"],
                    indicator_candidates=d["indicator_candidates"],
                )
            )
            | RunnablePassthrough.assign(
                weak_queries=self._add_indicators_to_strong_availability_queries
            )
            | RunnablePassthrough.assign(
                # NOTE: previously,
                # when adding indicators to weak queries,
                # we used only indicator candidates from datasets present in strong queries.
                # Now, we need to filter indicator candidates accordingly, to match weak queries.
                # TODO: this can be combined in single function.
                indicator_candidates=lambda d: self._filter_indicator_candidates_by_queries(
                    queries=d["weak_queries"],
                    indicator_candidates=d["indicator_candidates"],
                )
            )
            # # TODO: there is no sense in this function, I guess
            # | RPA(weak_queries=self._add_indicators_to_strong_availability_queries)
            | RunnablePassthrough.assign(
                indicator_candidates_filtered_by_availabiility=lambda inputs: deepcopy(
                    inputs["indicator_candidates"]
                )
            )
            | RunnablePassthrough.assign(
                indicators_llm_selection_output=indicator_selection_chain_factory.create_chain()
            )
            | RunnablePassthrough.assign(
                indicator_candidates=self._filter_irrelevant_indicators_by_llm
            )
            | RunnablePassthrough.assign(
                indicator_candidates_llm_selects=lambda inputs: deepcopy(
                    inputs["indicator_candidates"]
                )
            )
            | RunnablePassthrough.assign(
                indicator_candidates=self._limit_indicator_candidates_number
            )
            | RunnablePassthrough.assign(
                strong_queries=self._add_indicators_to_strong_availability_queries
            )
            # TODO: return filter by availability?
            # | RunnablePassthrough.assign(strong_availability=self._availability_by_strong_queries)
            | self._pack_results
        )

    def get_retrieval_results(self, inputs: dict) -> RetrievalStagesResults:
        chain_state = ChainState(**inputs)

        # NOTE: here we call model_dump() explicitly to perform proper fields exclusion.
        # otherwise, ScoredIndicatorCandidate will be casted to dict
        # without excluduing specified fields.
        # NOTE: when we switch to pydantic V2, we can
        # change RetrievalStagesResults.indicators members signature
        # from list[dict] to list[ScoredIndicatorCandidate]
        # and remove this explicit model_dump(),
        # since the final dump will be performed automatically
        # (automatic dump is not possible, since we mix pydantic V1 and V2 models here)
        def _dump_indicator_candidates(candidates: list[ScoredIndicatorCandidate]) -> list[dict]:
            return [x.model_dump() for x in candidates]

        retrieval_results = RetrievalStagesResults(
            indicators=dict(
                vector_search=_dump_indicator_candidates(
                    chain_state.indicator_candidates_vector_search_outputs
                ),
                availability=_dump_indicator_candidates(
                    chain_state.indicator_candidates_filtered_by_availabiility
                ),
                llm_selects=_dump_indicator_candidates(
                    chain_state.indicator_candidates_llm_selects
                ),
            ),
            stages_description_ordered=self._STAGES_DESCRIPTIONS,
        )
        return retrieval_results
