from statgpt.app.chains.data_query.parameters import DataQueryParameters
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.schemas.query_builder import (
    ChainState,
    DataQueryEvalAttachment,
    DatasetAvailabilityQueriesType,
    QueryBuilderAgentState,
)
from statgpt.app.services.chat_facade import ScoredDimensionCandidate
from statgpt.common.data.base import (
    DataSetAvailabilityQuery,
    DimensionQuery,
    QueryOperator,
    VirtualDimensionValue,
)


def filter_empty_dataset_availability_queries(queries: DatasetAvailabilityQueriesType):
    res = {
        dataset_id: dataset_query
        for dataset_id, dataset_query in queries.items()
        if not dataset_query.is_empty()
    }
    return res


def dimension_candidates_to_queries(
    candidates: list[ScoredDimensionCandidate],
    date_time_query: DimensionQuery | None = None,
    dataset_2_dim_2_all_values_term: dict[str, dict[str, VirtualDimensionValue]] | None = None,
    dataset_ids_to_be_present: list[str] | None = None,
) -> DatasetAvailabilityQueriesType:
    candidates_grouped: dict[str, dict[str, set[ScoredDimensionCandidate]]] = {}
    for c in candidates:
        candidates_grouped.setdefault(c.dataset_id, {}).setdefault(c.dimension_id, set()).add(c)

    def _dataset_candidates_to_query(
        ds_candidates: dict[str, set[ScoredDimensionCandidate]], dataset_id: str
    ) -> DataSetAvailabilityQuery:
        query = DataSetAvailabilityQuery()

        for dim_id, dim_candidates in ds_candidates.items():
            candidates_ids = {c.query_id for c in dim_candidates}
            dim_query = None
            if dataset_2_dim_2_all_values_term:
                all_values_term = dataset_2_dim_2_all_values_term.get(dataset_id, {}).get(dim_id)
                if all_values_term and all_values_term.id in candidates_ids:
                    dim_query = DimensionQuery(
                        dimension_id=dim_id, values=[], operator=QueryOperator.ALL
                    )
            if dim_query is None:
                dim_query = DimensionQuery(
                    dimension_id=dim_id,
                    values=list(candidates_ids),
                    operator=QueryOperator.IN,
                )
            query.add_dimension_query(dim_query)

        if date_time_query:
            query.add_dimension_query(date_time_query)

        return query

    queries = {
        ds_id: _dataset_candidates_to_query(ds_candidates, dataset_id=ds_id)
        for ds_id, ds_candidates in candidates_grouped.items()
    }

    if dataset_ids_to_be_present:
        for ds_id in dataset_ids_to_be_present:
            if ds_id not in queries:
                queries[ds_id] = _dataset_candidates_to_query({}, dataset_id=ds_id)

    return queries


def set_tool_state(inputs: dict) -> dict:
    chain_state = ChainState(**inputs)

    indexed_datasets_id_map = {
        entity_id: ds.data.source_id
        for entity_id, ds in chain_state.versioned_datasets_dict.items()
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
        dimension_id_to_name=chain_state.dimension_id_to_name,
        special_dims_outputs=chain_state.special_dims_outputs,
        hybrid_search_timings=chain_state.hybrid_search_timings,
    )
    agent_state_dict = agent_state.model_dump(mode='json')

    eval_attachment = DataQueryEvalAttachment(
        retrieval_results=chain_state.retrieval_results,
    )
    eval_attachment_dict = eval_attachment.model_dump(mode='json')

    inputs[DataQueryParameters.STATE] = agent_state_dict
    inputs[DataQueryParameters.EVAL_ATTACHMENT] = eval_attachment_dict

    return inputs
