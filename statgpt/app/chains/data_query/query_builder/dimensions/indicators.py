from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough

from statgpt.app.chains.data_query.query_builder.indicator_selection.factory import (
    IndicatorSelectionFactory,
)
from statgpt.app.utils.callbacks import StageCallback
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import DataSetAvailabilityQuery

from .base import DimensionSearchChainFactoryBase
from .schemas import SearchInput


class IndicatorsSearchChainFactory(DimensionSearchChainFactoryBase):
    def _filter_queries_by_required_dimensions(
        self, inputs: dict
    ) -> dict[str, DataSetAvailabilityQuery]:
        search_input = SearchInput(**inputs)

        if not search_input.strong_queries:
            return {}

        filtered_queries = {}
        for dataset_id, dataset_query in search_input.strong_queries.items():
            dataset = search_input.datasets_dict[dataset_id].data
            required_dims = dataset.required_dimensions
            dim_queries = dataset_query.dimensions_queries_dict

            if required_dims and all(
                dim_id not in dim_queries or dim_queries[dim_id].is_empty()
                for dim_id in required_dims
            ):
                logger.info(
                    f'filter by required dims: removing "{dataset_id}" dataset query, since it does not contain '
                    'query to at least 1 required dimensions '
                    f'({required_dims}). dataset query: {dataset_query}'
                )
            else:
                logger.debug(
                    f'filter by required dims: no required dims for "{dataset_id}" dataset '
                    'or at least one required dim has non-empty query'
                )
                filtered_queries[dataset_id] = dataset_query

        return filtered_queries

    async def _run_indicators_selection(self, inputs: dict) -> Runnable:
        search_input = SearchInput(**inputs)

        meta_factory = IndicatorSelectionFactory(
            config=self._config,
            models_api_key=search_input.auth_context.api_key,
            vector_store=await search_input.data_service._get_indicators_vector_store(
                search_input.auth_context
            ),
            matching_index_name=search_input.data_service.channel.matching_index_name,
            indicators_index_name=search_input.data_service.channel.indicators_index_name,
        )

        chain_factory = await meta_factory.get_indicator_selection(
            indicator_selection_version=self._config.indicator_selection_version
        )
        chain = chain_factory.create_chain()
        return chain

    def create(self) -> Runnable:
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
                strong_queries=self._filter_queries_by_required_dimensions
            ).with_config(
                config=RunnableConfig(
                    callbacks=[
                        StageCallback(
                            "Strong Queries, filtered by required dimensions",
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
