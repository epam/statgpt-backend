from copy import deepcopy

from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from common.config import multiline_logger as logger
from common.data.base import DataSetAvailabilityQuery, QueryOperator
from common.schemas import DataQueryDetails
from statgpt.chains.data_query.parameters import DataQueryParameters
from statgpt.chains.data_query.query_builder import utils as query_utils
from statgpt.chains.parameters import ChainParameters
from statgpt.schemas.query_builder import ChainState, MetaStateKeys
from statgpt.utils.callbacks import StageCallback

from .base import DimensionSearchChainFactoryBase
from .indicators import IndicatorsSearchChainFactory
from .non_indicators import NonIndicatorsSearchChainFactory
from .schemas import SearchInput
from .special import SpecialDimensionsSearchChainFactory


class DimensionSearchChainFactory(DimensionSearchChainFactoryBase):
    def __init__(self, config: DataQueryDetails):
        super().__init__(config)

        self._non_indicators_search_chain = NonIndicatorsSearchChainFactory(
            config=self._config
        ).create()
        self._indicators_search_chain = IndicatorsSearchChainFactory(config=self._config).create()
        self._special_dimensions_search_chain = SpecialDimensionsSearchChainFactory(
            config=self._config
        ).create()

    def _combine_indicators_and_special_dimensions_chain_outputs(self, inputs: dict) -> dict:
        state = inputs[MetaStateKeys.CHAIN_STATE]
        state['special_dims_outputs'] = inputs[MetaStateKeys.SPECIAL_DIMENSIONS_OUTPUTS]
        return state

    @staticmethod
    def _add_special_dims_to_strong_queries(inputs: dict) -> dict[str, DataSetAvailabilityQuery]:
        state = SearchInput(**inputs)
        strong_queries = state.strong_queries

        for _, sdim_out in state.special_dims_outputs.items():
            if sdim_out.no_queries():
                continue

            for ds_id, ds_strong in strong_queries.items():

                if (ds_sdim_query := sdim_out.dataset_queries.get(ds_id)) is None:
                    continue

                dim = ds_sdim_query.dimension_id
                if ds_sdim_query.operator != QueryOperator.IN:
                    raise ValueError(f'unexpected query operator: {ds_sdim_query.operator}')

                if (strong_dim_query := ds_strong.dimensions_queries_dict.get(dim)) is not None:
                    missing = set(ds_sdim_query.values).difference(strong_dim_query.values)
                    if missing:
                        strong_dim_query.values.extend(missing)
                else:
                    ds_strong.add_dimension_query(query=deepcopy(ds_sdim_query))

        return strong_queries

    def _create_nonindicators_ok_chain(self) -> Runnable:
        return (
            RunnableParallel(
                {
                    MetaStateKeys.CHAIN_STATE: self._indicators_search_chain,
                    MetaStateKeys.SPECIAL_DIMENSIONS_OUTPUTS: self._special_dimensions_search_chain,
                }
            )
            | self._combine_indicators_and_special_dimensions_chain_outputs
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
            inputs['skip_finalization'] = True
            target = ChainParameters.get_target(inputs)
            target.append_content(message)
            return RunnableLambda(query_utils.set_tool_state)
        else:
            return self._create_nonindicators_ok_chain()

    def create(self) -> Runnable:
        return self._non_indicators_search_chain | self._route_based_on_nonindicators_status
