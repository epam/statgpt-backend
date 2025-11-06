from langchain_core.runnables import Runnable

from common.schemas import DataQueryDetails

from .dimensions.factory import DimensionSearchChainFactory
from .misc.search_preparation import SearchPreparationChainFactory
from .query.finalize_query import FinalizeQueryChainFactory


class QueryBuilderFactory:
    def __init__(self, config: DataQueryDetails):
        self._config = config

        self._search_preparation_chain = SearchPreparationChainFactory(config=self._config)
        self._dimensions_search_chain = DimensionSearchChainFactory(config=self._config)
        self._finalize_query_chain = FinalizeQueryChainFactory(config=self._config)

    async def create_chain(self, inputs: dict | None) -> Runnable:
        if inputs is None:
            raise ValueError("Request context inputs are required")

        auth_context = inputs.get('auth_context')
        if not auth_context:
            raise ValueError("auth_context is required in inputs")

        return (
            self._search_preparation_chain.create()
            | self._dimensions_search_chain.create()
            | self._finalize_query_chain.create()
        )
