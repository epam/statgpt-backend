from common.config import logger
from common.data.base import DataSet
from common.indexer import IndexerFactory
from common.schemas import DataQueryDetails, IndicatorSelectionVersion
from common.utils.elastic import ElasticSearchFactory
from common.vectorstore import VectorStore

from .base import IndicatorSelectionBase, SemanticIndicatorSelectionBase
from .indicators_hybrid import IndicatorsSelectionHybrid
from .semantic import (
    IndicatorSelectionSemanticV1ChainFactory,
    IndicatorSelectionSemanticV2ChainFactory,
    IndicatorSelectionSemanticV3ChainFactory,
    IndicatorSelectionSemanticV4ChainFactory,
)


class IndicatorSelectionFactory:
    SEMANTIC_FACTORIES: dict[IndicatorSelectionVersion, SemanticIndicatorSelectionBase] = {
        IndicatorSelectionVersion.semantic_v1: IndicatorSelectionSemanticV1ChainFactory,
        IndicatorSelectionVersion.semantic_v2: IndicatorSelectionSemanticV2ChainFactory,
        IndicatorSelectionVersion.semantic_v3: IndicatorSelectionSemanticV3ChainFactory,
        IndicatorSelectionVersion.semantic_v4: IndicatorSelectionSemanticV4ChainFactory,
    }

    def __init__(
        self,
        config: DataQueryDetails,
        models_api_key: str,
        vector_store: VectorStore,
        matching_index_name: str,
        indicators_index_name: str,
        list_datasets: list[DataSet],
    ):
        self._config = config
        self._models_api_key = models_api_key
        self._vector_store = vector_store
        self._matching_index_name = matching_index_name
        self._indicators_index_name = indicators_index_name
        self._list_datasets = list_datasets

    def _create_semantic(
        self, version: IndicatorSelectionVersion
    ) -> SemanticIndicatorSelectionBase:
        return self.SEMANTIC_FACTORIES[version](self._config)

    async def _create_hybrid(self):
        matching_index = await ElasticSearchFactory.get_index(self._matching_index_name)
        indicators_index = await ElasticSearchFactory.get_index(self._indicators_index_name)
        searcher = IndexerFactory(
            self._models_api_key, matching_index, indicators_index, self._vector_store
        ).get_search()
        return IndicatorsSelectionHybrid(searcher)

    async def get_indicator_selection(
        self, indicator_selection_version: IndicatorSelectionVersion
    ) -> IndicatorSelectionBase:
        logger.info(
            f'Creating indicator selection chain for version: "{indicator_selection_version}"'
        )

        if indicator_selection_version in self.SEMANTIC_FACTORIES:
            return self._create_semantic(indicator_selection_version)
        elif indicator_selection_version == IndicatorSelectionVersion.hybrid:
            return await self._create_hybrid()

        raise ValueError(f"Invalid indicator selection version: '{indicator_selection_version}'")
