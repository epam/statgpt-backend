from common.utils.elastic import ElasticIndex
from common.vectorstore import VectorStore

from .cache_factory import CacheFactory
from .indexer import Indexer
from .searcher import Search


class IndexerFactory:
    def __init__(
        self,
        models_api_key: str,
        matching_index: ElasticIndex,
        indicators_index: ElasticIndex,
        vectorstore: VectorStore,
    ):
        self._models_api_key = models_api_key
        self._matching_index = matching_index
        self._indicators_index = indicators_index
        self._vectorstore = vectorstore
        self._cache = CacheFactory.get_instance()

    def get_indexer(self, normalize: bool, harmonize: bool) -> Indexer:
        return Indexer(
            self._models_api_key,
            self._matching_index,
            self._indicators_index,
            self._vectorstore,
            normalize=normalize,
            harmonize=harmonize,
        )

    def get_search(self) -> Search:
        return Search(
            self._models_api_key,
            self._matching_index,
            self._indicators_index,
            self._vectorstore,
            self._cache,
        )
