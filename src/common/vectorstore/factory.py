from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.settings.langchain import langchain_settings

from .base import VectorStore
from .embeddings import EmbeddingModels
from .pg_vector_store import PgVectorStore


class VectorStoreFactory:
    PG_VECTOR_STORE = "PgVectorStore"

    _embedding_models: EmbeddingModels = EmbeddingModels()
    _vector_stores: dict[str, type[VectorStore]] = {
        PG_VECTOR_STORE: PgVectorStore,
    }

    def __init__(self, **kwargs):
        # TODO: 'session' parameter seems to be required to be present in 'kwargs'!
        self._kwargs = kwargs

    async def get_vector_store(
        self,
        collection_name: str,
        auth_context: AuthContext,
        storage_name: str = PG_VECTOR_STORE,
        embedding_model_name: str = langchain_settings.embedding_default_model.value,
        **kwargs,
    ) -> VectorStore:
        for key, value in self._kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        embedding_model = await self._embedding_models.get(embedding_model_name, auth_context)
        logger.info(
            f'Initializing pgvector storage with following options: {storage_name=} {embedding_model=}'
        )
        return self._vector_stores[storage_name](collection_name, embedding_model, **kwargs)

    def deepcopy(self):
        cls = self.__class__
        return cls(**self._kwargs)

    def update_kwargs(self, **kwargs):
        self._kwargs.update(kwargs)
