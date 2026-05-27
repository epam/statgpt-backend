from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.settings.langchain import langchain_settings

from .base import EmbeddinglessVectorStore, VectorStore
from .embeddings import EmbeddingModels
from .pg_vector_store import PgEmbeddinglessVectorStore, PgVectorStore


class VectorStoreFactory:
    PG_VECTOR_STORE = "PgVectorStore"

    _embedding_models: EmbeddingModels = EmbeddingModels()
    _vector_stores: dict[str, type[VectorStore]] = {
        PG_VECTOR_STORE: PgVectorStore,
    }
    _embeddingless_vector_stores: dict[str, type[EmbeddinglessVectorStore]] = {
        PG_VECTOR_STORE: PgEmbeddinglessVectorStore,
    }

    def __init__(self, **kwargs):
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

    async def get_embeddingless_vector_store(
        self,
        collection_name: str,
        storage_name: str = PG_VECTOR_STORE,
        **kwargs,
    ) -> EmbeddinglessVectorStore:
        """Return a vector store that supports only embeddingless operations.

        Use this for delete, status, and dedup paths. It does not resolve an
        embedding model, so it works even when the channel's embedding deployment
        is unreachable.
        """
        for key, value in self._kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        logger.info(
            f'Initializing embeddingless pgvector storage with following options: {storage_name=}'
        )
        return self._embeddingless_vector_stores[storage_name](collection_name, **kwargs)

    def deepcopy(self):
        cls = self.__class__
        return cls(**self._kwargs)

    def update_kwargs(self, **kwargs):
        self._kwargs.update(kwargs)
