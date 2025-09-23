from abc import ABC, abstractmethod
from collections.abc import Iterable

from langchain_core.documents import Document

from .document import EmbeddedDocument
from .embeddings import EmbeddingModel


class VectorStore(ABC):
    def __init__(
        self,
        collection_name: str,
        embedding_model: EmbeddingModel,
        datasource: str | None = None,
        **kwargs,
    ) -> None:
        self._collection_name = collection_name
        self._datasource = datasource
        self._embedding_model = embedding_model

    @abstractmethod
    async def clear(self) -> None:
        pass

    @abstractmethod
    async def import_documents(
        self, documents: Iterable[EmbeddedDocument], dataset_id: int | None = None
    ) -> None:
        pass

    @abstractmethod
    async def add_documents(
        self, documents: Iterable[Document], dataset_id: int | None = None
    ) -> None:
        pass

    @abstractmethod
    async def remove_documents_by_dataset_id(self, dataset_id: int) -> None:
        """Remove all documents associated with the given dataset id"""

    @abstractmethod
    async def search(self, query: str, k: int = 10) -> list[Document]:
        """For a given query, get its nearest neighbors"""

    @abstractmethod
    async def search_with_similarity_score(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        """For a given query, get its nearest neighbors with similarity scores."""

    @abstractmethod
    async def search_with_similarity_score_and_dataset_id(
        self,
        query: str,
        k: int = 10,
        dataset_ids: set[int] | None = None,
        metadata_filters: dict[str, set] | None = None,
    ) -> list[tuple[Document, float, int]]:
        """
        For a given query, get its nearest neighbors with similarity scores along with the dataset ids it belongs to.
        Optionally filter by dataset ids and metadata fields.
        """

    @abstractmethod
    async def get_documents(
        self,
        limit: int | None = None,
        offset: int | None = None,
        ids: Iterable[int] | None = None,
        dataset_id: int | None = None,
        include_embeddings: bool = False,
    ) -> list[EmbeddedDocument]:
        pass

    @abstractmethod
    async def get_dataset_ids_by_documents_ids(self, ids: Iterable[int]) -> list[int]:
        pass

    @abstractmethod
    async def get_document_ids_by_dataset_id(self, dataset_id: int) -> list[int]:
        pass
