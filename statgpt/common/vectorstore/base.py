import uuid
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import NamedTuple

from langchain_core.documents import Document

from .document import ScoredVectorStoreDocument
from .embeddings import EmbeddingModel


class DedupCounts(NamedTuple):
    """Per-call result of a deduplicate-by-document-content run."""

    remapped: int
    deleted: int


class EmbeddinglessVectorStore(ABC):
    """Narrower vector store interface for operations that do not need an embedding model.

    Use this for delete, status, and dedup paths that should remain available
    even when the channel's embedding deployment is unreachable.
    """

    def __init__(self, collection_name: str, **kwargs) -> None:
        self._collection_name = collection_name

    @abstractmethod
    async def clear(self) -> None:
        pass

    @abstractmethod
    async def remove_documents_by(
        self, *, dataset_id: uuid.UUID | None = None, version_ids: list[int] | None = None
    ) -> None:
        """Removes all documents by dataset_id or version_ids."""

    @abstractmethod
    async def deduplicate_by_document_content(self) -> DedupCounts:
        """Removes and remaps duplicate documents based on document content.

        Returns the number of metadata rows remapped to keeper documents and the
        number of orphaned duplicate documents that were deleted.
        """

    @abstractmethod
    async def has_duplicates(self) -> tuple[bool, int]:
        """Checks if there are duplicate documents based on document content."""

    @abstractmethod
    async def has_duplicates_in_versions(self, version_ids: set[int]) -> tuple[bool, int]:
        """Checks if there are duplicate documents based on document content in the specified version IDs."""

    @abstractmethod
    async def get_total_size(self) -> int:
        """Returns the total number of documents in the vector store."""

    @abstractmethod
    async def get_size(self, version_ids: set[int]) -> int:
        """Returns the number of documents in the vector store for the specified version IDs."""

    @abstractmethod
    async def get_size_per_version(self, version_ids: set[int]) -> dict[int, int]:
        """Returns the number of documents per version_id."""


class VectorStore(EmbeddinglessVectorStore):
    """Full vector store interface.

    Extends `EmbeddinglessVectorStore` with operations that require an embedding model:
    ingest (`add_documents`), similarity search (`search_with_similarity_score`),
    and export/import that round-trip embeddings.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: EmbeddingModel,
        **kwargs,
    ) -> None:
        super().__init__(collection_name, **kwargs)
        self._embedding_model = embedding_model

    @abstractmethod
    async def add_documents(
        self, documents: Iterable[Document], dataset_id: uuid.UUID, version_id: int
    ) -> None:
        pass

    @abstractmethod
    async def search_with_similarity_score(
        self,
        query: str,
        *,
        k: int = 10,
        version_ids: set[int],
        metadata_filters: dict[str, set] | None = None,
    ) -> list[ScoredVectorStoreDocument]:
        """For a given query, get its nearest neighbors with similarity scores."""

    @abstractmethod
    async def export_to_folder(self, folder_path: str, version_ids: set[int]) -> None:
        """Exports the vector store data to the specified folder."""

    @abstractmethod
    async def import_from_zipfile(
        self,
        zip_file: zipfile.ZipFile,
        folder_prefix: str,
        dataset_versions: dict[uuid.UUID, int],
        data_sources: dict[uuid.UUID, int],
    ) -> None:
        """Clears the current vector store and imports data from the specified zip file folder."""
