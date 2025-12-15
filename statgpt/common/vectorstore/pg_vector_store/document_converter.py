from statgpt.common.vectorstore.document import (
    EmbeddedDocument,
    ScoredVectorStoreDocument,
    VectorStoreDocument,
)

from .models import BaseDocument as PgDocument
from .models import BaseDocumentMetadata as PgDocMetadata


class DocumentConverter:
    """Converts VectorStore items to LangChain Documents"""

    @staticmethod
    def to_embedded_document(
        pg_document: PgDocument, pg_metadata: PgDocMetadata
    ) -> EmbeddedDocument:
        return EmbeddedDocument(
            # ~~~ LangChain fields ~~~
            page_content=pg_document.document,
            metadata=pg_metadata.details,
            # ~~~ Custom fields ~~~
            embeddings=pg_document.embeddings,
        )

    @staticmethod
    def to_vector_store_document(
        pg_document: PgDocument, pg_metadata: PgDocMetadata
    ) -> VectorStoreDocument:
        return VectorStoreDocument(
            # ~~~ LangChain fields ~~~
            page_content=pg_document.document,
            metadata=pg_metadata.details,
            # ~~~ Custom fields ~~~
            table_name=pg_document.__tablename__,
            document_id=pg_document.id,
            dataset_id=pg_metadata.dataset_id,
            version_id=pg_metadata.version_id,
        )

    @staticmethod
    def to_scored_vector_store_document(
        pg_document: PgDocument,
        pg_metadata: PgDocMetadata,
        score: float,
    ) -> ScoredVectorStoreDocument:
        return ScoredVectorStoreDocument(
            # ~~~ LangChain fields ~~~
            page_content=pg_document.document,
            metadata=pg_metadata.details,
            # ~~~ Custom fields ~~~
            table_name=pg_document.__tablename__,
            document_id=pg_document.id,
            dataset_id=pg_metadata.dataset_id,
            version_id=pg_metadata.version_id,
            score=score,
        )
