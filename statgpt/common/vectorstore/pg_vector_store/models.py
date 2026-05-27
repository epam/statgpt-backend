import uuid
from typing import Any

from pgvector.sqlalchemy import Vector  # type: ignore
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from statgpt.common.utils import DateMixin, IdMixin


class Base(AsyncAttrs, DeclarativeBase):
    __table_args__ = {"schema": "collections"}


class NoEmbeddingsBase(AsyncAttrs, DeclarativeBase):
    """Separate SQLAlchemy metadata for document models that omit the vector column.

    Lives in its own MetaData so a no-embeddings concrete model can share a
    `__tablename__` with the full document model without conflicting.
    """

    __table_args__ = {"schema": "collections"}


class BaseModel(IdMixin, DateMixin, Base):
    __abstract__ = True


class BaseDocument(BaseModel):
    """Document model used to store `Indicators` and `Dimensions`"""

    __abstract__ = True  # this line is necessary

    document: Mapped[str]
    embeddings: Mapped[list[float]] = mapped_column(Vector(None))


class BaseDocumentNoEmbeddings(IdMixin, DateMixin, NoEmbeddingsBase):
    """Document model without the `embeddings` column.

    Used by code paths that only need `id` / `document` (delete, dedup, duplicate checks)
    so they do not have to know the embedding length and do not require an embedding model.
    """

    __abstract__ = True  # this line is necessary

    document: Mapped[str]


class BaseDocumentMetadata(BaseModel):
    __abstract__ = True  # this line is necessary

    document_id: Mapped[int]
    dataset_id: Mapped[uuid.UUID] = mapped_column(type_=postgresql.UUID(as_uuid=True))
    version_id: Mapped[int]
    details: Mapped[dict[str, Any]] = mapped_column(type_=postgresql.JSONB)


class ModelsStore:
    _models: dict[str, Any] = {}

    @classmethod
    def get_document_model(cls, name: str, embedding_length: int) -> type[BaseDocument]:
        if name not in cls._models:
            embeddings: Mapped[list[float]] = mapped_column(Vector(embedding_length))
            cls._models[name] = type(
                name, (BaseDocument,), {"__tablename__": name, "embeddings": embeddings}
            )
        return cls._models[name]

    @classmethod
    def get_document_no_embeddings_model(cls, name: str) -> type[BaseDocumentNoEmbeddings]:
        cache_key = f"{name}:no_embeddings"
        if cache_key not in cls._models:
            cls._models[cache_key] = type(
                name, (BaseDocumentNoEmbeddings,), {"__tablename__": name}
            )
        return cls._models[cache_key]

    @classmethod
    def get_document_metadata_model(cls, name: str) -> type[BaseDocumentMetadata]:
        if name not in cls._models:
            cls._models[name] = type(name, (BaseDocumentMetadata,), {"__tablename__": name})
        return cls._models[name]
