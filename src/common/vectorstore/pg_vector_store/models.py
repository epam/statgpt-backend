import uuid as uuid_module
from typing import Any

from pgvector.sqlalchemy import Vector  # type: ignore
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from common.config import VectorStoreMetadataFields
from common.utils import DateMixin, IdMixin
from common.vectorstore.document import EmbeddedDocument


class Base(AsyncAttrs, DeclarativeBase):
    __table_args__ = {"schema": "collections"}


class BaseModel(IdMixin, DateMixin, Base):
    __abstract__ = True


class BaseDocument(BaseModel):
    """Document model used to store `Indicators` and `Dimensions`"""

    __abstract__ = True  # this line is necessary

    document: Mapped[str]
    details: Mapped[dict[str, Any]] = mapped_column(type_=postgresql.JSONB)
    embeddings: Mapped[list[float]] = mapped_column(Vector(None))

    def to_document(self, include_embeddings: bool = False) -> EmbeddedDocument:
        metadata: dict = self.details

        # TODO: Move it to the `Document` class:
        metadata[VectorStoreMetadataFields.DOCUMENT_ID] = self.id
        metadata[VectorStoreMetadataFields.TABLE_NAME] = self.__tablename__

        return EmbeddedDocument(
            self.document,
            metadata=metadata,
            # Use None to save RAM space:
            embeddings=self.embeddings if include_embeddings else None,
        )


class BaseDatasetDocumentMapping(BaseModel):
    __abstract__ = True  # this line is necessary

    dataset_id: Mapped[int]
    document_id: Mapped[int]


class CollectionName(DateMixin, Base):
    __tablename__ = "_names"

    uuid: Mapped[uuid_module.UUID] = mapped_column(
        primary_key=True, type_=postgresql.UUID, default=uuid_module.uuid4
    )
    collection_name: Mapped[str] = mapped_column()
    datasource: Mapped[str | None] = mapped_column(default=None)
    embedding_model_name: Mapped[str] = mapped_column()

    @property
    def table_name(self) -> str:
        return f"c_{self.uuid}"


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
    def get_dataset_document_mapping_model(cls, name: str) -> type[BaseDatasetDocumentMapping]:
        if name not in cls._models:
            cls._models[name] = type(name, (BaseDatasetDocumentMapping,), {"__tablename__": name})
        return cls._models[name]
