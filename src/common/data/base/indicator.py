import typing as t
from abc import ABC, abstractmethod

from langchain_core.documents import Document

from .base import BaseEntity, EntityType


class BaseIndicator(BaseEntity, ABC):

    def __init__(self):
        BaseEntity.__init__(self)

    @property
    def entity_id(self) -> str:
        return self.query_id

    @property
    def description(self) -> t.Optional[str]:
        return None

    @property
    def source_id(self) -> str:
        return self.query_id

    @property
    def entity_type(self) -> EntityType:
        return EntityType.INDICATOR

    @property
    def dimension_id(self) -> str | None:
        return None

    @property
    def query_id(self) -> str:
        return self.entity_id

    def to_document(self) -> Document:
        return Document(
            page_content=self.get_document_content(),
            metadata=self.get_document_metadata(),
        )

    @abstractmethod
    def get_document_content(self) -> str:
        pass

    @abstractmethod
    def get_document_metadata(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_document(cls, document: Document) -> 'BaseIndicator':
        pass
