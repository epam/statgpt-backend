import typing as t
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from pydantic import BaseModel

from .base import BaseEntity, EntityType


class VirtualDimensionValue(BaseModel):
    id: str
    name: str
    description: str | None


class Category(BaseEntity, ABC):
    @property
    def query_id(self) -> str:
        """The ID used in queries"""
        return self.entity_id

    @property
    def dimension_id(self) -> str | None:
        return None

    def to_document(self, **kwargs) -> Document:
        return Document(
            page_content=self.get_document_content(**kwargs),
            metadata=self.get_document_metadata(**kwargs),
        )

    def get_document_content(self, **kwargs) -> str:
        s = f"id: {self.entity_id}, name: {self.name}"
        if self.description:
            s += f", description: {self.description}"
        return s

    def get_document_metadata(self, **kwargs) -> dict:
        return {}

    @classmethod
    @abstractmethod
    def from_document(cls, document: Document) -> 'Category':
        pass

    @property
    def entity_type(self) -> EntityType:
        return EntityType.CATEGORY


class DimensionCategory(Category, ABC):
    @property
    @abstractmethod
    def dimension_id(self) -> str:
        pass

    @property
    @abstractmethod
    def dimension_name(self) -> str:
        pass

    @property
    def dimension_alias(self) -> str | None:
        return None


class VirtualDimensionCategory(DimensionCategory):

    METADATA_DIMENSION_ID = "virtual_dimension_id"
    METADATA_DIMENSION_NAME = "virtual_dimension_name"
    METADATA_VIRTUAL_DIMENSION_VALUE_ID = "virtual_dimension_value_id"
    METADATA_VIRTUAL_DIMENSION_VALUE_NAME = "virtual_dimension_value_name"
    METADATA_VIRTUAL_DIMENSION_VALUE_DESCRIPTION = "virtual_dimension_value_description"

    def __init__(
        self, dimension_id: str, dimension_name: str, virtual_dimension_value: VirtualDimensionValue
    ):
        self._dimension_id = dimension_id
        self._dimension_name = dimension_name
        self._id = virtual_dimension_value.id
        self._name = virtual_dimension_value.name
        self._description = virtual_dimension_value.description

    @property
    def dimension_id(self) -> str:
        return self._dimension_id

    @property
    def dimension_name(self) -> str:
        return self._dimension_name

    @classmethod
    def from_document(cls, document: Document) -> "VirtualDimensionCategory":
        return cls(
            dimension_id=document.metadata[cls.METADATA_DIMENSION_ID],
            dimension_name=document.metadata[cls.METADATA_DIMENSION_NAME],
            virtual_dimension_value=VirtualDimensionValue(
                id=document.metadata[cls.METADATA_VIRTUAL_DIMENSION_VALUE_ID],
                name=document.metadata[cls.METADATA_VIRTUAL_DIMENSION_VALUE_NAME],
                description=document.metadata[cls.METADATA_VIRTUAL_DIMENSION_VALUE_DESCRIPTION],
            ),
        )

    def get_document_content(self, include_description: bool = False, **kwargs) -> str:
        content = f"id: {self._id}, name: {self._name}"
        if self.description and include_description:
            content += f", description: {self._description}"
        return f"{content} ({self._dimension_name})"

    def get_document_metadata(self, **kwargs) -> dict:
        return {
            self.METADATA_DIMENSION_ID: self._dimension_id,
            self.METADATA_DIMENSION_NAME: self._dimension_name,
            self.METADATA_VIRTUAL_DIMENSION_VALUE_ID: self._id,
            self.METADATA_VIRTUAL_DIMENSION_VALUE_NAME: self._name,
            self.METADATA_VIRTUAL_DIMENSION_VALUE_DESCRIPTION: self._description,
        }

    @property
    def entity_id(self) -> str:
        return self._id

    @property
    def source_id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> t.Optional[str]:
        return self._description
