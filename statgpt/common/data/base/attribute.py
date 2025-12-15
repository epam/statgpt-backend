import typing as t
from abc import ABC, abstractmethod

from .base import BaseEntity, EntityType
from .category import Category
from .enums import AttributeType

AttributeValueType = t.TypeVar("AttributeValueType")


class Attribute(BaseEntity, t.Generic[AttributeValueType], ABC):
    def __init__(self):
        BaseEntity.__init__(self)

    @property
    def entity_type(self) -> EntityType:
        return EntityType.ATTRIBUTE

    @property
    @abstractmethod
    def attribute_type(self) -> AttributeType:
        pass

    @abstractmethod
    def format_value(self, value: AttributeValueType) -> str:
        pass


CategoryType = t.TypeVar("CategoryType", bound=Category)


class CategoricalAttribute(Attribute[CategoryType], t.Generic[CategoryType], ABC):
    def __init__(self):
        super().__init__()

    @property
    def attribute_type(self) -> AttributeType:
        return AttributeType.CATEGORY

    @property
    @abstractmethod
    def values(self) -> t.Sequence[CategoryType]:
        """
        Return all possible values for this attribute.
        :return: A list of all possible values.
        """

    def __iter__(self):
        return iter(self.values)


class StringAttribute(Attribute[str], ABC):
    def __init__(self):
        super().__init__()

    @property
    def attribute_type(self) -> AttributeType:
        return AttributeType.STRING
