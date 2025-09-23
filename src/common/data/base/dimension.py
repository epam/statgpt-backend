import json
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from .base import BaseEntity, EntityType
from .category import Category, VirtualDimensionCategory, VirtualDimensionValue
from .enums import DimensionType, QueryOperator

DimensionValueType = t.TypeVar("DimensionValueType")


class VirtualDimensionConfig(BaseModel):
    id: str = Field(description="The ID of the virtual dimension")
    name: str = Field(description="The name of the virtual dimension")
    description: str | None = Field(description="The description of the virtual dimension")
    value: VirtualDimensionValue = Field(description="The value of the virtual dimension")


class Dimension(BaseEntity, t.Generic[DimensionValueType], ABC):
    def __init__(self):
        BaseEntity.__init__(self)

    @property
    def entity_type(self) -> EntityType:
        return EntityType.DIMENSION

    @property
    def alias(self) -> str | None:
        return None

    @property
    @abstractmethod
    def dimension_type(self) -> DimensionType:
        pass

    @property
    @abstractmethod
    def is_mandatory(self) -> bool:
        pass

    @abstractmethod
    def format_value(self, value: DimensionValueType) -> str:
        pass

    @abstractmethod
    def available_operators(self) -> t.List[QueryOperator]:
        # TODO: we don't use this, should we? Maybe we can delete it?
        pass

    def available_operators_str(self) -> str:
        return json.dumps([op.value for op in self.available_operators()])


CategoryType = t.TypeVar("CategoryType", bound=Category)


class CategoricalDimension(Dimension[CategoryType], t.Generic[CategoryType], ABC):
    def __init__(self):
        super().__init__()

    @property
    def dimension_type(self) -> DimensionType:
        return DimensionType.CATEGORY

    @property
    @abstractmethod
    def values(self) -> t.Sequence[CategoryType]:
        """
        Return all possible values for this dimension.
        :return: A list of all possible values.
        """

    @property
    def available_values(self) -> t.Sequence[CategoryType]:
        """
        Returns the available values for this dimension (only the ones that are present in the data).
        Default implementation returns all values.
        :return: A list of available values.
        """
        return self.values

    @abstractmethod
    def name_by_query_id(self, query_id: str) -> str | None:
        pass

    @abstractmethod
    def has_value(self, value: CategoryType) -> bool:
        pass

    def __iter__(self):
        return iter(self.values)

    def available_operators(self) -> t.List[QueryOperator]:
        return [
            QueryOperator.IN,
        ]


class VirtualDimension(CategoricalDimension[VirtualDimensionCategory]):
    _id: str
    _name: str
    _alias: str | None
    _description: t.Optional[str]
    _value: VirtualDimensionCategory

    def __init__(self, virtual_dimension_config: VirtualDimensionConfig, alias: str | None = None):
        super().__init__()
        self._id = virtual_dimension_config.id
        self._name = virtual_dimension_config.name
        self._alias = alias
        self._description = virtual_dimension_config.description
        self._value = VirtualDimensionCategory(self._id, self._name, virtual_dimension_config.value)

    @property
    def values(self) -> t.Sequence[VirtualDimensionCategory]:
        return [self._value]

    def has_value(self, value: VirtualDimensionCategory) -> bool:
        return value.entity_id == self._value.entity_id

    def name_by_query_id(self, query_id: str) -> str | None:
        if query_id == self._value.query_id:
            return self._value.name
        return None

    @property
    def is_mandatory(self) -> bool:
        return False

    def format_value(self, value: VirtualDimensionCategory) -> str:
        return ""

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

    @property
    def alias(self) -> str | None:
        return self._alias


class DateTimeDimension(Dimension[str], ABC):
    def __init__(self):
        super().__init__()

    @property
    def dimension_type(self) -> DimensionType:
        return DimensionType.DATETIME

    def available_operators(self) -> t.List[QueryOperator]:
        # NOTE: does EQUALS operator make sense?
        return [
            QueryOperator.EQUALS,
            QueryOperator.GREATER_THAN_OR_EQUALS,
            QueryOperator.LESS_THAN_OR_EQUALS,
            QueryOperator.BETWEEN,
        ]
