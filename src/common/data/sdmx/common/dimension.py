import typing as t
from abc import ABC

from sdmx.model import common

from common.data.base import CategoricalDimension, Category, DateTimeDimension, Dimension

from .base import BaseIdentifiableArtefact
from .category import DimensionCodeCategory
from .codelist import BaseSdmxCodeList


class SdmxDimension(BaseIdentifiableArtefact[common.DimensionComponent], Dimension[t.Any], ABC):
    _name: str
    _description: t.Optional[str]
    _alias: str | None

    def __init__(
        self,
        dimension: common.DimensionComponent,
        name: str,
        description: t.Optional[str],
        alias: str | None = None,
    ):
        super().__init__(dimension)
        Dimension.__init__(self)
        self._name = name
        self._description = description
        self._alias = alias

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> t.Optional[str]:
        return self._description

    @property
    def alias(self) -> str | None:
        return self._alias

    @property
    def is_mandatory(self) -> bool:
        return True

    @property
    def entity_id(self) -> str:
        return self._artefact.id

    @property
    def is_time_dimension(self) -> bool:
        return False


class SdmxCodeListDimension(CategoricalDimension[DimensionCodeCategory], SdmxDimension):
    _code_list: BaseSdmxCodeList

    def __init__(
        self,
        dimension: common.DimensionComponent,
        name: str,
        description: t.Optional[str],
        code_list: BaseSdmxCodeList,
        available_codes: t.Iterable[str] | None = None,
        alias: str | None = None,
    ):
        SdmxDimension.__init__(self, dimension, name, description, alias=alias)
        self._code_list = code_list
        self._available_codes = set(available_codes) if available_codes else None

    @property
    def available_values(self) -> t.Sequence[DimensionCodeCategory]:
        if not self._available_codes:
            return self.values
        # TODO: could probably use a simpler version
        # return [item for item in self.values if item.query_id in self._available_codes]
        return [
            DimensionCodeCategory.from_code_category(item, self.entity_id, self._name, self._alias)
            for item in self.values
            if item.query_id in self._available_codes
        ]

    @property
    def code_list(self) -> BaseSdmxCodeList:
        return self._code_list

    @property
    def values(self) -> t.List[DimensionCodeCategory]:
        return [
            DimensionCodeCategory.from_code_category(c, self.entity_id, self._name, self._alias)
            for c in self._code_list.codes()
        ]

    def has_value(self, value: Category) -> bool:
        return value.entity_id in self._code_list

    def format_value(self, value: DimensionCodeCategory) -> str:
        return f"id={value.entity_id}, name={value.name}, description={value.description}"


class SdmxTimeDimension(SdmxDimension, DateTimeDimension):
    _time_dimension: bool

    def __init__(
        self,
        dimension: common.DimensionComponent,
        name: str,
        description: t.Optional[str],
        time_dimension: bool,
        alias: str | None = None,
    ):
        super().__init__(dimension, name, description, alias=alias)
        DateTimeDimension.__init__(self)
        self._time_dimension = time_dimension

    @property
    def is_time_dimension(self) -> bool:
        return self._time_dimension

    def format_value(self, value: str) -> str:
        return value
