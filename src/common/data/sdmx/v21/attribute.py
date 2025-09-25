import typing as t
from abc import ABC

from sdmx.model import common

from common.data.base import Attribute, CategoricalAttribute, Category, StringAttribute
from common.data.sdmx.common.base import BaseIdentifiableArtefact
from common.data.sdmx.common.category import CodeCategory
from common.data.sdmx.common.codelist import BaseSdmxCodeList


class Sdmx21Attribute(BaseIdentifiableArtefact[common.DataAttribute], Attribute[t.Any], ABC):
    _name: str
    _description: t.Optional[str]
    _alias: str | None

    def __init__(
        self,
        attribute: common.DataAttribute,
        name: str,
        description: t.Optional[str],
        alias: str | None = None,
    ):
        super().__init__(attribute)
        Attribute.__init__(self)
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
    def entity_id(self) -> str:
        return self._artefact.id


class Sdmx21CodeListAttribute(CategoricalAttribute[CodeCategory], Sdmx21Attribute):
    _code_list: BaseSdmxCodeList

    def __init__(
        self,
        attribute: common.DataAttribute,
        name: str,
        description: t.Optional[str],
        code_list: BaseSdmxCodeList,
    ):
        Sdmx21Attribute.__init__(self, attribute, name, description)
        self._code_list = code_list

    @property
    def code_list(self) -> BaseSdmxCodeList:
        return self._code_list

    @property
    def values(self) -> t.List[CodeCategory]:
        return self._code_list.codes()  # type: ignore

    def has_value(self, value: Category) -> bool:
        return value.entity_id in self._code_list

    def format_value(self, value: CodeCategory) -> str:
        return f"id={value.entity_id}, name={value.name}, description={value.description}"


class Sdmx21StringAttribute(StringAttribute, Sdmx21Attribute):

    def __init__(
        self,
        attribute: common.DataAttribute,
        name: str,
        description: t.Optional[str],
    ):
        Sdmx21Attribute.__init__(self, attribute, name, description)

    def format_value(self, value: str) -> str:
        return value
