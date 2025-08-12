import typing as t
from abc import ABC, abstractmethod

from sdmx.model import common

from common.data.base import EntityType

from .base import BaseNameableArtefact
from .category import CodeCategory


class BaseSdmxCodeList(BaseNameableArtefact[common.Codelist], ABC):
    @property
    def entity_type(self) -> EntityType:
        return EntityType.OTHER

    @property
    def entity_id(self) -> str:
        return self.short_urn

    @property
    @abstractmethod
    def code_list(self) -> common.Codelist:
        pass

    @abstractmethod
    def codes(self) -> t.Sequence[CodeCategory]:
        pass

    @abstractmethod
    def __contains__(self, item: str) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, item: str) -> CodeCategory | None:
        pass


class InMemoryCodeList(BaseSdmxCodeList):
    _code_list: common.Codelist
    _codes: t.Dict[str, CodeCategory]

    def __init__(
        self,
        code_list: common.Codelist,
        locale: str,
    ):
        super().__init__(code_list, locale)
        self._code_list = code_list
        self._codes = {code.id: CodeCategory(code, locale) for code in code_list.items.values()}

    @property
    def code_list(self) -> common.Codelist:
        return self._code_list

    def codes(self) -> t.Sequence[CodeCategory]:
        return list(self._codes.values())

    def __getitem__(self, item: str) -> CodeCategory | None:
        return self._codes.get(item)

    def __contains__(self, item: str) -> bool:
        return item in self._codes
