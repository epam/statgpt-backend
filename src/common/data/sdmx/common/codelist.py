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
    def __getitem__(self, item: str) -> CodeCategory:
        pass


class InMemoryCodeList(BaseSdmxCodeList):
    _code_list: common.Codelist
    _codes: t.Dict[str, CodeCategory]

    def __init__(self, code_list: common.Codelist, locale: str):
        super().__init__(code_list, locale)
        self._code_list = code_list
        self._codes = {}

    def _get_item_and_cache(self, item: str) -> CodeCategory | None:
        if item not in self._codes:
            code = self._code_list[item]
            if code is None:
                return None
            self._codes[item] = CodeCategory(code, self._locale)
        return self._codes[item]

    def _get_item_and_cache_or_raise(self, item: str) -> CodeCategory:
        code = self._get_item_and_cache(item)
        if code is None:
            raise KeyError(f"Code '{item}' not found in codelist '{self.code_list.id}'")
        return code

    @property
    def code_list(self) -> common.Codelist:
        return self._code_list

    def codes(self) -> t.Sequence[CodeCategory]:
        if len(self._codes) == len(self._code_list.items):
            return list(self._codes.values())
        return [self._get_item_and_cache_or_raise(code) for code in self._code_list.items.values()]

    def __getitem__(self, item: str) -> CodeCategory:
        return self._get_item_and_cache_or_raise(item)

    def __contains__(self, item: str) -> bool:
        return item in self._code_list
