import gettext
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from common.schemas.enums import LocaleEnum

_FormatValue = TypeVar("_FormatValue")


class BaseFormatter(ABC, Generic[_FormatValue]):
    def __init__(self, domain: str, locale: LocaleEnum):

        locale_dir = Path(__file__).parent / "locales"
        self.translation: gettext.NullTranslations | gettext.GNUTranslations
        try:
            # noinspection PyTypeChecker
            self.translation = gettext.translation(
                domain, localedir=locale_dir, languages=[locale.value], fallback=False
            )
        except FileNotFoundError:
            # noinspection PyTypeChecker
            self.translation = gettext.translation(
                domain, localedir=locale_dir, languages=[LocaleEnum.EN.value], fallback=True
            )

        self._ = self.translation.gettext

    @abstractmethod
    async def format(self, value: _FormatValue) -> str:
        pass
