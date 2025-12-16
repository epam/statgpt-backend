import gettext
from abc import ABC
from pathlib import Path

from statgpt.common.schemas.enums import LocaleEnum


class BaseFormatter(ABC):
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
