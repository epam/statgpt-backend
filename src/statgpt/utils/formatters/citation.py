from pydantic import BaseModel

from common.data.base import DatasetCitation
from common.schemas.enums import LocaleEnum
from statgpt.utils.formatters.base import BaseFormatter


class CitationFormatterConfig(BaseModel):
    as_md_list: bool = True
    n_tabs: int = 0
    use_provider: bool = True
    use_last_updated: bool = True
    use_url: bool = True
    last_updated_override_value: str = ""

    @property
    def is_use_any(self) -> bool:
        return self.use_provider or self.use_last_updated or self.use_url


class CitationFormatter(BaseFormatter[DatasetCitation]):
    def __init__(self, config: CitationFormatterConfig, locale: LocaleEnum):
        super().__init__("dataset", locale)
        self._config: CitationFormatterConfig = config

    def _format_provider(self, provider: str | None) -> str:
        if self._config.use_provider and provider:
            return f'{self._("Provider")}: {provider}'
        return ""

    def _format_last_updated(self, last_updated: str | None) -> str:
        if self._config.use_last_updated and last_updated:
            return f'{self._("Last updated")}: {last_updated}'
        return ""

    def _format_url(self, url: str | None) -> str:
        if self._config.use_url and url:
            return f'{self._("URL")}: {url}'
        return ""

    async def format(self, citation: DatasetCitation) -> str:
        lines = []
        if provider := self._format_provider(citation.provider):
            lines.append(provider)

        if last_updated := self._format_last_updated(citation.last_updated):
            lines.append(last_updated)

        if url := self._format_url(citation.get_url()):
            lines.append(url)

        if self._config.as_md_list is False:
            return ', '.join(lines)

        prefix = '\t' * self._config.n_tabs + '* '
        lines = [f'{prefix}{line}' for line in lines]
        return '\n'.join(lines)


class CitationOverrideFormatter(CitationFormatter):
    def __init__(
        self,
        config: CitationFormatterConfig,
        locale: LocaleEnum,
        last_updated_override_value: str = "",
    ):
        super().__init__(config, locale)
        self._last_updated_override_value = last_updated_override_value

    def _format_last_updated(self, last_updated: str | None) -> str:
        if self._last_updated_override_value:
            return super()._format_last_updated(self._last_updated_override_value)
        return super()._format_last_updated(last_updated)
