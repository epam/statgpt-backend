from pydantic import BaseModel

from statgpt.app.utils.formatters.base import BaseFormatter
from statgpt.common.data.base import DatasetCitation, ProviderAgency
from statgpt.common.schemas.enums import LocaleEnum


class CitationFormatterConfig(BaseModel):
    as_md_list: bool = True
    n_tabs: int = 0
    use_provider: bool = True
    use_last_updated: bool = True
    use_url: bool = True
    last_updated_override_value: str = ""
    include_provider_agencies: bool = False

    @property
    def is_use_any(self) -> bool:
        return self.use_provider or self.use_last_updated or self.use_url


class CitationFormatter(BaseFormatter):
    def __init__(self, config: CitationFormatterConfig, locale: LocaleEnum):
        super().__init__("dataset", locale)
        self._config: CitationFormatterConfig = config

    def _format_provider(self, citation: DatasetCitation) -> str:
        if not self._config.use_provider:
            return ""
        prefix = f'{self._("Provider")}: '
        if template := citation.provider_template:
            agencies = citation.provider_agencies or []
            # ids are expected to be unique - no more than one "<NA>" could be present in the data
            sample = [x for x in agencies[:4] if x.id != '<NA>'][:3]
            sample_str = ', '.join([x.name for x in sample])
            if len(agencies) > len(sample):
                sample_str += ' ' + self._("and others")
            formatted = template.format(n_agencies=len(agencies), agencies_sample=sample_str)
            return f'{prefix}{formatted}'
        if provider := citation.provider:
            return f'{prefix}{provider}'
        return ""

    def _format_last_updated(self, last_updated: str | None) -> str:
        if self._config.use_last_updated and last_updated:
            return f'{self._("Last updated")}: {last_updated}'
        return ""

    def _format_url(self, url: str | None) -> str:
        if self._config.use_url and url:
            return f'{self._("URL")}: {url}'
        return ""

    def _format_provider_agencies(self, provider_agencies: list[ProviderAgency] | None) -> str:
        if not self._config.include_provider_agencies:
            return ""
        if not provider_agencies:
            return ""

        # NOTE: first line is going to be prefixed outside of this function.
        # we need to prefix all lines after the first one.
        inner_lines_prefix = '\t' * (self._config.n_tabs + 1)

        lines = [self._("Provider agencies")]
        agency_names = [agency.name for agency in provider_agencies]
        lines += [
            f'{inner_lines_prefix}{i}. {name}' for i, name in enumerate(agency_names, start=1)
        ]
        joined = '\n'.join(lines)
        return joined

    async def format(self, citation: DatasetCitation) -> str:
        lines = []
        if provider := self._format_provider(citation):
            lines.append(provider)

        if last_updated := self._format_last_updated(citation.last_updated):
            lines.append(last_updated)

        if url := self._format_url(citation.get_url()):
            lines.append(url)

        if agencies_str := self._format_provider_agencies(citation.provider_agencies):
            lines.append(agencies_str)

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
        url_override_value: str | None = None,
    ):
        super().__init__(config, locale)
        self._last_updated_override_value = last_updated_override_value
        self._url_override_value = url_override_value

    def _format_last_updated(self, last_updated: str | None) -> str:
        if self._last_updated_override_value:
            return super()._format_last_updated(self._last_updated_override_value)
        return super()._format_last_updated(last_updated)

    def _format_url(self, url: str | None) -> str:
        if self._url_override_value:
            return super()._format_url(self._url_override_value)
        return super()._format_url(url)
