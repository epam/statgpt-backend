from pydantic import BaseModel

from common.auth.auth_context import AuthContext
from common.data.base import DataSet, DatasetCitation


class CitationFormatterConfig(BaseModel):
    as_md_list: bool = True
    n_tabs: int = 0
    use_provider: bool = True
    use_last_updated: bool = True
    use_url: bool = True
    last_updated_override_value: str = ""


class CitationFormatter:
    def __init__(self, config: CitationFormatterConfig):
        self._config: CitationFormatterConfig = config

    def _format_provider(self, provider: str | None) -> str:
        if self._config.use_provider and provider:
            return f'Provider: {provider}'
        return ""

    def _format_last_updated(self, last_updated: str | None) -> str:
        if self._config.use_last_updated and last_updated:
            return f'Last updated: {last_updated}'
        return ""

    def _format_url(self, url: str | None) -> str:
        if self._config.use_url and url:
            return f'URL: {url}'
        return ""

    def format(self, citation: DatasetCitation) -> str:
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
    def __init__(self, config: CitationFormatterConfig, last_updated_override_value: str = ""):
        self._last_updated_override_value = last_updated_override_value
        super().__init__(config)

    def _format_last_updated(self, last_updated: str) -> str:
        if self._last_updated_override_value:
            return super()._format_last_updated(self._last_updated_override_value)
        return super()._format_last_updated(last_updated)


class DatasetFormatterConfig(BaseModel):
    include_name: bool = True

    add_source_id: bool = True
    source_id_name: str = 'ID'
    add_entity_id: bool = False
    entity_id_name: str = 'Internal ID'
    use_description: bool = True
    highlight_name_in_bold: bool = True
    official_dataset_label: str | None = None
    citation: CitationFormatterConfig = CitationFormatterConfig(n_tabs=1, as_md_list=True)

    @classmethod
    def create_citation_only(cls, citation=CitationFormatterConfig(as_md_list=True)):
        return cls(
            include_name=False,
            add_source_id=False,
            add_entity_id=False,
            use_description=False,
            citation=citation,
        )


class DatasetFormatter:
    def __init__(self, config: DatasetFormatterConfig, auth_context: AuthContext):
        self._config: DatasetFormatterConfig = config
        self._auth_context: AuthContext = auth_context

    async def _get_dataset_update_at(self, dataset: DataSet) -> str:
        last_updated = ""
        if last_updated_date := await dataset.updated_at(self._auth_context):
            last_updated = last_updated_date.strftime("%b %Y")
        return last_updated

    async def format(self, dataset: DataSet) -> str:
        citation = dataset.config.citation
        result = []
        if self._config.include_name:
            name_str = (
                f'**{dataset.name}**' if self._config.highlight_name_in_bold else dataset.name
            )
            if self._config.official_dataset_label and dataset.config.is_official:
                name_str += f' {self._config.official_dataset_label}'
            result.append(f'* {name_str}')

        if self._config.add_source_id:
            result.append(f'\t* {self._config.source_id_name}: {dataset.source_id}')
        if self._config.add_entity_id:
            result.append(f'\t* {self._config.entity_id_name}: {dataset.entity_id}')
        if self._config.use_description:
            description = (
                citation.description if citation and citation.description else dataset.description
            )
            if description:
                result.append(f'\t* Description: {description}')
        if (
            self._config.citation.use_provider
            or self._config.citation.use_last_updated
            or self._config.citation.use_url
        ) and citation:
            last_updated = await self._get_dataset_update_at(dataset)
            formatted_citation = CitationOverrideFormatter(
                self._config.citation, last_updated
            ).format(citation)
            result.append(formatted_citation)
        return "\n".join(result)


class DatasetListFormatter:
    def __init__(self, config: DatasetFormatterConfig, auth_context: AuthContext):
        self._config: DatasetFormatterConfig = config
        self._auth_context: AuthContext = auth_context

    async def format(self, datasets: list[DataSet], sort_by_id: bool = False) -> str:
        if not self._config.add_source_id and not self._config.add_entity_id:
            raise ValueError('At least one of add_source_id or add_entity_id must be True')

        dataset_entries = []

        if sort_by_id is True:
            iterable = sorted(datasets, key=lambda ds: (not ds.config.is_official, ds.entity_id))
        else:
            iterable = sorted(datasets, key=lambda ds: ds.config.is_official, reverse=True)

        formatter = DatasetFormatter(self._config, self._auth_context)
        for dataset in iterable:
            entry = await formatter.format(dataset)
            dataset_entries.append(entry)

        datasets_list = '\n'.join(dataset_entries)
        result = datasets_list
        return result
