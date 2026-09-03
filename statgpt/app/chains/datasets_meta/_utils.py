import asyncio
from collections import defaultdict

from statgpt.app.schemas.mcp import (
    AvailableDatasetsStructuredContent,
    DatasetRecord,
    ProviderRecord,
)
from statgpt.app.utils.formatters import CitationFormatterConfig, DatasetFormatterConfig
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import DataSet
from statgpt.common.schemas.enums import (
    AvailableDatasetsHeaderFormat,
    AvailableDatasetsVersion,
    LocaleEnum,
)


async def _dataset_last_updated(dataset: DataSet, auth_context: AuthContext) -> str | None:
    """The dataset's last-updated date as an ISO 8601 string, from the source when known,
    otherwise the free-text citation value."""
    if updated_at := await dataset.updated_at(auth_context):
        return updated_at.date().isoformat()
    citation = dataset.config.citation
    return citation.last_updated if citation else None


async def datasets_to_structured_content(
    datasets: list[DataSet],
    auth_context: AuthContext,
    indicator_counts: dict[str, int] | None,
) -> AvailableDatasetsStructuredContent:
    """Build the MCP structured content for the available-datasets tool: each dataset as a record
    keyed by its stable URN (source id), the distinct providers with their dataset counts, and
    channel-wide totals."""
    last_updated = await asyncio.gather(
        *(_dataset_last_updated(ds, auth_context) for ds in datasets)
    )

    records: list[DatasetRecord] = []
    provider_counts: dict[str, int] = defaultdict(int)
    agencies: set[str] = set()
    for dataset, dataset_last_updated in zip(datasets, last_updated):
        citation = dataset.config.citation
        description = (
            citation.description if citation and citation.description else dataset.description
        )
        provider = citation.provider if citation else None
        if provider:
            provider_counts[provider] += 1
        if citation and (agency_names := citation.provider_agency_names_with_fallback_to_provider):
            agencies.update(agency_names)
        records.append(
            DatasetRecord(
                id=dataset.source_id,
                name=dataset.name,
                description=description,
                provider=provider,
                last_updated=dataset_last_updated,
                url=dataset.dataset_url,
                number_of_indicators=(
                    indicator_counts.get(dataset.entity_id) if indicator_counts else None
                ),
            )
        )

    providers = [
        ProviderRecord(name=name, dataset_count=count)
        for name, count in sorted(provider_counts.items())
    ]
    total_indicators = sum(indicator_counts.values()) if indicator_counts else None

    return AvailableDatasetsStructuredContent(
        providers=providers,
        datasets=records,
        total_datasets=len(datasets),
        total_indicators=total_indicators,
        total_agencies=len(agencies),
    )


def _create_formatter_config(
    version: AvailableDatasetsVersion,
    locale: LocaleEnum,
    stats_header_format: AvailableDatasetsHeaderFormat = AvailableDatasetsHeaderFormat.totals,
) -> DatasetFormatterConfig:
    """Create a dataset formatter config based on the tool configuration."""
    match version:
        case AvailableDatasetsVersion.full:
            return DatasetFormatterConfig(
                locale=locale,
                source_id_name='Source ID',
                add_source_id=True,
                add_entity_id=False,
                use_description=True,
                citation=CitationFormatterConfig(
                    as_md_list=True,
                    n_tabs=1,
                    use_provider=True,
                    use_last_updated=True,
                    use_url=True,
                ),
                highlight_name_in_bold=False,
                stats_header_format=stats_header_format,
            )
        case AvailableDatasetsVersion.short:
            return DatasetFormatterConfig(
                locale=locale,
                add_source_id=True,
                add_entity_id=False,
                use_description=False,
                citation=None,
                highlight_name_in_bold=False,
                stats_header_format=stats_header_format,
            )
        case _:
            raise ValueError(f"Unsupported AvailableDatasetsVersion: {version}")
