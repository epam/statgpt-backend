from statgpt.app.schemas.mcp import AvailableDatasetsStructuredContent, DatasetRecord
from statgpt.app.utils.formatters import CitationFormatterConfig, DatasetFormatterConfig
from statgpt.common.data.base import DataSet
from statgpt.common.schemas.enums import (
    AvailableDatasetsHeaderFormat,
    AvailableDatasetsVersion,
    LocaleEnum,
)


def datasets_to_structured_content(
    datasets: list[DataSet],
) -> AvailableDatasetsStructuredContent:
    """Build the MCP structured content shared by the available-datasets and datasets-metadata
    tools: each dataset as a record keyed by its stable URN (source id)."""
    return AvailableDatasetsStructuredContent(
        datasets=[
            DatasetRecord(id=ds.source_id, name=ds.name, url=ds.dataset_url) for ds in datasets
        ],
        count=len(datasets),
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
