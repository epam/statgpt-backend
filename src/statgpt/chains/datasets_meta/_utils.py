from common.schemas.enums import AvailableDatasetsVersion, LocaleEnum
from statgpt.utils.formatters import CitationFormatterConfig, DatasetFormatterConfig


def _create_formatter_config(
    version: AvailableDatasetsVersion, locale: LocaleEnum
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
            )
        case AvailableDatasetsVersion.short:
            return DatasetFormatterConfig(
                locale=locale,
                add_source_id=True,
                add_entity_id=False,
                use_description=False,
                citation=None,
                highlight_name_in_bold=False,
            )
        case _:
            raise ValueError(f"Unsupported AvailableDatasetsVersion: {version}")
