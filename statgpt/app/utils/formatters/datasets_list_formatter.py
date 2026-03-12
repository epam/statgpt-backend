import logging
from collections import defaultdict

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import DataSet
from statgpt.common.schemas.enums import AvailableDatasetsHeaderFormat

from .dataset_base import BaseDatasetFormatter, DatasetFormatterConfig
from .dataset_detailed import DetailedDatasetFormatter
from .dataset_simple import SimpleDatasetFormatter

_log = logging.getLogger(__name__)


class DatasetsListFormatter:
    """Dataset list formatter with localization support."""

    def __init__(
        self, config: DatasetFormatterConfig, auth_context: AuthContext, detailed: bool = False
    ):
        self._config = config
        self._auth_context = auth_context
        self._formatter: BaseDatasetFormatter

        # Choose formatter based on detailed flag
        if detailed:
            self._formatter = DetailedDatasetFormatter(config, auth_context)
        else:
            self._formatter = SimpleDatasetFormatter(config, auth_context)

        # Get translation function
        self._ = self._formatter._

    async def format(
        self,
        datasets: list[DataSet],
        sort_by_id: bool = False,
        sort_by_name: bool = False,
        add_stats: bool = False,
        group_by_provider: bool = False,
        indicator_counts: dict[str, int] | None = None,
    ) -> str:
        if sort_by_id and sort_by_name:
            raise ValueError(self._("Cannot sort by both id and name."))

        dataset_entries = defaultdict(list)

        # Sort datasets based on criteria
        if sort_by_id:
            iterable = sorted(datasets, key=lambda ds: (not ds.config.is_official, ds.entity_id))
        elif sort_by_name:
            iterable = sorted(datasets, key=lambda ds: (not ds.config.is_official, ds.name.lower()))
        else:
            iterable = sorted(datasets, key=lambda ds: ds.config.is_official, reverse=True)

        # Format each dataset
        for dataset in iterable:
            count = indicator_counts.get(dataset.entity_id) if indicator_counts else None
            entry = await self._formatter.format(dataset, indicator_count=count)
            provider = dataset.config.citation.provider if dataset.config.citation else None
            dataset_entries[provider].append(entry)

        # Group by provider if requested
        if group_by_provider:
            grouped_entries = []
            for provider, entries in dataset_entries.items():
                if provider:
                    grouped_entries.append(f'### {self._("Provider")}: {provider}\n')
                else:
                    grouped_entries.append(f'### {self._("Provider")}: {self._("Unknown")}\n')
                if add_stats:
                    grouped_entries.append(
                        f'{self._("Total datasets from this provider")}: {len(entries)}\n'
                    )
                grouped_entries.extend(entries)
                grouped_entries.append('')  # Add a newline between providers
            datasets_list = '\n'.join(grouped_entries).strip()
        else:
            all_entries = [entry for entries in dataset_entries.values() for entry in entries]
            datasets_list = '\n'.join(all_entries)

        # Add overall statistics if requested
        if add_stats:
            stats_str = self._format_stats_header(
                datasets=datasets,
                dataset_entries=dataset_entries,
                group_by_provider=group_by_provider,
                indicator_counts=indicator_counts,
            )
            result = f'{stats_str}\n\n{datasets_list}'
        else:
            result = datasets_list

        return result

    def _format_stats_header(
        self,
        datasets: list[DataSet],
        dataset_entries: dict,
        group_by_provider: bool,
        indicator_counts: dict[str, int] | None,
    ) -> str:
        if self._config.stats_header_format == AvailableDatasetsHeaderFormat.agencies:
            return self._format_agencies_header(
                datasets=datasets,
                indicator_counts=indicator_counts,
            )
        return self._format_totals_header(
            datasets=datasets,
            dataset_entries=dataset_entries,
            group_by_provider=group_by_provider,
            indicator_counts=indicator_counts,
        )

    def _format_totals_header(
        self,
        datasets: list[DataSet],
        dataset_entries: dict,
        group_by_provider: bool,
        indicator_counts: dict[str, int] | None,
    ) -> str:
        stats_header = f'{self._("Total datasets")}: {len(datasets)}'
        if group_by_provider:
            providers = [p for p in dataset_entries.keys() if p is not None]
            stats_header += f'\n{self._("Total providers")}: {len(providers)}'
        if indicator_counts is not None:
            stats_header += (
                f'\n{self._("Total number of indicators")}: {sum(indicator_counts.values())}'
            )
        return stats_header

    def _format_agencies_header(
        self,
        datasets: list[DataSet],
        indicator_counts: dict[str, int] | None,
    ) -> str:
        if indicator_counts is not None:
            n_indicators = sum(indicator_counts.values())
        else:
            n_indicators = None

        providers_set: set[str] = set()
        for dataset in datasets:
            if not (citation := dataset.config.citation):
                continue
            agency_names = citation.provider_agency_names_with_fallback_to_provider
            if not agency_names:
                _log.warning(f'Dataset {dataset.entity_id} has no provider information')
                continue
            providers_set.update(agency_names)

        n_providers = len(providers_set)
        providers_sample = sorted(providers_set)[:3]
        providers_sample_str = ', '.join(providers_sample)
        and_others = ' ' + self._("and others") if n_providers > len(providers_sample) else ""
        providers_str = self._(
            "provided by {n_providers} agencies, including: {providers_sample_str}{and_others}."
        ).format(
            n_providers=n_providers,
            providers_sample_str=providers_sample_str,
            and_others=and_others,
        )

        if n_indicators is not None:
            return self._("I have access to {n_indicators} indicators {providers_str}").format(
                n_indicators=n_indicators, providers_str=providers_str
            )
        return self._("I have access to data {providers_str}").format(providers_str=providers_str)

    async def format_summary(
        self, datasets: list[DataSet], include_official_count: bool = True
    ) -> str:
        """Generate a summary of the datasets."""
        total = len(datasets)

        if include_official_count:
            official_count = sum(1 for ds in datasets if ds.config.is_official)
            unofficial_count = total - official_count

            summary_lines = [
                f'{self._("Total datasets")}: {total}',
                f'  - {self._("Official")}: {official_count}',
                f'  - {self._("Unofficial")}: {unofficial_count}',
            ]
        else:
            summary_lines = [f'{self._("Total datasets")}: {total}']

        # Count by provider
        provider_counts: dict = defaultdict(int)
        for ds in datasets:
            provider = ds.config.citation.provider if ds.config.citation else self._("Unknown")
            provider_counts[provider] += 1

        if provider_counts:
            summary_lines.append(f'\n{self._("By provider")}:')
            for provider, count in sorted(
                provider_counts.items(), key=lambda x: x[1], reverse=True
            ):
                summary_lines.append(f'  - {provider}: {count}')

        return '\n'.join(summary_lines)


class IndexedDatasetsListFormatter:
    """Dataset list formatter with localization support. Includes index numbers for each dataset."""

    def __init__(
        self, config: DatasetFormatterConfig, auth_context: AuthContext, detailed: bool = False
    ):
        self._config = config
        self._auth_context = auth_context
        self._formatter: BaseDatasetFormatter

        # Choose formatter based on detailed flag
        if detailed:
            self._formatter = DetailedDatasetFormatter(config, auth_context)
        else:
            self._formatter = SimpleDatasetFormatter(config, auth_context)

        # Get translation function
        self._ = self._formatter._

    async def format(self, datasets: dict[int, DataSet], index_name: str = 'Index') -> str:
        dataset_entries = []
        for index, dataset in sorted(datasets.items(), key=lambda item: item[0]):
            entry = await self._formatter.format(dataset)
            item_tabs = '\t' * self._config.list_level
            first_row, other_rows = entry.split('\n', 1)
            entry = f"{first_row}\n{item_tabs}* {index_name}: {index}\n{other_rows}"
            dataset_entries.append(entry)

        return '\n\n'.join(dataset_entries)
