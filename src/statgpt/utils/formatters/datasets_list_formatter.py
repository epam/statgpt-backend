from collections import defaultdict

from common.auth.auth_context import AuthContext
from common.data.base import DataSet

from .dataset_base import BaseDatasetFormatter, DatasetFormatterConfig
from .dataset_detailed import DetailedDatasetFormatter
from .dataset_simple import SimpleDatasetFormatter


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
            entry = await self._formatter.format(dataset)
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
            stats_header = f'{self._("Total datasets")}: {len(datasets)}'
            if group_by_provider:
                # Count unique providers
                providers = [p for p in dataset_entries.keys() if p is not None]
                stats_header += f'\n{self._("Total providers")}: {len(providers)}'
            result = f'{stats_header}\n\n{datasets_list}'
        else:
            result = datasets_list

        return result

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
