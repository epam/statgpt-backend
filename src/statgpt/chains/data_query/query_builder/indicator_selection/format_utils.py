from common.auth.auth_context import AuthContext
from common.data.base import CategoricalDimension, DataSet, DimensionQuery
from common.data.sdmx.v21.dataset import Sdmx21DataSet
from common.schemas.enums import LocaleEnum
from statgpt.schemas.query_builder import DatasetDimensionTermNameType, DatasetDimQueriesType
from statgpt.services.chat_facade import VersionedDataSet
from statgpt.services.hybrid_searcher import HybridCandidateScored
from statgpt.settings.dial_app import dial_app_settings
from statgpt.utils.formatters import DatasetFormatterConfig, SimpleDatasetFormatter


class HybridCandidateScoredWithQuery(HybridCandidateScored):
    query: DatasetDimensionTermNameType


def format_hybrid_scored_dicts(dicts: list[dict]) -> dict:
    return {'count': len(dicts), 'items': dicts}


def format_hybrid_llm_scored(
    llm_scored: list[HybridCandidateScored], datasets_dict: dict[str, VersionedDataSet]
) -> dict:
    def _format_single(item: HybridCandidateScored) -> HybridCandidateScoredWithQuery:
        versioned_dataset = datasets_dict.get(str(item.dataset_id))
        if not versioned_dataset:
            raise ValueError(f"Dataset with ID {item.dataset_id} not found")
        dataset = versioned_dataset.data

        dataset_query = {}
        dim_mapping = {d.entity_id: d for d in dataset.dimensions()}
        for dim_query in item.series:
            dim_id, term_id = list(dim_query.items())[0]
            dimension = dim_mapping.get(dim_id)
            term_name = ''
            if isinstance(dimension, CategoricalDimension):
                term_name = dimension.name_by_query_id(term_id) or ''
            dataset_query[dim_id] = {term_id: term_name}

        query = {dataset.source_id: dataset_query}
        res = HybridCandidateScoredWithQuery(**item.model_dump(), query=query)
        return res

    results = [_format_single(item) for item in llm_scored]
    results = sorted(results, key=lambda x: (x.score, x.dataset_id), reverse=True)
    items = [item.model_dump(exclude={'where', 'series'}) for item in results]

    return {'count': len(items), 'items': items}


def format_hybrid_final_queries(
    final_queries: dict[str, list[DimensionQuery]], datasets_dict: dict[str, VersionedDataSet]
) -> dict:
    dataset_queries_formatted = {}
    for dataset_id, dimension_queries in final_queries.items():
        versioned_dataset = datasets_dict.get(str(dataset_id))
        if not versioned_dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        dataset = versioned_dataset.data
        if not isinstance(dataset, Sdmx21DataSet):
            raise TypeError(f'Expected Sdmx21DataSet, got {type(dataset)}')

        dim_queries_dict = {dq.dimension_id: dq.values for dq in dimension_queries}
        dataset_queries_formatted[dataset.source_id] = dataset.map_dim_queries_2_names(
            dim_queries_dict
        )

    return dataset_queries_formatted


class DatasetDimQueriesSimpleDictFormatter:
    """
    Format queries stored as a simple "dataset_id -> dimension_id -> list of dim values ids" dict
    """

    def __init__(self, datasets: dict[str, DataSet], auth_context: AuthContext):
        self.datasets = datasets
        self._auth_context = auth_context

    def format_query_single_dataset(
        self, dataset_id, query: dict[str, list[str]], n_tabs: int = 0
    ) -> str:
        dataset = self.datasets[dataset_id]
        if not isinstance(dataset, Sdmx21DataSet):
            raise TypeError(f'Expected Sdmx21DataSet, got {type(dataset)}')

        query_id2name_mapping = dataset.map_dim_queries_2_names(query)
        lines = []

        for dim_id, terms_id2name in query_id2name_mapping.items():
            terms_mapping_imputed = {id_: name or id_ for id_, name in terms_id2name.items()}
            lines.append(f'* {dim_id}:')
            lines.extend(
                f'\t* [{term_id}] {term_name}'
                for term_id, term_name in terms_mapping_imputed.items()
            )

        if n_tabs > 0:
            prefix = '\t' * n_tabs
            lines = [f'{prefix}{line}' for line in lines]
        concat = '\n'.join(lines)
        return concat

    async def format_multidataset_queries(
        self,
        queries: DatasetDimQueriesType,
        header_level: int = 4,
        dataset_citation: bool = False,
        print_is_official: bool = True,
    ) -> str:
        lines = []
        for dataset_id, dataset_query in sorted(queries.items(), key=lambda x: x[0]):
            dataset = self.datasets[dataset_id]

            prefix = '#' * header_level
            title = f'{prefix} {dataset.name}'
            if print_is_official and dataset.config.is_official:
                title += f' {dial_app_settings.official_dataset_label}'
            lines.append(title)

            n_tabs_for_query = 0
            if dataset_citation is True:
                lines.append(f'* ID: {dataset.source_id}')
                if dataset.config.citation:
                    dataset_entry = await SimpleDatasetFormatter(
                        DatasetFormatterConfig.create_citation_only(
                            locale=LocaleEnum.EN
                        ),  # ToDo: refactor as part of data query formatting localization
                        auth_context=self._auth_context,
                    ).format(dataset)

                    lines.append(dataset_entry)
                lines.append('* Query:')
                n_tabs_for_query = 1

            dataset_query_formatted = self.format_query_single_dataset(
                dataset_id=dataset_id, query=dataset_query, n_tabs=n_tabs_for_query
            )
            lines.append(dataset_query_formatted)
        concat = '\n'.join(lines)
        return concat
