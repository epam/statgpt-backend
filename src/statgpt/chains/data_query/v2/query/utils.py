from common.auth.auth_context import AuthContext
from common.config import logger
from common.data.base import (
    CategoricalDimension,
    DataResponse,
    DataSet,
    DataSetAvailabilityQuery,
    DataSetQuery,
    DateTimeDimension,
    Dimension,
    DimensionQuery,
    Query,
    QueryOperator,
)
from common.data.sdmx.v21.dataset import Sdmx21DataSet
from statgpt.config import DialAppConfig
from statgpt.schemas.query_builder import DatasetAvailabilityQueriesType, DatasetDimQueriesType
from statgpt.utils.dataset_formatter import DatasetFormatter, DatasetFormatterConfig


def filter_empty_dataset_availability_queries(queries: DatasetAvailabilityQueriesType):
    res = {
        dataset_id: dataset_query
        for dataset_id, dataset_query in queries.items()
        if not dataset_query.is_empty()
    }
    return res


def get_indicators_for_retrieval_results(
    best_of: dict[str, list[DimensionQuery]],  # TODO: rename
    datasets_dict: dict[str, DataSet],
    stage_name: str,
):
    results: list[dict] = []
    for dataset_id, dimension_queries in best_of.items():
        dataset = datasets_dict.get(str(dataset_id))
        if not dataset:
            logger.warning(f"Dataset with ID {dataset_id} not found")
            continue

        for dim_query in dimension_queries:
            dimension_id = dim_query.dimension_id

            for value in dim_query.values:
                indicator_name = next(
                    (
                        v.name
                        for d in dataset.indicator_dimensions()
                        for v in d.available_values
                        if v.query_id == value
                    ),
                    None,
                )
                results.append(
                    {
                        "score": 0.0,  # Dummy score value
                        "dataset_id": dataset.entity_id,
                        "dimension_id": dimension_id,
                        "query_id": value,
                        "name": indicator_name,
                    }
                )

    return {stage_name: results}


class DatasetDimQueriesSimpleDictFormatter:
    """
    Format queries stored as a simple "dataset_id -> dimension_id -> list of dim values ids" dict
    """

    def __init__(self, datasets: dict[str, DataSet], auth_context: AuthContext):
        self.datasets = datasets
        self._auth_context = auth_context

    def format_query_single_dataset(self, dataset_id, query: dict[str, list[str]], n_tabs: int = 0):
        dataset = self.datasets[dataset_id]
        if not isinstance(dataset, Sdmx21DataSet):
            raise TypeError(f'Expected InMemorySdmx21DataSet, got {type(dataset)}')

        query_id2name_mapping = dataset.map_dim_queries_2_names(query)
        lines = []

        # NOTE: we assume the query is valid
        for dim_id, values_id2name in query_id2name_mapping.items():
            lines.append(f'* {dim_id}:')
            lines.extend(
                f'\t* [{value_id}] {value_name}' for value_id, value_name in values_id2name.items()
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
    ):
        lines = []
        for dataset_id, dataset_query in sorted(queries.items(), key=lambda x: x[0]):
            dataset = self.datasets[dataset_id]

            prefix = '#' * header_level
            title = f'{prefix} {dataset.name}'
            if print_is_official and dataset.config.is_official:
                title += f' {DialAppConfig.OFFICIAL_DATASET_LABEL}'
            lines.append(title)

            n_tabs_for_query = 0
            if dataset_citation is True:
                lines.append(f'* ID: {dataset.source_id}')
                if dataset.config.citation:
                    dataset_entry = await DatasetFormatter(
                        DatasetFormatterConfig.create_citation_only(),
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


def format_datetime_dimension_query(dim_query: Query, dimension: DateTimeDimension) -> str:
    if dim_query.operator == QueryOperator.BETWEEN:
        start = dim_query.values[0]
        end = dim_query.values[1]
        if not start and not end:
            return 'no filter'
        return f"from **{start}** to **{end}**"
    elif dim_query.operator == QueryOperator.GREATER_THAN_OR_EQUALS:
        return f"from **{dim_query.values[0]}**"
    elif dim_query.operator == QueryOperator.LESS_THAN_OR_EQUALS:
        return f"until **{dim_query.values[0]}**"
    elif dim_query.operator == QueryOperator.EQUALS:
        # NOTE: does EQUALS operator make sense?
        return f"on **{dim_query.values[0]}**"
    else:
        raise ValueError(
            "Unsupported operator for DateTimeDimension: "
            f"{dim_query.operator}. dim_query: {dim_query}"
        )


async def format_dataset_queries(
    auth_context: AuthContext,
    dataset_queries: dict[str, DataSetQuery],
    datasets_dict: dict[str, DataSet],
    include_missing_dimensions: bool = False,
    include_default_queries: bool = False,
    include_auto_selects: bool = False,
    availability: DataSetAvailabilityQuery | None = None,
    print_is_official: bool = False,
    data_responses: dict[str, DataResponse | None] | None = None,
) -> str:
    # NOTE: there is code duplication with "format_availability_queries" function

    logger.info(f'formatting following dataset_queries: {dataset_queries}')

    datasets_entries: list[str] = []
    for dataset_id, query in sorted(dataset_queries.items(), key=lambda x: x[0]):
        dataset = datasets_dict[dataset_id]
        citation = dataset.config.citation

        dataset_entry = f'### {dataset.name}'

        if print_is_official and dataset.config.is_official:
            dataset_entry += f' {DialAppConfig.OFFICIAL_DATASET_LABEL}'

        if data_responses:
            response = data_responses.get(dataset_id)
            if response and not response.visual_dataframe.empty:
                dataset_entry += f'\n✅ **Execution result**: Data received, contains {response.visual_dataframe.shape[0]} series.'
                if response.url_query:
                    dataset_entry += f'\n\n[🔍 View data in explorer]({response.url_query})'
            else:
                dataset_entry += (
                    '\n❌ **Execution result**: A response was received, but it does not contain any data.'
                    '\n\n💡 **Advice:** Most likely, the query is generally correct, but there is no data for'
                    ' the specified time period. You may want to try selecting a different time period.'
                    ' Another option is to try to find relevant data in other datasets or using other tools.'
                )

        dataset_entry += f'\n* ID: {dataset.source_id}'

        if citation:
            formatted_citation = await DatasetFormatter(
                DatasetFormatterConfig.create_citation_only(),
                auth_context=auth_context,
            ).format(dataset)
            dataset_entry += f"\n{formatted_citation}"

        dataset_entry += "\n* Query:"

        indicators: set[str] = {d.entity_id for d in dataset.indicator_dimensions()}

        # TODO: can use a simpler function: dataset.map_dim_values_id_2_name()

        for dimension in dataset.dimensions():
            dim_query = next(
                (d for d in query.dimensions_queries if d.dimension_id == dimension.entity_id), None
            )

            if not dim_query:
                continue

            if not dim_query.values and dim_query.operator != QueryOperator.ALL:
                continue

            if dim_query.is_default and not include_default_queries:
                continue

            if dim_query.is_all_selected and not include_auto_selects:
                continue

            if dim_query.dimension_id in indicators:
                dataset_entry += f"\n\t* _{dimension.name}_ (Indicator): "
            else:
                dataset_entry += f"\n\t* _{dimension.name}_: "

            if dim_query.is_all_selected:
                dataset_entry += "**\\***"
                continue

            if isinstance(dimension, CategoricalDimension):
                available_values = {v.query_id: v.name for v in dimension.available_values}
                dataset_entry += '; '.join(f"**{available_values[v]}**" for v in dim_query.values)
            elif isinstance(dimension, DateTimeDimension):
                dataset_entry += format_datetime_dimension_query(
                    dim_query=dim_query, dimension=dimension
                )
            else:
                dataset_entry += '; '.join(f"**{v}**" for v in dim_query.values)

            if dim_query.is_default:
                dataset_entry += " (default)"

        if include_missing_dimensions:
            missing_dimensions = [
                d
                for d in dataset.dimensions()
                if d.entity_id not in query.dimensions_queries_dict
                or not query.dimensions_queries_dict[d.entity_id].values
            ]
            if missing_dimensions:
                dataset_entry += "\n* Missing dimensions:"
                for dimension in missing_dimensions:
                    dataset_entry += f"\n\t* _{dimension.name}_, ID: {dimension.entity_id}"
                    if availability and isinstance(dimension, CategoricalDimension):
                        values_names = {v.query_id: v.name for v in dimension.available_values}
                        available_values_query = availability.dimensions_queries_dict.get(
                            dimension.entity_id
                        )
                        if available_values_query is None:
                            logger.warning(
                                f'There are no available values for dimension "{dimension.name}". '
                                'Can\'t include samples values to data queries.'
                            )
                            continue
                        sample_values = available_values_query.values[:10]
                        dataset_entry += (
                            f", example values: {', '.join(values_names[v] for v in sample_values)}"
                        )

        if dataset.config.is_official:
            datasets_entries.insert(0, dataset_entry)
        else:
            datasets_entries.append(dataset_entry)

    return '\n\n'.join(datasets_entries)


async def format_availability_queries(
    auth_context: AuthContext,
    dataset_queries: DatasetAvailabilityQueriesType,
    datasets_dict: dict[str, DataSet],
    format_values_as_list: bool = False,
    header_level: int = 3,
    add_value_ids: bool = False,
    add_citation: bool = True,
) -> str:
    # NOTE: there is code duplication with "format_dataset_queries" function

    logger.info(f'formatting following availability queries (dataset_queries): {dataset_queries}')

    datasets_entries = []
    for dataset_id, query in sorted(dataset_queries.items(), key=lambda x: x[0]):
        dataset = datasets_dict[dataset_id]

        dataset_entry_lines = []

        prefix = '#' * header_level
        dataset_entry_lines.append(f'{prefix} {dataset.name}')
        dataset_entry_lines.append(f'* ID: {dataset.source_id}')

        formatter = DatasetFormatter(
            DatasetFormatterConfig.create_citation_only(), auth_context=auth_context
        )
        if add_citation:
            dataset_entry_lines.append(await formatter.format(dataset))

        dataset_entry_lines.append("* Query:")

        categorical_dimensions: dict[str, CategoricalDimension] = {
            d.entity_id: d for d in dataset.dimensions() if isinstance(d, CategoricalDimension)
        }
        datetime_dimensions: dict[str, DateTimeDimension] = {
            d.entity_id: d for d in dataset.dimensions() if isinstance(d, DateTimeDimension)
        }
        indicators: set[str] = {d.entity_id for d in dataset.indicator_dimensions()}

        for dimension_id, dim_query in query.dimensions_queries_dict.items():
            if cat_dimension := categorical_dimensions.get(dimension_id):
                dim_postfix = ''
                if dimension_id in indicators:
                    dim_postfix = " (Indicator)"
                available_dim_values = {v.query_id: v.name for v in cat_dimension.available_values}

                dimension_str = f'\t* _{cat_dimension.name}_{dim_postfix}'
                if add_value_ids is True:
                    values_str_gen = (
                        f"**[{v}] {available_dim_values[v]}**" for v in dim_query.values
                    )
                else:
                    values_str_gen = (f"**{available_dim_values[v]}**" for v in dim_query.values)

                if format_values_as_list:
                    values_concat = '\n\t\t* ' + '\n\t\t* '.join(values_str_gen)
                else:
                    values_concat = ' ' + '; '.join(values_str_gen)

                dataset_entry_lines.append(f"{dimension_str}:{values_concat}")
            elif datetime_dimension := datetime_dimensions.get(dimension_id):
                values_str = format_datetime_dimension_query(dim_query, datetime_dimension)
                dataset_entry_lines.append(f"\t* _{dimension_id}_: {values_str}")
            else:
                values_str = '; '.join(f"**{v}**" for v in dim_query.values)
                dataset_entry_lines.append(f"\t* _{dimension_id}_: {values_str}")

        concant = '\n'.join(dataset_entry_lines)
        datasets_entries.append(concant)

    return '\n\n'.join(datasets_entries)


def format_missing_dimensions(
    datasets_dict: dict[str, DataSet], dataset_to_missing_dimensions: dict[str, list[Dimension]]
) -> str:
    datasets_entries = []
    for dataset_id, missing_dimensions in dataset_to_missing_dimensions.items():
        dataset = datasets_dict[dataset_id]
        dataset_entry = f'* {dataset.name}, ID: {dataset.source_id}'
        dataset_entry += "\n* Missing dimensions:"
        for dimension in missing_dimensions:
            dataset_entry += f"\n\t* _{dimension.name}_, ID: {dimension.entity_id}"
        datasets_entries.append(dataset_entry)
    return '\n\n'.join(datasets_entries)
