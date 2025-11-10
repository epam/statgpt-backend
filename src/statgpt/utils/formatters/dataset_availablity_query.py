from aidial_sdk.chat_completion import Stage
from pydantic import BaseModel, Field

from common.auth.auth_context import AuthContext
from common.data.base import (
    CategoricalDimension,
    DataSetAvailabilityQuery,
    DateTimeDimension,
    Query,
)
from common.schemas.enums import LocaleEnum
from statgpt.schemas.query_builder import DatasetAvailabilityQueriesType
from statgpt.services.chat_facade import VersionedDataSet

from .base import BaseFormatter
from .dataset_base import DatasetFormatterConfig
from .dataset_query import DatasetQueryFormatter, DatasetQueryFormatterConfig
from .dataset_simple import SimpleDatasetFormatter


class DatasetAvailabilityQueryFormatterConfig(BaseModel):
    locale: LocaleEnum = Field(LocaleEnum.EN, description="Locale for formatting")
    format_values_as_list: bool = Field(
        False, description="Format dimension values as bulleted list vs inline"
    )
    header_level: int = Field(3, description="Markdown header level (number of #)")
    add_value_ids: bool = Field(False, description="Include value IDs in brackets")
    add_citation: bool = Field(True, description="Include dataset citation")


class DatasetAvailabilityQueryFormatter(BaseFormatter):

    def __init__(
        self,
        config: DatasetAvailabilityQueryFormatterConfig,
        auth_context: AuthContext,
    ):
        super().__init__("dataset", config.locale)
        self._config = config
        self._auth_context = auth_context

    def _format_header(self, dataset, header_level: int) -> str:
        """Format dataset header with configurable level."""
        prefix = '#' * header_level
        return f'{prefix} {dataset.name}'

    def _format_categorical_dimension_values(
        self, dimension: CategoricalDimension, values: list[str]
    ) -> str:
        """Format categorical dimension values based on config."""
        if self._config.add_value_ids:
            values_gen = (f"**[{v}] {dimension.name_by_query_id(v)}**" for v in values)
        else:
            values_gen = (f"**{dimension.name_by_query_id(v)}**" for v in values)

        if self._config.format_values_as_list:
            return '\n\t\t* ' + '\n\t\t* '.join(values_gen)
        else:
            return ' ' + '; '.join(values_gen)

    def _format_dimension_query(
        self,
        dimension_id: str,
        dim_query: Query,
        categorical_dimensions: dict[str, CategoricalDimension],
        datetime_dimensions: dict[str, DateTimeDimension],
        indicators: set[str],
    ) -> str:
        """Format a single dimension query."""
        if cat_dimension := categorical_dimensions.get(dimension_id):
            dim_postfix = ''
            if dimension_id in indicators:
                dim_postfix = f" ({self._('Indicator')})"

            dimension_str = f'\t* _{cat_dimension.name}_{dim_postfix}'
            values_concat = self._format_categorical_dimension_values(
                cat_dimension, dim_query.values
            )
            return f"{dimension_str}:{values_concat}"

        elif datetime_dimension := datetime_dimensions.get(dimension_id):
            formatter = DatasetQueryFormatter(
                DatasetQueryFormatterConfig(locale=self._config.locale),
                auth_context=self._auth_context,
            )
            values_str = formatter._format_datetime_dimension_query(dim_query, datetime_dimension)
            return f"\t* _{dimension_id}_: {values_str}"
        else:
            values_str = '; '.join(f"**{v}**" for v in dim_query.values)
            return f"\t* _{dimension_id}_: {values_str}"

    async def format(
        self,
        query: DataSetAvailabilityQuery,
        versioned_dataset: VersionedDataSet,
    ) -> str:
        """Format a single availability query."""
        dataset = versioned_dataset.data
        lines: list[str] = []

        # Header
        lines.append(self._format_header(dataset, self._config.header_level))

        # ID
        lines.append(f'* {self._("ID")}: {dataset.source_id}')

        # Citation
        if self._config.add_citation and dataset.config.citation:
            formatted_citation = await SimpleDatasetFormatter(
                DatasetFormatterConfig.create_citation_only(locale=self._config.locale),
                auth_context=self._auth_context,
            ).format(dataset)
            lines.append(formatted_citation)

        # Query header
        lines.append(f'* {self._("Query")}:')

        # Prepare dimension mappings
        dimensions = dataset.dimensions()
        categorical_dimensions: dict[str, CategoricalDimension] = {
            d.entity_id: d for d in dimensions if isinstance(d, CategoricalDimension)
        }
        datetime_dimensions: dict[str, DateTimeDimension] = {
            d.entity_id: d for d in dimensions if isinstance(d, DateTimeDimension)
        }
        indicators: set[str] = {d.entity_id for d in dataset.indicator_dimensions(non_virtual=True)}

        # Format dimensions
        for dimension_id, dim_query in query.dimensions_queries_dict.items():
            dimension_line = self._format_dimension_query(
                dimension_id,
                dim_query,
                categorical_dimensions,
                datetime_dimensions,
                indicators,
            )
            lines.append(dimension_line)

        return '\n'.join(lines)

    async def format_queries(
        self,
        dataset_queries: dict[str, DataSetAvailabilityQuery],
        datasets_dict: dict[str, VersionedDataSet],
    ) -> str:
        """Format multiple availability queries.

        Args:
            dataset_queries: Dictionary mapping dataset IDs to availability queries
            datasets_dict: Dictionary mapping dataset IDs to versioned datasets

        Returns:
            Formatted string with all queries
        """
        datasets_entries: list[str] = []

        for dataset_id, query in sorted(dataset_queries.items(), key=lambda x: x[0]):
            versioned_dataset = datasets_dict[dataset_id]
            dataset_entry = await self.format(
                query=query,
                versioned_dataset=versioned_dataset,
            )
            datasets_entries.append(dataset_entry)

        return '\n\n'.join(datasets_entries)

    @classmethod
    async def populate_queries_stage(
        cls,
        stage: Stage,
        queries: DatasetAvailabilityQueriesType,
        auth_context: AuthContext,
        datasets_dict: dict[str, VersionedDataSet],
    ) -> None:
        if not queries:
            stage.append_content("No queries")
            return

        query_formatter = cls(
            config=DatasetAvailabilityQueryFormatterConfig(
                locale=LocaleEnum.EN,
                format_values_as_list=True,
                add_value_ids=True,
                add_citation=False,
            ),
            auth_context=auth_context,
        )

        content = await query_formatter.format_queries(
            dataset_queries=queries,
            datasets_dict=datasets_dict,
        )
        stage.append_content(content)
