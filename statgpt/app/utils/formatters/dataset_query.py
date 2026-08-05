from collections.abc import Callable

from pydantic import BaseModel, Field

from statgpt.app.services.chat_facade import VersionedDataSet
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import (
    CategoricalDimension,
    DataResponse,
    DataSet,
    DataSetAvailabilityQuery,
    DataSetQuery,
    DateTimeDimension,
    Query,
    QueryOperator,
)
from statgpt.common.data.base.enums import InvalidDataSetQueryReasonType
from statgpt.common.data.base.query import InvalidDataSetQueryReason
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus, LocaleEnum

from .base import BaseFormatter
from .dataset_base import DatasetFormatterConfig
from .dataset_simple import SimpleDatasetFormatter


class DatasetQueryFormatterConfig(BaseModel):
    locale: LocaleEnum = Field(LocaleEnum.EN, description="Locale for formatting")
    include_missing_dimensions: bool = Field(
        False, description="Whether to include missing dimensions in the output"
    )
    include_default_queries: bool = Field(
        False, description="Whether to include default queries in the output"
    )
    include_auto_selects: bool = Field(
        False, description="Whether to include auto-selected queries in the output"
    )
    include_is_official: bool = Field(
        False, description="Whether to indicate if the dataset is official"
    )
    include_query_uuid: bool = Field(
        False, description="Whether to include the query UUID in the output"
    )


class DatasetQueryFormatter(BaseFormatter):

    def __init__(
        self,
        config: DatasetQueryFormatterConfig,
        auth_context: AuthContext,
    ):
        super().__init__("dataset", config.locale)
        self._config = config
        self._auth_context = auth_context

    def _format_datetime_dimension_query(
        self, dim_query: Query, dimension: DateTimeDimension
    ) -> str:
        """Format datetime dimension query with localization."""
        if dim_query.operator == QueryOperator.BETWEEN:
            start = dim_query.values[0]
            end = dim_query.values[1]
            if not start and not end:
                return self._('no filter')
            return self._("from **{start}** to **{end}**").format(start=start, end=end)
        elif dim_query.operator == QueryOperator.GREATER_THAN_OR_EQUALS:
            return self._("from **{start}**").format(start=dim_query.values[0])
        elif dim_query.operator == QueryOperator.LESS_THAN_OR_EQUALS:
            return self._("until **{end}**").format(end=dim_query.values[0])
        elif dim_query.operator == QueryOperator.EQUALS:
            return self._("on **{date}**").format(date=dim_query.values[0])
        else:
            raise ValueError(
                f"Unsupported operator for DateTimeDimension: {dim_query.operator}. "
                f"dim_query: {dim_query}"
            )

    def _format_header(self, dataset) -> str:
        """Format dataset header with name and official label."""
        header = f'### {dataset.name}'
        if self._config.include_is_official and dataset.config.is_official:
            header += f' {dial_app_settings.official_dataset_label}'
        return header

    def _format_execution_result(self, data_response: DataResponse) -> list[str]:
        """Format execution result section."""
        lines = []

        if not data_response.visual_dataframe.empty:
            lines.append(
                f'✅ **{self._("Execution result")}**: {self._("Data received, contains {count} series.").format(count=data_response.get_display_series_count())}'
            )
        elif data_response.status.request_status == DataRequestStatus.FAILED:
            detail = data_response.status.reason or self._("The request to the data source failed.")
            lines.append(f'❌ **{self._("Execution result")}**: {detail}')
            lines.append("")
            lines.append(
                f'💡 **{self._("Advice")}:** '
                f'{self._("This looks like a temporary issue with the data source. You may want to retry the query, or try again shortly.")}'
            )
        elif data_response.status.parsing_status in (
            DataParsingStatus.FAILED,
            DataParsingStatus.PARTIALLY_FAILED,
        ):
            lines.append(
                f"❗ **{self._('Execution result')}**: {self._('The query was executed, but parsing the response failed.')}"
            )
            lines.append("")
            lines.append(
                f"💡 **{self._('Note')}:** {self._('While the data is not visible to you, user will be able to see it in UI.')} {self._('It is recommended to inform user that you were not able to see the data due to parsing issues.')}"
            )
        else:
            lines.append(
                f'❌ **{self._("Execution result")}**: {self._("A response was received, but it does not contain any data.")}'
            )
            lines.append("")
            lines.append(
                f'💡 **{self._("Advice")}:** {self._("Most likely, the query is generally correct, but there is no data for the specified time period.")} {self._("You may want to try selecting a different time period.")} {self._("Another option is to try to find relevant data in other datasets or using other tools.")}'
            )

        if data_response.url_query:
            lines.append("")
            lines.append(f'[🔍 {self._("View data in explorer")}]({data_response.url_query})')

        return lines

    async def _format_basic_info(
        self, query: DataSetQuery, dataset: DataSet, citation, data_response: DataResponse | None
    ) -> list[str]:
        """Format basic information section (ID, time period, citation)."""
        lines = [f'* {self._("ID")}: {dataset.source_id}']

        if self._config.include_query_uuid:
            lines.append(f'* {self._("Query UUID")}: {query.uuid}')

        if query.short_summary:
            lines.append(f'* {self._("Query summary")}: {query.short_summary}')

        try:
            if data_response and (time_period := data_response.time_period):
                start, end = time_period
                lines.append(f'* {self._("Factual time period")}: {start} {self._("to")} {end}')
        except Exception as e:
            logger.exception("Error formatting time period for dataset query", exc_info=e)

        if citation:
            formatted_citation = await SimpleDatasetFormatter(
                DatasetFormatterConfig.create_citation_only(locale=self._config.locale),
                auth_context=self._auth_context,
            ).format(dataset)
            lines.append(formatted_citation)

        return lines

    def _format_dimension_value(self, dimension, dim_query: Query) -> str:
        """Format dimension value based on dimension type."""
        if dim_query.is_all_selected:
            return "**\\***"

        if isinstance(dimension, CategoricalDimension):
            return '; '.join(f"**{dimension.name_by_query_id(v)}**" for v in dim_query.values)
        elif isinstance(dimension, DateTimeDimension):
            return self._format_datetime_dimension_query(dim_query=dim_query, dimension=dimension)
        else:
            return '; '.join(f"**{v}**" for v in dim_query.values)

    def _format_invalid_period_reason(self, reason: InvalidDataSetQueryReason) -> str:
        if reason.details['field'] == 'selected_end':
            template = self._('invalid_selected_end_time')
        elif reason.details['field'] == 'selected_start':
            template = self._('invalid_selected_start_time')
        else:
            raise ValueError(
                f"Unsupported reason details for invalid time period: {reason.details}"
            )

        return template.format(**reason.details)

    def _format_invalidity_reason(self, reason: InvalidDataSetQueryReason) -> str:
        """Format reason for invalid query."""
        reason_formatters: dict[
            InvalidDataSetQueryReasonType, Callable[[InvalidDataSetQueryReason], str]
        ] = {InvalidDataSetQueryReasonType.INVALID_TIME_PERIOD: self._format_invalid_period_reason}
        if reason.type not in reason_formatters:
            raise ValueError(f"Unsupported invalidity reason type: {reason.type}")

        reason_str = reason_formatters[reason.type](reason)
        return f"\t* {self._('Reason for invalid query')}: **{reason_str}**"

    def _format_query_dimensions(self, dataset, query: DataSetQuery) -> list[str]:
        """Format query dimensions section."""
        status = "" if query.is_valid else f" {self._('(invalid)')}"
        lines = [f'* {self._("Query")}{status}:']

        if (reason := query.invalidity_reason) is not None:
            lines.append(self._format_invalidity_reason(reason))

        indicators: set[str] = {d.entity_id for d in dataset.indicator_dimensions(non_virtual=True)}

        for dimension in dataset.dimensions():
            dim_query = next(
                (d for d in query.dimensions_queries if d.dimension_id == dimension.entity_id),
                None,
            )

            if not dim_query or dim_query.is_empty():
                continue

            if dim_query.is_default and not self._config.include_default_queries:
                continue

            if dim_query.is_all_selected and not self._config.include_auto_selects:
                continue

            # Build dimension line
            if dim_query.dimension_id in indicators:
                line = f"\t* _{dimension.name}_ ({self._('Indicator')}): "
            else:
                line = f"\t* _{dimension.name}_: "

            line += self._format_dimension_value(dimension, dim_query)

            if dim_query.is_default:
                if isinstance(dimension, DateTimeDimension):
                    default_note = self._(
                        'default time filter is used, as user did not specify any time filter'
                    )
                    line += f" ({default_note})"
                else:
                    line += f" ({self._('default')})"

            lines.append(line)

        return lines

    def _format_missing_dimensions(
        self, dataset, query: DataSetQuery, availability: DataSetAvailabilityQuery | None
    ) -> list[str]:
        """Format missing dimensions section."""
        if not self._config.include_missing_dimensions:
            return []

        missing_dimensions = [
            d
            for d in dataset.dimensions()
            if d.entity_id not in query.dimensions_queries_dict
            or query.dimensions_queries_dict[d.entity_id].is_empty()
        ]

        if not missing_dimensions:
            return []

        lines = [f'* {self._("Missing dimensions")}:']

        for dimension in missing_dimensions:
            line = f"\t* _{dimension.name}_, {self._('ID')}: {dimension.entity_id}"

            if availability and isinstance(dimension, CategoricalDimension):
                available_values_query = availability.dimensions_queries_dict.get(
                    dimension.entity_id
                )
                if available_values_query is None:
                    logger.warning(
                        f'There are no available values for dimension "{dimension.name}". '
                        'Can\'t include samples values to data queries.'
                    )
                else:
                    sample_values = available_values_query.values[:10]
                    sample_names = [
                        name
                        for v in sample_values
                        if (name := dimension.name_by_query_id(v)) is not None
                    ]
                    if sample_names:
                        line += f", {self._('example values')}: {', '.join(sample_names)}"

            lines.append(line)

        return lines

    async def format(
        self,
        query: DataSetQuery,
        versioned_dataset: VersionedDataSet,
        availability: DataSetAvailabilityQuery | None = None,
        data_response: DataResponse | None = None,
    ) -> str:
        """Format a single dataset query."""
        dataset = versioned_dataset.data
        citation = dataset.config.citation

        lines: list[str] = [self._format_header(dataset)]

        if data_response:
            lines.extend(self._format_execution_result(data_response))
        lines.extend(await self._format_basic_info(query, dataset, citation, data_response))
        lines.extend(self._format_query_dimensions(dataset, query))
        lines.extend(self._format_missing_dimensions(dataset, query, availability))

        return '\n'.join(lines)

    async def format_queries(
        self,
        dataset_queries: dict[str, DataSetQuery],
        datasets_dict: dict[str, VersionedDataSet],
        availability_queries: dict[str, DataSetAvailabilityQuery | None] | None = None,
        data_responses: dict[str, DataResponse | None] | None = None,
    ) -> str:
        """Format multiple dataset queries.

        Args:
            dataset_queries: Dictionary mapping dataset IDs to queries
            datasets_dict: Dictionary mapping dataset IDs to versioned datasets
            availability_queries: Optional dictionary mapping dataset IDs to availability queries
            data_responses: Optional dictionary mapping dataset IDs to data responses

        Returns:
            Formatted string with all queries, official datasets first
        """
        logger.info(f'formatting following dataset_queries: {dataset_queries}')

        datasets_entries: list[str] = []

        for dataset_id, query in sorted(dataset_queries.items(), key=lambda x: x[0]):
            versioned_dataset = datasets_dict[dataset_id]
            availability = availability_queries.get(dataset_id) if availability_queries else None
            data_response = data_responses.get(dataset_id) if data_responses else None

            dataset_entry = await self.format(
                query=query,
                versioned_dataset=versioned_dataset,
                availability=availability,
                data_response=data_response,
            )

            # Put official datasets first
            if versioned_dataset.data.config.is_official:
                datasets_entries.insert(0, dataset_entry)
            else:
                datasets_entries.append(dataset_entry)

        return '\n\n'.join(datasets_entries)
