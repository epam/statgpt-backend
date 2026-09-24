import json
import logging
import numbers
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import blake2s
from typing import Any, Self
from urllib.parse import quote

import pandas as pd
from mcp.types import Annotations, EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

from statgpt.app.schemas.data_query_outcome import (
    DataQueryStatus,
    InvalidPeriodInfo,
    MissingDimensionsInfo,
    QueryDetails,
)
from statgpt.app.schemas.mcp import (
    CandidateDatasetRecord,
    ClientQueryRecord,
    DataQueryClientMeta,
    DataQueryMcpAppMeta,
    DataQueryStructuredContent,
    DataQueryToolsInfo,
    ExecutionResult,
    FilterValue,
    InvalidityReason,
    InvalidPeriodRecord,
    McpAppQuery,
    MissingDimensionRecord,
    MissingDimensionsRecord,
    PeriodRange,
    QueryExecution,
    QueryFilter,
    QueryInvalidity,
    QueryRecord,
    RequestedPeriod,
)
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.app.schemas.tool_artifact import DataQueryOutcome
from statgpt.app.services.python_code_generator import generate_merged_python_code
from statgpt.common.data.base import DataResponse
from statgpt.common.schemas import (
    ChannelConfig,
    DataQueryMcpResources,
    DataQueryMcpStructuredContent,
)
from statgpt.common.schemas import DataQueryTool as DataQueryToolConfig
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus, ExplorerLinkPolicy
from statgpt.common.schemas.query import (
    JsonComponentQuery,
    JsonQuery,
    JsonQueryOperator,
    JsonQueryWithMetadata,
)

_log = logging.getLogger(__name__)

_DATE_FORMAT = "%Y-%m-%d"
_NAME_SUFFIX = "_Name"
_MISSING = "NA"
# Alignment markers at their minimum width. Padding every cell to the widest one in its column,
# as `DataFrame.to_markdown` does, inflates the payload without changing how it renders.
_LEFT_RULE = ":---"
_RIGHT_RULE = "---:"
# Statuses whose queries reached the data source. Their outcome differs, but all three know what
# was asked, so a caller can show the query and re-run it.
_EXECUTED_STATUSES = frozenset(
    {DataQueryStatus.DATA_AVAILABLE, DataQueryStatus.EXECUTED_NO_DATA, DataQueryStatus.FAILED}
)
_QUERY_ID_PREFIX = "dq_"
_QUERY_ID_DIGEST_SIZE = 5
# How many of a dimension's available values are sampled when reporting a missing dimension. The
# source there is the codelist, which can hold thousands of values.
_MAX_SAMPLE_VALUES = 10
_PARTIALLY_PARSED_WIDGET_ADVICE = (
    "The missing data is still visible to the user in the widget. Tell the user that you could"
    " only see part of the data."
)
_PARTIALLY_PARSED_ADVICE = (
    "Tell the user that the returned data is incomplete, since part of it could not be parsed."
)
_PARSING_FAILED_WIDGET_ADVICE = (
    "While the data is not visible to you, the user will be able to see it in the widget. Tell"
    " the user that you were not able to see the data due to parsing issues."
)
_PARSING_FAILED_ADVICE = (
    "The data could not be read. Tell the user, and retry the query or look for the data in"
    " another dataset."
)
_REQUEST_FAILED_ADVICE = (
    "This looks like a temporary issue with the data source. You may want to retry the query, or"
    " try again shortly."
)
_NO_DATA_ADVICE = (
    "Most likely, the query is generally correct, but there is no data for the specified time"
    " period. You may want to try selecting a different time period. Another option is to try to"
    " find relevant data in other datasets or using other tools."
)


@dataclass(frozen=True)
class _Reporting:
    """What the tool config lets the structured content report."""

    fields: DataQueryMcpStructuredContent
    explorer_link: ExplorerLinkPolicy
    widget_bound: bool

    @classmethod
    def from_tool_config(cls, tool_config: DataQueryToolConfig) -> Self:
        details = tool_config.details
        return cls(
            fields=details.mcp_structured_content,
            explorer_link=details.explorer_link.mcp,
            widget_bound=tool_config.mcp_app_resource_uri is not None,
        )

    def official_mark(self, value: bool) -> bool | None:
        """The official mark, when the channel reports it."""
        return value if self.fields.is_official else None


def data_query_outcome_to_resources(
    outcome: DataQueryOutcome,
    config: DataQueryMcpResources,
) -> list[EmbeddedResource]:
    """Serialize each DataResponse into the inline resources enabled by `config`.

    Per response, in `data_responses` insertion order, emits a `text/csv` resource built from
    `csv_dataframe` (observation-level, machine-readable) and/or a `text/markdown` table built
    from `visual_dataframe` (time periods as columns, meant to be shown to the user).

    Skips empty responses so MCP clients don't receive empty payloads. `is_empty` is checked
    instead of the dataframes: it is cheaper, and `visual_dataframe` returns the raw
    MultiIndexed frame when there are no observations.
    """
    builders = {"csv": _csv_resource, "md": _markdown_resource}
    extensions = _enabled_extensions(config)
    return [
        builders[extension](response)
        for response in outcome.data_responses.values()
        if not response.is_empty
        for extension in extensions
    ]


def _enabled_extensions(config: DataQueryMcpResources) -> list[str]:
    """The resource extensions `config` enables, in the order they are emitted."""
    extensions = []
    if config.csv.enabled:
        extensions.append("csv")
    if config.markdown_table.enabled:
        extensions.append("md")
    return extensions


def _resource_uris(response: DataResponse, config: DataQueryMcpResources) -> list[str]:
    """The URIs of the resources this response contributes to the result, as `content` has them."""
    if response.is_empty:
        return []
    return [str(_resource_uri(response, extension)) for extension in _enabled_extensions(config)]


def _resource_uri(response: DataResponse, extension: str) -> AnyUrl:
    # The query id is the file stem, so a client can join a resource to the query that produced
    # it. A response without a query keeps the timestamp.
    stem = _response_query_id(response) or response.created_at.strftime("%Y%m%dT%H%M%SZ")
    path = quote(response.resource_path, safe="")
    return AnyUrl(f"statgpt://data_query/{path}/{stem}.{extension}")


def _query_id(query: JsonQuery, execution_date: date | None) -> str:
    """A short id for one query, derived from the query itself and the date it ran.

    Deterministic: the same query executed again on the same date gets the same id, while the same
    query a month later - which may well return different data - gets a new one. A query that was
    never executed has no date, so its id depends on the query alone.
    """
    canonical = json.dumps(
        {
            "urn": query.urn,
            "filters": sorted(
                [component.component_code, component.operator.value, sorted(component.values)]
                for component in query.filters
            ),
            "date": execution_date.isoformat() if execution_date is not None else None,
        },
        separators=(",", ":"),
    )
    digest = blake2s(canonical.encode(), digest_size=_QUERY_ID_DIGEST_SIZE).hexdigest()
    return f"{_QUERY_ID_PREFIX}{digest}"


def _response_query_id(response: DataResponse) -> str | None:
    """The id of the query this response answers, or `None` when it carries no query."""
    query = response.json_query
    return _query_id(query, response.created_at.date()) if query is not None else None


def _csv_resource(response: DataResponse) -> EmbeddedResource:
    csv_text = response.csv_dataframe.to_csv(
        index=False,
        date_format=_DATE_FORMAT,
        lineterminator="\n",
    )
    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=_resource_uri(response, "csv"),
            mimeType="text/csv",
            text=csv_text,
        ),
    )


def _markdown_resource(response: DataResponse) -> EmbeddedResource:
    text = f"### {response.dataset_name}\n\n{_markdown_table(response)}\n"
    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=_resource_uri(response, "md"),
            mimeType="text/markdown",
            text=text,
        ),
        # The table is written for the person, not the model. Clients are free to ignore the
        # annotation (rendering of embedded resources is up to the client), which is why the
        # `data_query_executed_mcp_only` message also asks the model to reproduce it verbatim.
        annotations=Annotations(audience=["user"]),
    )


def _markdown_table(response: DataResponse) -> str:
    """Render the response as a compact Markdown table.

    Written out by hand rather than through `DataFrame.to_markdown`, which delegates to
    tabulate: tabulate pads every cell to the widest value in its column, and no combination
    of its options gets rid of that. `stralign=None` un-pads the data rows but still emits a
    full-width delimiter row and loses the alignment markers, and `colalign` re-pads. The
    padding is dead weight in a payload sent over the wire on every call and changes nothing
    about how the table renders. Doing it here also lets us escape pipes inside values, which
    tabulate does not, and keep the exact numeric text the CSV payload carries.
    """
    df = _collapse_coded_columns(response.visual_dataframe.copy())
    df = _rename_to_display_names(df, response.component_names)
    # Alignment is decided before stringifying, while the numeric columns are still numeric.
    rules = [
        _RIGHT_RULE if _holds_numbers(df.iloc[:, i]) else _LEFT_RULE for i in range(df.shape[1])
    ]
    lines = [
        _markdown_row(_format_cell(col) for col in df.columns),
        f"|{'|'.join(rules)}|",
    ]
    lines.extend(
        _markdown_row(_format_cell(value) for value in row)
        for row in df.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _markdown_row(cells: Iterable[str]) -> str:
    return f"| {' | '.join(cells)} |"


def _collapse_coded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop each coded `{col}` column that has a `{col}_Name` twin, keeping the names.

    `_enrich_df_with_names` emits both, and the codes are noise for a human reader. Columns
    without a twin — the unstacked time-period columns among them — are left untouched.
    """
    labels = set(df.columns)
    coded = {
        col.removesuffix(_NAME_SUFFIX)
        for col in df.columns
        if isinstance(col, str)
        and col.endswith(_NAME_SUFFIX)
        and col.removesuffix(_NAME_SUFFIX) in labels
    }
    if not coded:
        return df
    df = df.drop(columns=list(coded))
    return df.rename(columns={f"{base}{_NAME_SUFFIX}": base for base in coded})


def _rename_to_display_names(df: pd.DataFrame, component_names: dict[str, str]) -> pd.DataFrame:
    """Head each component column with its display name instead of its coded entity id.

    The id is kept whenever the display name is blank or would collide with another label in
    the frame (another component's name, or a time-period column), since a duplicated header
    would make the table ambiguous.
    """
    taken = set(df.columns)
    renames: dict[str, str] = {}
    for col in df.columns:
        if not isinstance(col, str):
            continue
        name = component_names.get(col)
        if not name or name in taken:
            continue
        renames[col] = name
        taken.add(name)
    return df.rename(columns=renames) if renames else df


def _holds_numbers(values: pd.Series) -> bool:
    """Whether every value in the column is a number, ignoring missing ones."""
    present = values.dropna()
    if present.empty:
        return False
    return all(
        isinstance(value, numbers.Number) and not isinstance(value, bool) for value in present
    )


def _format_cell(value: Any) -> str:
    """Render one value as text, fit for a table cell.

    `str` on a number is its shortest round-trip representation — the same one `to_csv` writes
    — so the table keeps full precision. A float that holds a whole number loses its `.0`,
    which `int` does exactly for any float, so `123.0` reads as `123`. Only actual numbers are
    reworked: a value that arrives as text is left alone, since a `.0` there may well be part
    of a code. Pipes and line breaks are neutralized: either would end the cell early and
    shift every value after it into the wrong column.
    """
    if not isinstance(value, (list, tuple, set, dict, pd.Series)) and pd.isna(value):
        return _MISSING
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.strftime(_DATE_FORMAT)
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        number = float(value)
        if number.is_integer():
            return str(int(number))
    return str(value).replace("|", "\\|").replace("\r\n", " ").replace("\n", " ")


def data_query_outcome_to_structured_content(
    outcome: DataQueryOutcome,
    tool_config: DataQueryToolConfig,
) -> DataQueryStructuredContent:
    """Build the data query tool's MCP structured content from the pipeline outcome.

    Written for the calling model: the outcome, with the queries the pipeline produced, or what a
    follow-up query would need when it produced none. Per outcome:
    - ``data_available`` / ``executed_no_data`` / ``failed``: the executed queries.
    - ``not_executed``: the constructed queries, flagged as unexecuted.
    - ``invalid_time_period``: the constructed queries, each with why it cannot run.
    - ``dataset_selection_required``: the datasets to narrow the query to, with their queries.
    - ``missing_dimensions``: the dimensions the query must still specify.
    - ``no_data``: the message alone.
    """
    reporting = _Reporting.from_tool_config(tool_config)
    status = outcome.state.status
    mcp_payload = outcome.mcp_payload
    queries: list[QueryRecord] = []
    candidate_datasets: list[CandidateDatasetRecord] = []
    missing_dimensions: MissingDimensionsRecord | None = None
    executed_at: str | None = None

    if status in _EXECUTED_STATUSES:
        queries = _executed_query_records(outcome, reporting)
        if reporting.fields.executed_at:
            executed_at = mcp_payload.executed_at
    elif status is DataQueryStatus.NOT_EXECUTED:
        queries = [
            _query_record(
                query_id=_query_id(query, None),
                json_query=query,
                executed=False,
                reporting=reporting,
                # Nothing ran, so there is no response to read the names off.
                dimension_names={},
                value_names={},
            )
            for query in mcp_payload.constructed_queries
        ]
    elif status is DataQueryStatus.INVALID_TIME_PERIOD:
        value_names = outcome.state.dimension_id_to_name
        queries = [
            record
            for dataset_id, details in mcp_payload.query_details.items()
            if (
                record := _constructed_query_record(details, value_names.get(dataset_id), reporting)
            )
        ]
    elif status is DataQueryStatus.DATASET_SELECTION_REQUIRED:
        candidate_datasets = _candidate_dataset_records(outcome, reporting)
    elif (
        status is DataQueryStatus.MISSING_DIMENSIONS and mcp_payload.missing_dimensions is not None
    ):
        missing_dimensions = _missing_dimensions_record(mcp_payload.missing_dimensions)

    return DataQueryStructuredContent(
        status=status,
        message=mcp_payload.message,
        executed_at=executed_at,
        queries=queries,
        candidate_datasets=candidate_datasets,
        missing_dimensions=missing_dimensions,
    )


def data_query_outcome_to_meta(
    outcome: DataQueryOutcome,
    channel_config: ChannelConfig,
    tool_config: DataQueryToolConfig,
    message: str | None = None,
) -> dict[str, Any] | None:
    """Build the result's ``_meta``: one namespaced payload per audience that has a reader.

    The MCP-App payload is carried when the tool binds a widget resource - without one there is
    nothing to render it - and the client payload when the config enables it. Returns ``None``
    when neither applies, so the result carries no ``_meta`` at all.
    """
    config = tool_config.details
    meta_config = config.mcp_meta
    meta: dict[str, Any] = {}

    if tool_config.mcp_app_resource_uri is not None:
        # Nulls are kept here: the widget's payload keeps the same shape whatever the outcome.
        meta[meta_config.mcp_app_key] = _mcp_app_meta(outcome, channel_config, message).model_dump(
            mode="json", by_alias=True
        )
    if meta_config.client.enabled:
        meta[meta_config.client_key] = _client_meta(outcome, config.mcp_resources).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )

    return meta or None


def _mcp_app_meta(
    outcome: DataQueryOutcome, channel_config: ChannelConfig, message: str | None
) -> DataQueryMcpAppMeta:
    """The MCP-App payload: the SDMX query model the widget renders and edits, with the pipeline
    status, the reproducible python code and the companion tool names."""
    status = outcome.state.status
    mcp_payload = outcome.mcp_payload
    sdmx_query_app = channel_config.sdmx_query_app
    tools = DataQueryToolsInfo(
        sdmx_proxy=sdmx_query_app.name if sdmx_query_app is not None else None
    )

    queries: list[McpAppQuery] = []
    if status in _EXECUTED_STATUSES:
        queries = [
            McpAppQuery.from_app_query(
                AppJsonQueryWithMetadata.from_common(response.json_query), query_id
            )
            for response in outcome.data_responses.values()
            if response.json_query is not None
            and (query_id := _response_query_id(response)) is not None
        ]
    elif status is DataQueryStatus.NOT_EXECUTED:
        queries = [
            McpAppQuery.from_app_query(query, _query_id(query, None))
            for query in mcp_payload.constructed_queries
        ]

    return DataQueryMcpAppMeta(
        status=status,
        message=message,
        queries=queries,
        candidate_datasets=(
            mcp_payload.candidate_datasets
            if status is DataQueryStatus.DATASET_SELECTION_REQUIRED
            else []
        ),
        missing_dimensions=(
            mcp_payload.missing_dimensions if status is DataQueryStatus.MISSING_DIMENSIONS else None
        ),
        # `failed` is also the default status, in which case there are no queries to report - don't
        # emit a snippet that would only contain the imports header.
        python_code=generate_merged_python_code(queries) if queries else None,
        tools=tools,
    )


def _client_meta(
    outcome: DataQueryOutcome, resources_config: DataQueryMcpResources
) -> DataQueryClientMeta:
    """The client payload: the pipeline status plus, per query, where to look at it and which of
    the result's resources belong to it."""
    status = outcome.state.status
    queries: list[ClientQueryRecord] = []

    if status in _EXECUTED_STATUSES:
        for response in outcome.data_responses.values():
            json_query = response.json_query
            query_id = _response_query_id(response)
            if json_query is None or query_id is None:
                continue
            queries.append(
                ClientQueryRecord(
                    query_id=query_id,
                    urn=json_query.urn,
                    dataset_name=response.dataset_name,
                    data_explorer_url=response.url_query,
                    dataset_url=json_query.metadata.dataset_url,
                    resource_uris=_resource_uris(response, resources_config),
                    series_count=_series_count(response),
                )
            )
    elif status is DataQueryStatus.NOT_EXECUTED:
        queries = [
            ClientQueryRecord(
                query_id=_query_id(query, None),
                urn=query.urn,
                dataset_url=query.metadata.dataset_url,
            )
            for query in outcome.mcp_payload.constructed_queries
        ]

    return DataQueryClientMeta(status=status, queries=queries)


def _executed_query_records(outcome: DataQueryOutcome, reporting: _Reporting) -> list[QueryRecord]:
    """One record per response that carries a query, in `data_responses` insertion order."""
    value_names = outcome.state.dimension_id_to_name
    records = []
    for dataset_id, response in outcome.data_responses.items():
        json_query = response.json_query
        query_id = _response_query_id(response)
        if json_query is None or query_id is None:
            continue
        details = outcome.mcp_payload.query_details.get(dataset_id)
        dimension_names = details.dimension_names if details is not None else {}
        records.append(
            _query_record(
                query_id=query_id,
                json_query=json_query,
                executed=True,
                reporting=reporting,
                details=details,
                dataset_name=response.dataset_name,
                dimension_names={**dimension_names, **response.component_names},
                value_names=value_names.get(dataset_id, {}),
                factual_period=_factual_period(response),
                series_count=_series_count(response),
                execution=_execution(response, reporting.widget_bound),
                data_explorer_url=(
                    response.url_query
                    if reporting.explorer_link.includes_link(not response.is_empty)
                    else None
                ),
            )
        )
    return records


def _constructed_query_record(
    details: QueryDetails, value_names: dict[str, dict[str, str]] | None, reporting: _Reporting
) -> QueryRecord | None:
    """The record of a query that was constructed but did not run, or `None` without a query."""
    if details.json_query is None:
        return None
    return _query_record(
        query_id=_query_id(details.json_query, None),
        json_query=details.json_query,
        executed=False,
        reporting=reporting,
        details=details,
        dataset_name=details.dataset_name,
        dimension_names=details.dimension_names,
        value_names=value_names or {},
    )


def _candidate_dataset_records(
    outcome: DataQueryOutcome, reporting: _Reporting
) -> list[CandidateDatasetRecord]:
    """The datasets to narrow the query to, each with the query that would run against it."""
    value_names = outcome.state.dimension_id_to_name
    queries = {
        details.dataset_urn: record
        for dataset_id, details in outcome.mcp_payload.query_details.items()
        if (record := _constructed_query_record(details, value_names.get(dataset_id), reporting))
    }
    return [
        CandidateDatasetRecord(
            id=candidate.id,
            name=candidate.name,
            is_official=reporting.official_mark(candidate.is_official),
            query=queries.get(candidate.id),
        )
        for candidate in outcome.mcp_payload.candidate_datasets
    ]


def _query_record(
    *,
    query_id: str,
    json_query: JsonQueryWithMetadata,
    executed: bool,
    reporting: _Reporting,
    dimension_names: dict[str, str],
    value_names: dict[str, dict[str, str]],
    details: QueryDetails | None = None,
    dataset_name: str | None = None,
    factual_period: PeriodRange | None = None,
    series_count: int | None = None,
    execution: QueryExecution | None = None,
    data_explorer_url: str | None = None,
) -> QueryRecord:
    filters, requested_period = _split_filters(json_query, dimension_names, value_names, details)
    return QueryRecord(
        query_id=query_id,
        dataset_urn=json_query.urn,
        dataset_name=dataset_name,
        is_official=reporting.official_mark(details.is_official) if details is not None else None,
        provider=details.provider if details is not None and reporting.fields.provider else None,
        last_updated=details.last_updated if details is not None else None,
        dataset_url=json_query.metadata.dataset_url if reporting.fields.dataset_url else None,
        query_summary=details.summary if details is not None else None,
        executed=executed,
        filters=filters,
        requested_period=requested_period,
        invalidity=_invalidity(details) if details is not None else None,
        factual_period=factual_period,
        series_count=series_count,
        execution=execution,
        data_explorer_url=data_explorer_url,
    )


def _split_filters(
    json_query: JsonQueryWithMetadata,
    dimension_names: dict[str, str],
    value_names: dict[str, dict[str, str]],
    details: QueryDetails | None,
) -> tuple[list[QueryFilter], RequestedPeriod | None]:
    """Split the query's components into the dimension filters and the time period.

    The time period is reported once, as `requestedPeriod`, instead of as another filter: it is
    the one component whose values are dates rather than codes.
    """
    metadata = json_query.metadata
    indicator_ids = set(metadata.indicator_dimensions or [])
    default_ids = set(details.default_dimension_ids) if details is not None else set()
    filters: list[QueryFilter] = []
    requested_period: RequestedPeriod | None = None

    for component in json_query.filters:
        code = component.component_code
        if code == metadata.time_period_dimension:
            requested_period = _period_range(
                component, is_default=details is not None and details.default_time_period
            )
            continue
        filters.append(
            QueryFilter(
                dimension_id=code,
                dimension_name=dimension_names.get(code),
                operator=component.operator,
                values=_filter_values(component, value_names.get(code, {})),
                is_indicator=code in indicator_ids,
                is_default=code in default_ids,
            )
        )

    return filters, requested_period


def _filter_values(component: JsonComponentQuery, names: dict[str, str]) -> list[FilterValue]:
    """Every value the component filters on, as records.

    The list is not truncated: it is what the query asked for, and the model cannot report on a
    query whose filter it only sees part of.
    """
    return [FilterValue(id=value, name=names.get(value)) for value in component.values]


def _period_range(component: JsonComponentQuery, is_default: bool) -> RequestedPeriod | None:
    """The time period the component selects, as a start/end pair."""
    values = component.values
    if not values:
        return None
    match component.operator:
        case JsonQueryOperator.BETWEEN:
            end = values[1] if len(values) > 1 else None
            return RequestedPeriod(start_period=values[0], end_period=end, is_default=is_default)
        case JsonQueryOperator.GE | JsonQueryOperator.GT:
            return RequestedPeriod(start_period=values[0], is_default=is_default)
        case JsonQueryOperator.LE | JsonQueryOperator.LT:
            return RequestedPeriod(end_period=values[0], is_default=is_default)
        case _:
            return RequestedPeriod(
                start_period=values[0], end_period=values[-1], is_default=is_default
            )


def _invalidity(details: QueryDetails) -> QueryInvalidity | None:
    """Why the query cannot run, or `None` for a valid query."""
    if (invalid_period := details.invalid_period) is not None:
        return QueryInvalidity(
            reason=InvalidityReason.INVALID_TIME_PERIOD,
            explanation=_invalid_period_explanation(invalid_period),
            rejected_period=InvalidPeriodRecord(
                rejected_bound=invalid_period.rejected_bound,
                requested_value=invalid_period.requested_value,
                available_period=PeriodRange(
                    start_period=invalid_period.available_start,
                    end_period=invalid_period.available_end,
                ),
            ),
        )
    if details.missing_dimensions is not None:
        dimensions = _missing_dimensions_record(details.missing_dimensions).dimensions
        names = ", ".join(dimension.name for dimension in dimensions)
        return QueryInvalidity(
            reason=InvalidityReason.MISSING_DIMENSIONS,
            explanation=(
                f"The query does not specify the required dimensions: {names}."
                if names
                else "The query does not specify every required dimension."
            ),
            missing_dimensions=dimensions,
        )
    return None


def _invalid_period_explanation(invalid_period: InvalidPeriodInfo) -> str:
    value = invalid_period.requested_value
    if invalid_period.rejected_bound == "startPeriod":
        return (
            f"The requested start period {value} is after the last period the dataset has data"
            f" for ({invalid_period.available_end})."
        )
    return (
        f"The requested end period {value} is before the first period the dataset has data for"
        f" ({invalid_period.available_start})."
    )


def _execution(response: DataResponse, widget_bound: bool) -> QueryExecution:
    """How the execution of the response's query went, with what the model should do about it.

    The widget fetches the data itself, so the user may see data the model could not read.
    """
    status = response.status

    if not response.is_empty:
        if status.parsing_status is DataParsingStatus.PARTIALLY_FAILED:
            return QueryExecution(
                result=ExecutionResult.PARTIALLY_PARSED,
                reason="Some of the data could not be parsed, so the returned data is incomplete.",
                advice=(
                    _PARTIALLY_PARSED_WIDGET_ADVICE if widget_bound else _PARTIALLY_PARSED_ADVICE
                ),
            )
        return QueryExecution(result=ExecutionResult.DATA_RECEIVED)

    if status.request_status is DataRequestStatus.FAILED:
        return QueryExecution(
            result=ExecutionResult.REQUEST_FAILED,
            reason=status.reason or "The request to the data source failed.",
            advice=_REQUEST_FAILED_ADVICE,
        )
    if status.parsing_status in (DataParsingStatus.FAILED, DataParsingStatus.PARTIALLY_FAILED):
        return QueryExecution(
            result=ExecutionResult.PARSING_FAILED,
            reason="The query was executed, but parsing the response failed.",
            advice=_PARSING_FAILED_WIDGET_ADVICE if widget_bound else _PARSING_FAILED_ADVICE,
        )
    return QueryExecution(
        result=ExecutionResult.NO_DATA,
        reason="A response was received, but it does not contain any data.",
        advice=_NO_DATA_ADVICE,
    )


def _missing_dimensions_record(missing: MissingDimensionsInfo) -> MissingDimensionsRecord:
    """The missing dimensions with a bounded sample of each one's values."""
    return MissingDimensionsRecord(
        dataset_urn=missing.dataset_urn,
        dimensions=[
            MissingDimensionRecord(
                dimension_id=dimension.dimension_id,
                name=dimension.name,
                total_values=len(dimension.available_values),
                sample_values=[
                    FilterValue(id=value.id, name=value.name)
                    for value in dimension.available_values[:_MAX_SAMPLE_VALUES]
                ],
            )
            for dimension in missing.dimensions
        ],
    )


def _factual_period(response: DataResponse) -> PeriodRange | None:
    """The period the returned data actually covers, or `None` when it cannot be determined."""
    try:
        period = response.time_period
    except Exception:
        _log.exception("Failed to read the factual time period of %s", response.dataset_name)
        return None
    if not period:
        return None
    start, end = period
    return PeriodRange(start_period=start, end_period=end)


def _series_count(response: DataResponse) -> int | None:
    """How many series the response carries, or `None` when it carries no data."""
    if response.is_empty:
        return None
    try:
        return response.get_display_series_count()
    except Exception:
        _log.exception("Failed to count the series of %s", response.dataset_name)
        return None
