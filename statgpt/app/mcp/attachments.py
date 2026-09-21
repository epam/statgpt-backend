import json
import logging
import numbers
from collections.abc import Iterable
from datetime import date, datetime
from hashlib import blake2s
from typing import Any
from urllib.parse import quote

import pandas as pd
from mcp.types import Annotations, EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

from statgpt.app.schemas.data_query_outcome import DataQueryStatus, MissingDimensionsInfo
from statgpt.app.schemas.mcp import (
    CandidateDatasetRecord,
    ClientQueryRecord,
    DataQueryClientMeta,
    DataQueryMcpAppMeta,
    DataQueryStructuredContent,
    DataQueryToolsInfo,
    FilterValue,
    McpAppQuery,
    MissingDimensionRecord,
    MissingDimensionsRecord,
    PeriodRange,
    QueryFilter,
    QueryRecord,
)
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.app.schemas.tool_artifact import DataQueryOutcome
from statgpt.app.services.python_code_generator import generate_merged_python_code
from statgpt.common.data.base import DataResponse
from statgpt.common.schemas import ChannelConfig, DataQueryMcpResources
from statgpt.common.schemas import DataQueryTool as DataQueryToolConfig
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
# A dimension filtered on more values than this is reported as a count plus the first few values:
# the full list belongs in the client payload, not in what the model reads.
_MAX_FILTER_VALUES = 10


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
) -> DataQueryStructuredContent:
    """Build the data query tool's MCP structured content from the pipeline outcome.

    Written for the calling model: the queries the pipeline produced, or what a follow-up query
    would need when it produced none. Per outcome:
    - ``data_available`` / ``executed_no_data`` / ``failed``: the executed queries.
    - ``not_executed``: the constructed queries, flagged as unexecuted.
    - ``dataset_selection_required``: the datasets to narrow the query to.
    - ``missing_dimensions``: the dimensions the query must still specify.
    - ``invalid_time_period`` / ``no_data``: nothing - the text content block explains the outcome.
    """
    status = outcome.state.status
    mcp_payload = outcome.mcp_payload

    if status in _EXECUTED_STATUSES:
        return DataQueryStructuredContent(queries=_executed_query_records(outcome))

    if status is DataQueryStatus.NOT_EXECUTED:
        return DataQueryStructuredContent(
            queries=[
                _query_record(
                    query_id=_query_id(query, None),
                    json_query=query,
                    executed=False,
                    # Nothing ran, so there is no response to read the names off.
                    dimension_names={},
                    value_names={},
                )
                for query in mcp_payload.constructed_queries
            ]
        )

    if status is DataQueryStatus.DATASET_SELECTION_REQUIRED:
        return DataQueryStructuredContent(
            candidate_datasets=[
                CandidateDatasetRecord(
                    id=candidate.id, name=candidate.name, is_official=candidate.is_official
                )
                for candidate in mcp_payload.candidate_datasets
            ]
        )

    if status is DataQueryStatus.MISSING_DIMENSIONS and mcp_payload.missing_dimensions is not None:
        return DataQueryStructuredContent(
            missing_dimensions=_missing_dimensions_record(mcp_payload.missing_dimensions)
        )

    return DataQueryStructuredContent()


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
        meta[meta_config.client_key] = _client_meta(
            outcome, config.mcp_resources, message
        ).model_dump(mode="json", by_alias=True, exclude_none=True)

    return meta or None


def _mcp_app_meta(
    outcome: DataQueryOutcome, channel_config: ChannelConfig, message: str | None
) -> DataQueryMcpAppMeta:
    """The MCP-App payload: the SDMX query model the widget renders and edits, with the pipeline
    status, the full value lists, the reproducible python code and the companion tool names."""
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
    outcome: DataQueryOutcome, resources_config: DataQueryMcpResources, message: str | None
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

    return DataQueryClientMeta(status=status, message=message, queries=queries)


def _executed_query_records(outcome: DataQueryOutcome) -> list[QueryRecord]:
    """One record per response that carries a query, in `data_responses` insertion order."""
    value_names = outcome.state.dimension_id_to_name
    records = []
    for dataset_id, response in outcome.data_responses.items():
        json_query = response.json_query
        query_id = _response_query_id(response)
        if json_query is None or query_id is None:
            continue
        records.append(
            _query_record(
                query_id=query_id,
                json_query=json_query,
                executed=True,
                dataset_name=response.dataset_name,
                dimension_names=response.component_names,
                value_names=value_names.get(dataset_id, {}),
                factual_period=_factual_period(response),
                series_count=_series_count(response),
            )
        )
    return records


def _query_record(
    *,
    query_id: str,
    json_query: JsonQueryWithMetadata,
    executed: bool,
    dimension_names: dict[str, str],
    value_names: dict[str, dict[str, str]],
    dataset_name: str | None = None,
    factual_period: PeriodRange | None = None,
    series_count: int | None = None,
) -> QueryRecord:
    filters, requested_period = _split_filters(json_query, dimension_names, value_names)
    return QueryRecord(
        query_id=query_id,
        dataset_urn=json_query.urn,
        dataset_name=dataset_name,
        executed=executed,
        filters=filters,
        requested_period=requested_period,
        factual_period=factual_period,
        series_count=series_count,
    )


def _split_filters(
    json_query: JsonQueryWithMetadata,
    dimension_names: dict[str, str],
    value_names: dict[str, dict[str, str]],
) -> tuple[list[QueryFilter], PeriodRange | None]:
    """Split the query's components into the dimension filters and the time period.

    The time period is reported once, as `requestedPeriod`, instead of as another filter: it is
    the one component whose values are dates rather than codes.
    """
    time_dimension = json_query.metadata.time_period_dimension
    filters: list[QueryFilter] = []
    requested_period: PeriodRange | None = None

    for component in json_query.filters:
        if component.component_code == time_dimension:
            requested_period = _period_range(component)
            continue
        values, total_values = _filter_values(
            component, value_names.get(component.component_code, {})
        )
        filters.append(
            QueryFilter(
                dimension_id=component.component_code,
                dimension_name=dimension_names.get(component.component_code),
                operator=component.operator,
                total_values=total_values,
                values=values,
            )
        )

    return filters, requested_period


def _filter_values(
    component: JsonComponentQuery, names: dict[str, str]
) -> tuple[list[FilterValue], int | None]:
    """The component's values as records, truncated to `_MAX_FILTER_VALUES`.

    The total is reported only when the list is truncated, so it does not restate `len(values)`.
    """
    values = component.values
    total_values = len(values) if len(values) > _MAX_FILTER_VALUES else None
    records = [
        FilterValue(id=value, name=names.get(value)) for value in values[:_MAX_FILTER_VALUES]
    ]
    return records, total_values


def _period_range(component: JsonComponentQuery) -> PeriodRange | None:
    """The time period the component selects, as a start/end pair."""
    values = component.values
    if not values:
        return None
    match component.operator:
        case JsonQueryOperator.BETWEEN:
            end = values[1] if len(values) > 1 else None
            return PeriodRange(start_period=values[0], end_period=end)
        case JsonQueryOperator.GE | JsonQueryOperator.GT:
            return PeriodRange(start_period=values[0])
        case JsonQueryOperator.LE | JsonQueryOperator.LT:
            return PeriodRange(end_period=values[0])
        case _:
            return PeriodRange(start_period=values[0], end_period=values[-1])


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
                    for value in dimension.available_values[:_MAX_FILTER_VALUES]
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
