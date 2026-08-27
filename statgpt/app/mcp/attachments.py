import numbers
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any
from urllib.parse import quote

import pandas as pd
from mcp.types import Annotations, EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

from statgpt.app.schemas.data_query_outcome import DataQueryStatus
from statgpt.app.schemas.mcp import DataQueryStructuredContent, DataQueryToolsInfo
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.app.services.python_code_generator import generate_merged_python_code
from statgpt.common.data.base import DataResponse
from statgpt.common.schemas import ChannelConfig, DataQueryMcpResources

_DATE_FORMAT = "%Y-%m-%d"
_NAME_SUFFIX = "_Name"
_MISSING = "NA"
# Alignment markers at their minimum width. Padding every cell to the widest one in its column,
# as `DataFrame.to_markdown` does, inflates the payload without changing how it renders.
_LEFT_RULE = ":---"
_RIGHT_RULE = "---:"


def data_query_artifact_to_resources(
    artifact: DataQueryArtifact,
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
    resources: list[EmbeddedResource] = []
    for response in artifact.data_responses.values():
        if response.is_empty:
            continue
        if config.csv.enabled:
            resources.append(_csv_resource(response))
        if config.markdown_table.enabled:
            resources.append(_markdown_resource(response))
    return resources


def _resource_uri(response: DataResponse, extension: str) -> AnyUrl:
    path = quote(response.resource_path, safe="")
    timestamp = response.created_at.strftime("%Y%m%dT%H%M%SZ")
    return AnyUrl(f"statgpt://data_query/{path}/{timestamp}.{extension}")


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


def data_query_artifact_to_structured_content(
    artifact: DataQueryArtifact,
    channel_config: ChannelConfig,
    message: str | None = None,
) -> DataQueryStructuredContent:
    """Build the data query tool's MCP structured content from the artifact.

    Always returns a value carrying the pipeline ``status``. Per outcome:
    - ``data_available`` / ``executed_no_data`` / ``failed``: the executed queries + reproducible
      sdmx1 code. The queries are known in all three cases, so a client can show what was asked
      and re-run it.
    - ``not_executed``: the constructed (unexecuted) queries + code.
    - ``dataset_selection_required``: the candidate datasets to disambiguate.
    - ``missing_dimensions``: the required dimensions the user must still specify.
    - ``invalid_time_period`` / ``no_data``: just the status and the human-readable message.
    """
    state = artifact.state
    mcp_payload = artifact.mcp_payload
    status = state.status

    sdmx_query_app = channel_config.sdmx_query_app
    tools = DataQueryToolsInfo(
        sdmx_proxy=sdmx_query_app.name if sdmx_query_app is not None else None
    )

    if status in (
        DataQueryStatus.DATA_AVAILABLE,
        DataQueryStatus.EXECUTED_NO_DATA,
        DataQueryStatus.FAILED,
    ):
        executed_queries = [
            AppJsonQueryWithMetadata.from_common(response.json_query)
            for response in artifact.data_responses.values()
            if response.json_query is not None
        ]
        return DataQueryStructuredContent(
            status=status,
            queries=executed_queries,
            # `failed` is also the default status, in which case there are no responses to
            # report — don't emit a snippet that would only contain the imports header.
            python_code=(
                generate_merged_python_code(executed_queries) if executed_queries else None
            ),
            tools=tools,
        )

    if status is DataQueryStatus.NOT_EXECUTED:
        constructed_queries = mcp_payload.constructed_queries
        return DataQueryStructuredContent(
            status=status,
            queries=constructed_queries,
            python_code=(
                generate_merged_python_code(constructed_queries) if constructed_queries else None
            ),
            tools=tools,
        )

    if status is DataQueryStatus.DATASET_SELECTION_REQUIRED:
        return DataQueryStructuredContent(
            status=status,
            message=message,
            candidate_datasets=mcp_payload.candidate_datasets,
            tools=tools,
        )

    if status is DataQueryStatus.MISSING_DIMENSIONS:
        return DataQueryStructuredContent(
            status=status,
            message=message,
            missing_dimensions=mcp_payload.missing_dimensions,
            tools=tools,
        )

    # invalid_time_period, no_data, and any status without a dedicated payload. The constructed
    # queries are deliberately not reported for invalid_time_period: the rejected time period was
    # never applied to them, so they would describe a query the user did not ask for.
    return DataQueryStructuredContent(status=status, message=message, tools=tools)
