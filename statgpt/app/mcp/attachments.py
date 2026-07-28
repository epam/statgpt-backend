from urllib.parse import quote

from mcp.types import EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

from statgpt.app.schemas.data_query_outcome import DataQueryStatus
from statgpt.app.schemas.mcp import DataQueryStructuredContent, DataQueryToolsInfo
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.app.services.python_code_generator import generate_merged_python_code
from statgpt.common.schemas import ChannelConfig


def data_query_artifact_to_resources(
    artifact: DataQueryArtifact,
) -> list[EmbeddedResource]:
    """Serialize each DataResponse's csv_dataframe as an inline text/csv resource.

    Skips responses with empty csv_dataframe so MCP clients don't receive empty payloads.
    """
    resources: list[EmbeddedResource] = []
    for response in artifact.data_responses.values():
        df = response.csv_dataframe
        if df.empty:
            continue
        csv_text = df.to_csv(
            index=False,
            date_format="%Y-%m-%d",
            lineterminator="\n",
        )
        path = quote(response.resource_path, safe="")
        timestamp = response.created_at.strftime("%Y%m%dT%H%M%SZ")
        resources.append(
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri=AnyUrl(f"statgpt://data_query/{path}/{timestamp}.csv"),
                    mimeType="text/csv",
                    text=csv_text,
                ),
            )
        )
    return resources


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
