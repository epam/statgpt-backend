from urllib.parse import quote

from mcp.types import EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

from statgpt.app.schemas.enums import DataQueryStatus
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
    - ``data_available``: queries collected from the executed responses + reproducible sdmx1 code.
    - ``invalid_time_period`` / ``not_executed``: the constructed (unexecuted) queries + code.
    - ``dataset_selection_required``: the candidate datasets to disambiguate.
    - ``missing_dimensions``: the required dimensions the user must still specify.
    - ``no_data``: just the status and the human-readable message.
    """
    state = artifact.state
    status = state.status

    sdmx_query_app = channel_config.sdmx_query_app
    tools = DataQueryToolsInfo(
        sdmx_proxy=sdmx_query_app.name if sdmx_query_app is not None else None
    )

    if status is DataQueryStatus.DATA_AVAILABLE:
        queries = [
            AppJsonQueryWithMetadata.from_common(response.json_query)
            for response in artifact.data_responses.values()
            if response.json_query is not None
        ]
        return DataQueryStructuredContent(
            status=status,
            queries=queries,
            python_code=generate_merged_python_code(queries),
            tools=tools,
        )

    if status in (DataQueryStatus.INVALID_TIME_PERIOD, DataQueryStatus.NOT_EXECUTED):
        queries = state.constructed_queries
        return DataQueryStructuredContent(
            status=status,
            queries=queries,
            python_code=generate_merged_python_code(queries) if queries else None,
            tools=tools,
        )

    if status is DataQueryStatus.DATASET_SELECTION_REQUIRED:
        return DataQueryStructuredContent(
            status=status, candidate_datasets=state.candidate_datasets, tools=tools
        )

    if status is DataQueryStatus.MISSING_DIMENSIONS:
        return DataQueryStructuredContent(
            status=status, missing_dimensions=state.missing_dimensions, tools=tools
        )

    # no_data (and any status without a dedicated payload)
    return DataQueryStructuredContent(status=status, message=message, tools=tools)
