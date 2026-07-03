from urllib.parse import quote

from mcp.types import EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

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
    artifact: DataQueryArtifact, channel_config: ChannelConfig
) -> DataQueryStructuredContent | None:
    """Build the data query tool's MCP structured content from the artifact.

    Collects each DataResponse's json_query, generates a reproducible sdmx1 snippet for
    them, and records the companion SDMX-proxy tool name. Returns None when no response
    carries a query so the provider omits structuredContent.
    """
    queries: list[AppJsonQueryWithMetadata] = []
    for response in artifact.data_responses.values():
        query = response.json_query
        if query is None:
            continue
        queries.append(AppJsonQueryWithMetadata.from_common(query))
    if not queries:
        return None

    python_code = generate_merged_python_code(queries)
    sdmx_query_app = channel_config.sdmx_query_app
    tools = DataQueryToolsInfo(
        sdmx_proxy=sdmx_query_app.name if sdmx_query_app is not None else None
    )
    return DataQueryStructuredContent(queries=queries, python_code=python_code, tools=tools)
