from urllib.parse import quote

from mcp.types import EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

from statgpt.app.schemas.tool_artifact import DataQueryArtifact


def data_query_artifact_to_resources(
    artifact: DataQueryArtifact,
) -> list[EmbeddedResource]:
    """Serialize each DataResponse's visual_dataframe as an inline text/csv resource.

    Skips responses with no visual_dataframe so MCP clients don't receive empty payloads.
    """
    resources: list[EmbeddedResource] = []
    for response in artifact.data_responses.values():
        df = getattr(response, "csv_dataframe", response.visual_dataframe)
        if df is None or df.empty:
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
