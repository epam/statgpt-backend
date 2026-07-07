from pydantic import ConfigDict, Field

from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.common.schemas.base import BaseYamlModel


class DataQueryToolsInfo(BaseYamlModel):
    """Names of companion MCP tools the caller can use to act on the queries."""

    model_config = ConfigDict(serialize_by_alias=True)

    sdmx_proxy: str | None = Field(
        default=None,
        description="Name of the SDMX-proxy MCP tool, if configured on the channel.",
    )


class DataQueryStructuredContent(BaseYamlModel):
    """MCP structured content for the data query tool.

    Serialized with camelCase aliases to match the DIAL attachment shape.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    queries: list[AppJsonQueryWithMetadata] = Field(
        description="The queries used to fetch the data, one per dataset."
    )
    python_code: str = Field(
        description="A self-contained sdmx1 snippet that reproduces the queries."
    )
    tools: DataQueryToolsInfo = Field(description="Companion MCP tools for these queries.")
    version: int = Field(default=1, description="Schema version of this structured content.")


class SdmxProxyStructuredContent(BaseYamlModel):
    """MCP structured content for the SDMX-proxy passthrough tool.

    Surfaces the upstream HTTP metadata so the MCP-App can distinguish success from
    error responses and know the body's media type.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    status_code: int = Field(description="HTTP status code returned by the upstream request.")
    content_type: str | None = Field(
        default=None, description="Content type of the upstream response body, if known."
    )
