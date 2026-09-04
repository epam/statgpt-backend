from fastmcp.tools import ToolResult
from pydantic import PrivateAttr

from statgpt.app.chains.sdmx_query_app import SdmxQueryAppArgs, SdmxQueryAppProxy
from statgpt.app.schemas.mcp import SdmxProxyStructuredContent
from statgpt.common.schemas import SdmxQueryAppTool as SdmxQueryAppToolConfig
from statgpt.common.schemas import ToolTypes

from .base import StatGptMcpTool


class SdmxQueryAppMcpTool(
    StatGptMcpTool[SdmxQueryAppToolConfig, SdmxQueryAppArgs], tool_type=ToolTypes.SDMX_QUERY_APP
):
    """MCP-only passthrough: forwards a frontend-built SDMX request to the configured backend and
    returns the raw body as text, with the upstream HTTP metadata as structured content so the
    MCP-App can distinguish success from error responses and know the body's media type."""

    _proxy: SdmxQueryAppProxy = PrivateAttr()

    def __init__(
        self, tool_config: SdmxQueryAppToolConfig, channel_config, inputs, auth_context, **kwargs
    ):
        super().__init__(tool_config, channel_config, inputs, auth_context, **kwargs)
        self._proxy = SdmxQueryAppProxy(tool_config.details)

    @classmethod
    def get_args_schema(cls, tool_config: SdmxQueryAppToolConfig) -> type[SdmxQueryAppArgs]:
        return SdmxQueryAppArgs

    async def _execute(self, args: SdmxQueryAppArgs) -> ToolResult:
        response = await self._proxy.forward(
            path=args.path, method=args.method, body=args.body, accept=args.accept
        )
        return ToolResult(
            content=self._text_content(response.body),
            structured_content=SdmxProxyStructuredContent(
                status_code=response.status_code, content_type=response.content_type
            ),
        )
