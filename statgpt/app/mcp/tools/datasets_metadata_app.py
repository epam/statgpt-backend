from fastmcp.tools import ToolResult

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import ToolArgs
from statgpt.app.services.channel_datasets_metadata import build_channel_datasets_metadata
from statgpt.common.schemas import DatasetsMetadataAppTool as DatasetsMetadataAppToolConfig
from statgpt.common.schemas import ToolTypes

from .base import StatGptMcpTool


class DatasetsMetadataAppMcpTool(
    StatGptMcpTool[DatasetsMetadataAppToolConfig, ToolArgs],
    tool_type=ToolTypes.DATASETS_METADATA_APP,
):
    """MCP-App-only: returns the channel's datasets metadata (the same payload as the
    `/metadata/datasets` service endpoint) as structured content for the UI widget."""

    async def _execute(self, args: ToolArgs) -> ToolResult:
        data_service = ChainParameters.get_data_service(args.inputs)
        response = await build_channel_datasets_metadata(data_service.channel, self._auth_context)
        return ToolResult(content=[], structured_content=response)
