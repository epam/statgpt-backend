from mcp.types import ToolAnnotations

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import StatGptTool
from statgpt.app.schemas import DatasetsMetadataAppArtifact, ToolMessageState
from statgpt.app.services.channel_datasets_metadata import build_channel_datasets_metadata
from statgpt.common.schemas import DatasetsMetadataAppTool as DatasetsMetadataAppToolConfig
from statgpt.common.schemas import ToolTypes


class DatasetsMetadataAppTool(
    StatGptTool[DatasetsMetadataAppToolConfig], tool_type=ToolTypes.DATASETS_METADATA_APP
):
    """MCP-App-only tool that returns the channel's datasets metadata (the same payload as the
    `/metadata/datasets` service endpoint) so the UI widget can render it. Hidden from the
    Supreme Agent / model, which does not need dataset metadata through this path.
    """

    @classmethod
    def get_mcp_annotations(cls) -> ToolAnnotations:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    async def _arun(self, inputs: dict, **kwargs) -> tuple[str, DatasetsMetadataAppArtifact]:
        data_service = ChainParameters.get_data_service(inputs)
        auth_context = ChainParameters.get_auth_context(inputs)

        response = await build_channel_datasets_metadata(data_service.channel, auth_context)

        return response.model_dump_json(), DatasetsMetadataAppArtifact(
            state=ToolMessageState(type=self.tool_type),
            response=response,
        )
