from mcp.types import ToolAnnotations

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import StatGptTool
from statgpt.app.schemas import ToolArtifact, ToolMessageState
from statgpt.app.schemas.mcp import AvailablePublicationsStructuredContent, PublicationTypeRecord
from statgpt.common.schemas import AvailablePublicationsTool as AvailablePublicationsToolConfig
from statgpt.common.schemas import ToolTypes


class AvailablePublicationsTool(
    StatGptTool[AvailablePublicationsToolConfig], tool_type=ToolTypes.AVAILABLE_PUBLICATIONS
):
    @classmethod
    def get_mcp_annotations(cls) -> ToolAnnotations:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @classmethod
    def get_mcp_output_model(cls) -> type[AvailablePublicationsStructuredContent]:
        return AvailablePublicationsStructuredContent

    async def _arun(self, inputs: dict) -> tuple[str, ToolArtifact]:
        publication_types = self._tool_config.details.publication_types
        response = (
            f"The following publication types are available:\n\n"
            f"{self._format_as_markdown(publication_types)}"
        )

        target = ChainParameters.get_target(inputs)
        target.append_content(response)

        structured = AvailablePublicationsStructuredContent(
            publication_types=[
                PublicationTypeRecord(name=pt.name, description=pt.description)
                for pt in publication_types
            ],
            count=len(publication_types),
        )
        return response, ToolArtifact(
            state=ToolMessageState(type=self.tool_type), mcp_structured=structured
        )

    @staticmethod
    def _format_as_markdown(publication_types: list) -> str:
        return "\n\n".join(
            [
                f"### {publication_type.name}\n\n{publication_type.description}"
                for publication_type in publication_types
            ]
        )
