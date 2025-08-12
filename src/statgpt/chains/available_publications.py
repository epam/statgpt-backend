from common.schemas import AvailablePublicationsTool as AvailablePublicationsToolConfig
from common.schemas import ToolTypes
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.tools import StatGptTool
from statgpt.schemas import ToolArtifact, ToolMessageState


class AvailablePublicationsTool(
    StatGptTool[AvailablePublicationsToolConfig], tool_type=ToolTypes.AVAILABLE_PUBLICATIONS
):
    async def _arun(self, inputs: dict) -> tuple[str, ToolArtifact]:
        response = (
            f"The following publication types are available:\n\n"
            f"{self._format_as_markdown(self._tool_config.details.publication_types)}"
        )

        target = ChainParameters.get_target(inputs)
        target.append_content(response)

        return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))

    @staticmethod
    def _format_as_markdown(publication_types: list) -> str:
        return "\n\n".join(
            [
                f"### {publication_type.name}\n\n{publication_type.description}"
                for publication_type in publication_types
            ]
        )
