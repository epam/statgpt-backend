from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import StatGptTool
from statgpt.app.schemas import ToolArtifact, ToolMessageState
from statgpt.common.schemas import AvailablePublicationsTool as AvailablePublicationsToolConfig
from statgpt.common.schemas import ToolTypes


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
