from typing import Annotated

from pydantic import Field

from statgpt.app.chains.tools import GuardrailInput, StatGptTool, ToolArgs
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas import DataQueryTool as DataQueryToolConfig
from statgpt.common.schemas.enums import ToolTypes

from .runner import DataQueryRunner


class DataQueryArgs(ToolArgs):
    query: Annotated[str, GuardrailInput] = Field(
        description="An indicator with all of its filters in plain text. "
        "Specify all countries, dates, frequencies, datasets the user requested. "
        "The query must reflect only what the user asked for — do not add, infer, or expand any filters."
    )


class DataQueryTool(StatGptTool[DataQueryToolConfig], tool_type=ToolTypes.DATA_QUERY):

    def __init__(self, tool_config: DataQueryToolConfig, channel_config: ChannelConfig, **kwargs):
        super().__init__(tool_config, channel_config, **kwargs)
        self._runner = DataQueryRunner(tool_config.details, channel_config)

    @classmethod
    def get_args_schema(cls, tool_config: DataQueryToolConfig) -> type[DataQueryArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return DataQueryArgs

    async def _arun(self, inputs: dict, query: str) -> tuple[str, DataQueryArtifact]:
        outcome = await self._runner.run(inputs, query)

        response = outcome.response
        if discovery_block := outcome.discovery_block:
            response = f"{response}\n\n{discovery_block}"

        return response, DataQueryArtifact.from_outcome(outcome)
