from pydantic import Field

from common.schemas import ChannelConfig, ToolTypes
from common.schemas import WebSearchAgentTool as WebSearchToolConfig
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.tools import StatGptTool, ToolArgs
from statgpt.schemas import ToolArtifact, ToolMessageState

from .response_producer import RagResponseProducer, UrlOnlyResponseProducer


class WebSearchArgs(ToolArgs):
    query: str = Field(description="Natural language query optimized for ai web search.")


class WebSearchAgentTool(StatGptTool[WebSearchToolConfig], tool_type=ToolTypes.WEB_SEARCH_AGENT):
    def __init__(self, tool_config: WebSearchToolConfig, channel_config: ChannelConfig, **kwargs):
        super().__init__(tool_config, channel_config, **kwargs)

        kwargs = dict(
            deployment_id=tool_config.details.deployment_id,
            stages_config=tool_config.details.stages_config,
            system_prompt=tool_config.details.system_prompt,
        )
        if tool_config.details.urls_only:
            self._response_producer = UrlOnlyResponseProducer(**kwargs)
        else:
            self._response_producer = RagResponseProducer(**kwargs)

    @classmethod
    def get_args_schema(cls, tool_config: WebSearchToolConfig) -> type[WebSearchArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return WebSearchArgs

    async def _arun(self, inputs: dict, query: str, **kwargs) -> tuple[str, ToolArtifact]:
        target = ChainParameters.get_target(inputs)
        target.append_name(f": {query}")

        str_response = await self._response_producer.run(inputs=inputs, query=query)
        target.append_content(str_response)

        artifact = ToolArtifact(state=ToolMessageState(type=self.tool_type))
        return str_response, artifact
