import asyncio
from typing import Any

from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import PrivateAttr

from statgpt.app.chains.data_query.data_query_tool import DataQueryArgs
from statgpt.app.chains.data_query.runner import DataQueryRunner
from statgpt.app.mcp.attachments import (
    data_query_outcome_to_meta,
    data_query_outcome_to_resources,
    data_query_outcome_to_structured_content,
)
from statgpt.app.schemas.mcp import DataQueryStructuredContent
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas import DataQueryTool as DataQueryToolConfig
from statgpt.common.schemas import ToolTypes

from .base import StatGptMcpTool


class DataQueryMcpTool(
    StatGptMcpTool[DataQueryToolConfig, DataQueryArgs], tool_type=ToolTypes.DATA_QUERY
):
    _runner: DataQueryRunner = PrivateAttr()

    def __init__(
        self,
        tool_config: DataQueryToolConfig,
        channel_config: ChannelConfig,
        inputs: dict[str, Any],
        auth_context: AuthContext,
        **kwargs: Any,
    ):
        super().__init__(tool_config, channel_config, inputs, auth_context, **kwargs)
        self._runner = DataQueryRunner(tool_config.details, channel_config)

    @classmethod
    def get_args_schema(cls, tool_config: DataQueryToolConfig) -> type[DataQueryArgs]:
        return DataQueryArgs

    @classmethod
    def get_output_model(cls) -> type[DataQueryStructuredContent]:
        return DataQueryStructuredContent

    async def _execute(self, args: DataQueryArgs) -> ToolResult:
        outcome = await self._runner.run(args.inputs, args.query)

        content = self._text_content(outcome.response)
        # The CSV and Markdown conversions are CPU-bound and can block on large dataframes;
        # offload them to a worker thread.
        content.extend(
            await asyncio.to_thread(
                data_query_outcome_to_resources,
                outcome,
                self._tool_config.details.mcp_resources,
            )
        )
        # The model reads the structured content; the clients read `_meta`, one payload per
        # audience. Null fields are dropped from what the model sees: they carry no information
        # and the declared output schema marks them optional.
        structured_content = data_query_outcome_to_structured_content(outcome).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        meta = data_query_outcome_to_meta(
            outcome,
            self._channel_config,
            self._tool_config,
            message=outcome.response or None,
        )
        # A content block of its own rather than a suffix on the response, which is also
        # reported as `message` for a client to parse.
        if discovery_block := outcome.discovery_block:
            content.append(TextContent(type="text", text=discovery_block))
        return ToolResult(content=content, structured_content=structured_content, meta=meta)
