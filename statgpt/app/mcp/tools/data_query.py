import asyncio
from typing import Annotated, Any, Self

from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field, PrivateAttr, model_validator

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
from statgpt.common.schemas import DataQueryValueRounding, ToolTypes

from .base import StatGptMcpTool

# A float carries about 15 significant digits, so rounding to more decimal places changes nothing.
_MAX_DECIMAL_PLACES = 15
_DecimalPlaces = Annotated[int | None, Field(ge=0, le=_MAX_DECIMAL_PLACES)]
_DECIMAL_PLACES_DESCRIPTION = (
    f"Number of decimal places, from 0 to {_MAX_DECIMAL_PLACES}, to round the returned values to."
    " A value too small for that precision keeps its first significant digit instead of reading"
    " as 0. Set it only when the user asks for a specific precision."
)
_FULL_PRECISION_DESCRIPTION = (
    "Return the values unrounded, at the full precision of the source."
    " Set it only when the user asks for exact or unrounded values."
    " Cannot be combined with `decimal_places`."
)


class DataQueryMcpArgs(DataQueryArgs):
    """The data query args, plus the rounding an MCP caller can ask for on one call."""

    decimal_places: _DecimalPlaces = Field(default=None, description=_DECIMAL_PLACES_DESCRIPTION)
    full_precision: bool = Field(default=False, description=_FULL_PRECISION_DESCRIPTION)

    @model_validator(mode="after")
    def _check_one_precision(self) -> Self:
        if self.full_precision and self.decimal_places is not None:
            raise ValueError("Set either `decimal_places` or `full_precision`, not both.")
        return self

    @property
    def value_rounding(self) -> DataQueryValueRounding | None:
        """The rounding this call asks for, or `None` to keep the configured one."""
        if self.full_precision:
            return DataQueryValueRounding(decimal_places=None)
        if self.decimal_places is not None:
            return DataQueryValueRounding(decimal_places=self.decimal_places)
        return None


def build_data_query_mcp_args(tool_config: DataQueryToolConfig) -> type[DataQueryMcpArgs]:
    """The MCP args schema, with the configured default precision spelled out in the description."""

    default = tool_config.details.value_rounding.decimal_places
    default_msg = (
        f" When omitted, the values are rounded to {default} decimal places."
        if default is not None
        else " When omitted, the values keep their full precision."
    )

    class DataQueryMcpArgsWithDefault(DataQueryMcpArgs):
        decimal_places: _DecimalPlaces = Field(
            default=None, description=_DECIMAL_PLACES_DESCRIPTION + default_msg
        )

    return DataQueryMcpArgsWithDefault


class DataQueryMcpTool(
    StatGptMcpTool[DataQueryToolConfig, DataQueryMcpArgs], tool_type=ToolTypes.DATA_QUERY
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
    def get_args_schema(cls, tool_config: DataQueryToolConfig) -> type[DataQueryMcpArgs]:
        return build_data_query_mcp_args(tool_config)

    @classmethod
    def get_output_model(cls) -> type[DataQueryStructuredContent]:
        return DataQueryStructuredContent

    async def _execute(self, args: DataQueryMcpArgs) -> ToolResult:
        outcome = await self._runner.run(
            args.inputs, args.query, value_rounding=args.value_rounding
        )

        # The model reads the structured content, also sent as its JSON text; the clients read
        # `_meta`, one payload per audience.
        result = self._structured_only(
            data_query_outcome_to_structured_content(outcome, self._tool_config)
        )
        content = list(result.content)
        # The CSV and Markdown conversions are CPU-bound and can block on large dataframes;
        # offload them to a worker thread.
        content.extend(
            await asyncio.to_thread(
                data_query_outcome_to_resources,
                outcome,
                self._tool_config.details.mcp_resources,
            )
        )
        meta = data_query_outcome_to_meta(
            outcome,
            self._channel_config,
            self._tool_config,
            message=outcome.response or None,
        )
        # A content block of its own: it comes from a lookup beside the query, not from its result.
        if discovery_block := outcome.discovery_block:
            content.append(TextContent(type="text", text=discovery_block))
        return ToolResult(content=content, structured_content=result.structured_content, meta=meta)
