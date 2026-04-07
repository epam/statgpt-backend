import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastmcp.server.dependencies import get_http_request
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool, ToolResult
from mcp.types import TextContent
from pydantic import PrivateAttr

from statgpt.app.chains.tools import StatGptTool
from statgpt.app.config import ChainParametersConfig
from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.security import create_auth_context
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.app.utils.dial_stages import DummyStage, NullChoice
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import BaseToolConfig, ChannelConfig

_log = logging.getLogger(__name__)


class _HeaderOnlyDialRequest:
    """Adapter for auth checks from MCP HTTP headers."""

    def __init__(self, headers: dict[str, str]):
        self._api_key = headers.get("api-key") or headers.get("x-api-key")
        token = headers.get("authorization")
        self._bearer_token = (
            token[7:] if token is not None and token.startswith("Bearer ") else None
        )

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def bearer_token(self) -> str | None:
        return self._bearer_token


def _build_mcp_inputs(
    auth_context: AuthContext,
    data_service: ChannelServiceFacade,
) -> dict:
    configuration = StatGPTConfiguration()
    return {
        ChainParametersConfig.CHOICE: NullChoice(),
        ChainParametersConfig.AUTH_CONTEXT: auth_context,
        ChainParametersConfig.DATA_SERVICE: data_service,
        ChainParametersConfig.STATE: {},
        ChainParametersConfig.CONFIGURATION: configuration,
        ChainParametersConfig.TARGET: DummyStage(),
        ChainParametersConfig.START_OF_REQUEST: datetime.now(configuration.tzinfo),
    }


def _get_tool_input_schema(tool: StatGptTool) -> dict[str, Any]:
    """Get JSON Schema for tool parameters, excluding injected args."""
    schema = tool.get_input_schema().model_json_schema()
    # Remove fields that are injected and not user-facing
    props = schema.get("properties", {})
    props.pop("inputs", None)
    required = schema.get("required", [])
    if "inputs" in required:
        required.remove("inputs")
    return schema


class _McpToolAdapter(Tool):
    """A FastMCP Tool backed by a StatGptTool instance."""

    _langchain_tool: StatGptTool = PrivateAttr()
    _inputs: dict = PrivateAttr()

    def __init__(
        self,
        langchain_tool: StatGptTool,
        inputs: dict,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._langchain_tool = langchain_tool
        self._inputs = inputs

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        arguments["inputs"] = self._inputs
        tool_call = {
            "name": self._langchain_tool.name,
            "args": arguments,
            "id": str(uuid4()),
        }
        result = await self._langchain_tool.ainvoke(tool_call)
        content = result.content if isinstance(result.content, str) else str(result.content)
        return ToolResult(content=[TextContent(type="text", text=content)])


class ChannelToolProvider(Provider):
    """MCP Provider that dynamically serves tools from a StatGPT channel config."""

    def _get_headers(self) -> dict[str, str]:
        request = get_http_request()
        return dict(request.headers)

    async def _resolve_context(
        self, headers: dict[str, str]
    ) -> tuple[AuthContext, ChannelServiceFacade]:
        auth_context = await create_auth_context(_HeaderOnlyDialRequest(headers))
        deployment_id = headers.get("x-dial-application-id")
        if not deployment_id:
            raise ValueError("Missing x-dial-application-id header")
        channel_service = await ChannelServiceFacade.get_channel(deployment_id)
        return auth_context, channel_service

    def _create_mcp_tool(
        self,
        tool_config: BaseToolConfig,
        channel_config: ChannelConfig,
        auth_context: AuthContext,
        channel_service: ChannelServiceFacade,
    ) -> _McpToolAdapter:
        langchain_tool = StatGptTool.from_config(tool_config, channel_config)
        inputs = _build_mcp_inputs(auth_context, channel_service)
        return _McpToolAdapter(
            langchain_tool=langchain_tool,
            inputs=inputs,
            name=tool_config.name,
            description=tool_config.description,
            parameters=_get_tool_input_schema(langchain_tool),
        )

    async def _list_tools(self) -> Sequence[Tool]:
        try:
            headers = self._get_headers()
            auth_context, channel_service = await self._resolve_context(headers)
        except Exception:
            _log.exception("Could not resolve channel context for tools/list")
            return []

        channel_config = channel_service.channel_config
        tools: list[Tool] = []
        for tool_config in channel_config.tools:
            try:
                mcp_tool = self._create_mcp_tool(
                    tool_config, channel_config, auth_context, channel_service
                )
                tools.append(mcp_tool)
            except Exception:
                _log.warning(f"Failed to create MCP tool for {tool_config.name}", exc_info=True)
        return tools

    async def _get_tool(self, name: str, version=None) -> Tool | None:
        try:
            headers = self._get_headers()
            auth_context, channel_service = await self._resolve_context(headers)
        except Exception:
            _log.exception("Could not resolve channel context for tools/call")
            return None

        channel_config = channel_service.channel_config
        for tool_config in channel_config.tools:
            if tool_config.name == name:
                return self._create_mcp_tool(
                    tool_config, channel_config, auth_context, channel_service
                )
        return None


channel_tool_provider = ChannelToolProvider()
