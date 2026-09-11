import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from fastmcp.resources import Resource
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool
from fastmcp.utilities.components import FastMCPComponent
from fastmcp.utilities.versions import VersionSpec
from starlette.requests import Request

from statgpt.app.config import ChainParametersConfig
from statgpt.app.mcp.decorators import guard_channel_resolution
from statgpt.app.mcp.exceptions import MissingDeploymentIdError
from statgpt.app.mcp.tools import StatGptMcpTool
from statgpt.app.mcp.widget_resource import WidgetResource
from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.security import DialAuthCredentials, create_auth_context
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.app.utils.dial_stages import DummyStage, NullChoice
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import InvocationSource, ProxiedResourceConfig

_log = logging.getLogger(__name__)


def _build_mcp_inputs(
    auth_context: AuthContext,
    data_service: ChannelServiceFacade,
) -> dict[str, Any]:
    configuration = StatGPTConfiguration()
    return {
        ChainParametersConfig.CHOICE: NullChoice(),
        ChainParametersConfig.AUTH_CONTEXT: auth_context,
        ChainParametersConfig.DATA_SERVICE: data_service,
        ChainParametersConfig.STATE: {},
        ChainParametersConfig.CONFIGURATION: configuration,
        ChainParametersConfig.TARGET: DummyStage(),
        ChainParametersConfig.START_OF_REQUEST: datetime.now(configuration.tzinfo),
        ChainParametersConfig.INVOCATION_SOURCE: InvocationSource.MCP,
    }


class ChannelToolProvider(Provider):
    """MCP Provider that dynamically serves tools from a StatGPT channel config."""

    async def _resolve_context(self, request: Request) -> tuple[AuthContext, ChannelServiceFacade]:
        deployment_id = request.path_params.get("deployment_id")
        if not deployment_id:
            raise MissingDeploymentIdError()
        channel_service = await ChannelServiceFacade.get_channel(deployment_id)
        auth_context = await create_auth_context(
            DialAuthCredentials.from_headers(request.headers),
            bearer_token_required=channel_service.channel_config.bearer_token_required,
        )
        return auth_context, channel_service

    @guard_channel_resolution(default=[], log_prefix="tools/list")
    async def _list_tools(self) -> Sequence[Tool]:
        auth_context, channel_service = await self._resolve_context(get_http_request())

        _log.info("Resolving MCP tools list for the `%s` channel", channel_service.deployment_id)

        channel_config = channel_service.channel_config
        inputs = _build_mcp_inputs(auth_context, channel_service)
        tools: list[Tool] = []
        for tool_config in channel_config.tools:
            try:
                tools.append(
                    StatGptMcpTool.from_config(tool_config, channel_config, inputs, auth_context)
                )
            except Exception:
                _log.warning("Failed to create MCP tool for %s", tool_config.name, exc_info=True)
        return tools

    @guard_channel_resolution(default=None, log_prefix="tools/call", detail_arg="name")
    async def _get_tool(self, name: str, version: VersionSpec | None = None) -> Tool | None:
        _log.info("%s tool is called via MCP", name)

        auth_context, channel_service = await self._resolve_context(get_http_request())

        channel_config = channel_service.channel_config
        prefix = channel_config.mcp.tool_name_prefix
        if prefix:
            if not name.startswith(prefix):
                _log.warning(
                    "MCP tool %s not found: name does not start with the `%s` prefix",
                    name,
                    prefix,
                )
                return None
            name = name.removeprefix(prefix)
        inputs = _build_mcp_inputs(auth_context, channel_service)
        for tool_config in channel_config.tools:
            if tool_config.effective_mcp_name == name:
                return StatGptMcpTool.from_config(tool_config, channel_config, inputs, auth_context)
        _log.warning(
            "MCP tool %s not found in the `%s` channel", name, channel_service.deployment_id
        )
        return None

    @staticmethod
    def _build_resource(config: ProxiedResourceConfig) -> Resource:
        # Today only ProxiedResourceConfig exists; dispatch on type as more kinds are added.
        return WidgetResource.from_config(config)

    @guard_channel_resolution(default=[], log_prefix="resources/list")
    async def _list_resources(self) -> Sequence[Resource]:
        _, channel_service = await self._resolve_context(get_http_request())
        resources: list[Resource] = []
        for resource_config in channel_service.channel_config.mcp.resources:
            try:
                resources.append(self._build_resource(resource_config))
            except Exception:
                _log.warning(
                    "Failed to create MCP resource for %s", resource_config.uri, exc_info=True
                )
        return resources

    @guard_channel_resolution(default=None, log_prefix="resources/read", detail_arg="uri")
    async def _get_resource(self, uri: str, version: VersionSpec | None = None) -> Resource | None:
        _log.info("%s resource is requested via MCP", uri)
        _, channel_service = await self._resolve_context(get_http_request())
        for resource_config in channel_service.channel_config.mcp.resources:
            if resource_config.uri == uri:
                return self._build_resource(resource_config)
        _log.warning(
            "MCP resource %s not found in the `%s` channel", uri, channel_service.deployment_id
        )
        return None

    async def get_tasks(self) -> Sequence[FastMCPComponent]:
        # Tools are per-request (depend on deployment_id + auth headers from the
        # HTTP request), so there is nothing to register as a Docket background
        # task at startup. Returning [] also prevents _list_tools() from being
        # called outside a request context during lifespan startup, which would
        # otherwise log a spurious "No active HTTP request found" error.
        return []


channel_tool_provider = ChannelToolProvider()
