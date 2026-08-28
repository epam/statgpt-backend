import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastmcp.apps import AppConfig, app_config_to_meta_dict
from fastmcp.exceptions import ToolError
from fastmcp.resources import Resource
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool, ToolResult
from fastmcp.utilities.components import FastMCPComponent
from fastmcp.utilities.versions import VersionSpec
from mcp.types import ContentBlock, TextContent
from pydantic import PrivateAttr, ValidationError
from starlette.requests import Request

from statgpt.app.chains.tools import StatGptTool, ToolUpstreamError
from statgpt.app.config import ChainParametersConfig
from statgpt.app.mcp.attachments import (
    data_query_artifact_to_resources,
    data_query_artifact_to_structured_content,
)
from statgpt.app.mcp.decorators import guard_channel_resolution
from statgpt.app.mcp.exceptions import MissingDeploymentIdError
from statgpt.app.mcp.guardrails import enforce_input_guardrail
from statgpt.app.mcp.widget_resource import WidgetResource
from statgpt.app.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.app.schemas.mcp import DataQueryStructuredContent, SdmxProxyStructuredContent
from statgpt.app.schemas.service import ChannelDatasetsMetadataResponse
from statgpt.app.schemas.tool_artifact import (
    DataQueryArtifact,
    DatasetsMetadataAppArtifact,
    SdmxQueryAppArtifact,
)
from statgpt.app.security import DialAuthCredentials, create_auth_context
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.app.utils.dial_stages import DummyStage, NullChoice
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import (
    BaseToolConfig,
    ChannelConfig,
    DataQueryMcpResources,
    DataQueryTool,
    InvocationSource,
    ProxiedResourceConfig,
)

_log = logging.getLogger(__name__)


def _tool_app_config(tool_config: BaseToolConfig) -> AppConfig | None:
    """Build the MCP Apps config (``_meta.ui``) from the config's MCP-App fields.

    Uses fastmcp's typed ``AppConfig`` so the wire format (camelCase aliases per the
    MCP Apps extension) stays in sync with the library. Returns ``None`` when neither
    field is set so ``_meta`` is omitted and the host applies the spec default
    visibility (``["model", "app"]``).
    """
    if tool_config.mcp_visibility is None and tool_config.mcp_app_resource_uri is None:
        return None
    return AppConfig(
        visibility=tool_config.mcp_visibility,
        resource_uri=tool_config.mcp_app_resource_uri,
    )


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


class _McpToolAdapter(Tool):
    """A FastMCP Tool backed by a StatGptTool instance."""

    _langchain_tool: StatGptTool = PrivateAttr()
    _inputs: dict[str, Any] = PrivateAttr()
    _channel_config: ChannelConfig = PrivateAttr()
    _tool_config: BaseToolConfig = PrivateAttr()
    _auth_context: AuthContext = PrivateAttr()

    def __init__(
        self,
        langchain_tool: StatGptTool,
        inputs: dict[str, Any],
        channel_config: ChannelConfig,
        tool_config: BaseToolConfig,
        auth_context: AuthContext,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._langchain_tool = langchain_tool
        self._inputs = inputs
        self._channel_config = channel_config
        self._tool_config = tool_config
        self._auth_context = auth_context

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        # Screen arbitrary free-text input with the out-of-scope guardrail before
        # executing. Raised ToolError propagates to the MCP client unchanged.
        await enforce_input_guardrail(
            self._langchain_tool, arguments, self._channel_config, self._auth_context
        )
        tool_call = {
            "name": self._langchain_tool.name,
            "args": {**arguments, "inputs": self._inputs},
            "id": str(uuid4()),
            "type": "tool_call",
        }
        try:
            result = await self._langchain_tool.ainvoke(tool_call)
        except ValidationError as e:
            # Argument-schema validation failures (e.g. missing required field, bad enum
            # value). Surface a concise message instead of the generic failure.
            _log.info("Invalid arguments for MCP tool %s: %s", self._langchain_tool.name, e)
            raise ToolError(f"Invalid arguments for {self._langchain_tool.name}: {e}") from e
        except ToolUpstreamError as e:
            # Upstream dependency failure (connection/timeout): surface the specific message.
            _log.warning("Upstream error in MCP tool %s: %s", self._langchain_tool.name, e)
            raise ToolError(str(e)) from e
        except Exception:
            # Catch-all for unexpected errors. Known error cases should return
            # proper content or raise a custom exception caught in a dedicated
            # except block above this one.
            _log.exception("Error executing MCP tool %s", self._langchain_tool.name)
            raise ToolError(f"{self._langchain_tool.name} tool failed to execute")
        text = result.content if isinstance(result.content, str) else str(result.content)
        content: list[ContentBlock] = [TextContent(type="text", text=text)] if text else []
        structured_content: (
            DataQueryStructuredContent
            | SdmxProxyStructuredContent
            | ChannelDatasetsMetadataResponse
            | None
        ) = None
        if isinstance(result.artifact, DataQueryArtifact):
            mcp_resources = (
                self._tool_config.details.mcp_resources
                if isinstance(self._tool_config, DataQueryTool)
                else DataQueryMcpResources()
            )
            # The CSV and Markdown conversions are CPU-bound and can block on large
            # dataframes; offload them to a worker thread.
            resources = await asyncio.to_thread(
                data_query_artifact_to_resources, result.artifact, mcp_resources
            )
            content.extend(resources)
            structured_content = data_query_artifact_to_structured_content(
                result.artifact, self._channel_config, message=text or None
            )
            # A content block of its own rather than a suffix on the response: the response is
            # also reported as `message`, which a client parses, and these datasets are an aside
            # a client is free to render separately or not at all.
            if discovery_block := result.artifact.discovery_datasets_block:
                content.append(TextContent(type="text", text=discovery_block))
        elif isinstance(result.artifact, SdmxQueryAppArtifact):
            # Surface the upstream HTTP metadata so the MCP-App can distinguish success from
            # error responses and know the body's media type. The raw body stays in the text
            # content block above to keep the passthrough behavior.
            structured_content = SdmxProxyStructuredContent(
                status_code=result.artifact.status_code,
                content_type=result.artifact.content_type,
            )
        elif isinstance(result.artifact, DatasetsMetadataAppArtifact):
            # Surface the datasets metadata payload as structured content so the UI widget can
            # consume it directly (the JSON body is also in the text content block above).
            structured_content = result.artifact.response
        _log.info(
            "Sending MCP tool %s response: %d content block(s), structured_content=%s",
            self._langchain_tool.name,
            len(content),
            type(structured_content).__name__ if structured_content else None,
        )
        return ToolResult(content=content, structured_content=structured_content)


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

    def _create_mcp_tool(
        self,
        tool_config: BaseToolConfig,
        channel_config: ChannelConfig,
        inputs: dict[str, Any],
        auth_context: AuthContext,
    ) -> _McpToolAdapter:
        langchain_tool = StatGptTool.from_config(tool_config, channel_config)
        app_config = _tool_app_config(tool_config)
        return _McpToolAdapter(
            langchain_tool=langchain_tool,
            inputs=inputs,
            channel_config=channel_config,
            tool_config=tool_config,
            auth_context=auth_context,
            name=channel_config.mcp.tool_name_prefix + tool_config.effective_mcp_name,
            description=tool_config.effective_mcp_description,
            parameters=langchain_tool.get_public_args_schema(),
            annotations=langchain_tool.get_mcp_annotations(),
            meta={"ui": app_config_to_meta_dict(app_config)} if app_config else None,
        )

    @guard_channel_resolution(default=[], log_prefix="tools/list")
    async def _list_tools(self) -> Sequence[Tool]:
        auth_context, channel_service = await self._resolve_context(get_http_request())

        _log.info("Resolving MCP tools list for the `%s` channel", channel_service.deployment_id)

        channel_config = channel_service.channel_config
        inputs = _build_mcp_inputs(auth_context, channel_service)
        tools: list[Tool] = []
        for tool_config in channel_config.tools:
            try:
                mcp_tool = self._create_mcp_tool(tool_config, channel_config, inputs, auth_context)
                tools.append(mcp_tool)
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
                return self._create_mcp_tool(tool_config, channel_config, inputs, auth_context)
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
