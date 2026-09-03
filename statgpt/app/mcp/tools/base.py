"""The MCP interface of a StatGPT tool.

A `StatGptMcpTool` is a FastMCP `Tool` that runs the tool's implementation and returns a complete
`ToolResult` (content blocks plus structured content) by itself. The base class owns what every
tool shares: argument validation, the input guardrail, error mapping and logging. Subclasses own
what differs per tool: the args schema, the annotations, the output model and `_execute`.

Tool classes register themselves by tool type (``class X(StatGptMcpTool[...], tool_type=...)``),
the same way the LangChain interfaces do. Tool types without a dedicated MCP interface are served
by `LangChainMcpTool`, which runs the LangChain tool and returns its text.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar
from uuid import uuid4

from fastmcp.apps import AppConfig, app_config_to_meta_dict
from fastmcp.exceptions import ToolError
from fastmcp.tools import Tool, ToolResult
from mcp.types import ContentBlock, TextContent, ToolAnnotations
from pydantic import BaseModel, PrivateAttr, ValidationError

from statgpt.app.chains.tools import StatGptTool, ToolArgs, ToolUpstreamError
from statgpt.app.mcp.guardrails import enforce_input_guardrail
from statgpt.app.mcp.output_schema import model_to_output_schema
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import BaseToolConfig, ChannelConfig, ToolTypes

_log = logging.getLogger(__name__)

ToolConfigType = TypeVar("ToolConfigType", bound=BaseToolConfig)
ArgsType = TypeVar("ArgsType", bound=ToolArgs)

_DEFAULT_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, openWorldHint=False
)
# Tools that reach out to the open web; every other tool only reads the channel's own data.
_OPEN_WORLD_TOOL_TYPES = frozenset(
    {
        ToolTypes.WEB_SEARCH,
        ToolTypes.WEB_SEARCH_AGENT,
    }
)

_MCP_TOOL_IMPLEMENTATIONS: dict[ToolTypes, type["StatGptMcpTool"]] = {}


def tool_app_config(tool_config: BaseToolConfig) -> AppConfig | None:
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


class StatGptMcpTool(Tool, ABC, Generic[ToolConfigType, ArgsType]):
    """A FastMCP Tool serving one StatGPT tool config. See the module docstring."""

    tool_type: ClassVar[ToolTypes | None] = None

    _tool_config: ToolConfigType = PrivateAttr()
    _channel_config: ChannelConfig = PrivateAttr()
    _inputs: dict[str, Any] = PrivateAttr()
    _auth_context: AuthContext = PrivateAttr()
    _args_schema: type[ArgsType] = PrivateAttr()

    def __init_subclass__(cls, tool_type: ToolTypes | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if tool_type is None:
            # Pydantic's generic parametrizations, and classes that serve several tool types
            # (the LangChain-backed default), are not registered.
            return
        if tool_type in _MCP_TOOL_IMPLEMENTATIONS:
            raise ValueError(
                f"Cant register {cls} MCP tool:"
                f" {tool_type=} already registered with {_MCP_TOOL_IMPLEMENTATIONS[tool_type]}"
            )
        cls.tool_type = tool_type
        _MCP_TOOL_IMPLEMENTATIONS[tool_type] = cls

    def __init__(
        self,
        tool_config: ToolConfigType,
        channel_config: ChannelConfig,
        inputs: dict[str, Any],
        auth_context: AuthContext,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._tool_config = tool_config
        self._channel_config = channel_config
        self._inputs = inputs
        self._auth_context = auth_context
        self._args_schema = type(self).get_args_schema(tool_config)

    # ~~~~~~~~~~~~~ per-tool hooks ~~~~~~~~~~~~~

    @classmethod
    def get_args_schema(cls, tool_config: ToolConfigType) -> type[ArgsType]:
        """The schema the tool's arguments are validated against (and advertised as)."""
        return ToolArgs  # type: ignore[return-value]

    @classmethod
    def get_annotations(cls, tool_config: ToolConfigType) -> ToolAnnotations:
        """MCP tool hints clients may use for UX decisions (e.g. skipping confirmation prompts
        for read-only tools, flagging open-world tools). Advisory only."""
        return _DEFAULT_ANNOTATIONS

    @classmethod
    def get_output_model(cls) -> type[BaseModel] | None:
        """The Pydantic model describing this tool's ``structuredContent``, or ``None`` when the
        tool declares no output schema. Declaring it pins the response shape so the advertised
        schema and the runtime payload cannot drift (guarded in tests)."""
        return None

    @abstractmethod
    async def _execute(self, args: ArgsType) -> ToolResult:
        """Run the tool with validated arguments and return the complete MCP result."""

    # ~~~~~~~~~~~~~ cross-cutting ~~~~~~~~~~~~~

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        # Validate first: it is cheap, fails fast, and guarantees the guardrail below screens a
        # real string. A concise message names the offending field instead of a generic failure.
        try:
            args = self._args_schema.model_validate({**arguments, "inputs": self._inputs})
        except ValidationError as e:
            _log.debug("Invalid arguments for MCP tool %s: %s", self.name, e)
            raise ToolError(f"Invalid arguments for {self.name}: {e}") from e

        # Screen arbitrary free-text input with the out-of-scope guardrail before executing.
        # Raised ToolError propagates to the MCP client unchanged.
        await enforce_input_guardrail(
            self.name,
            self._args_schema.get_guardrail_input(arguments),
            self._channel_config,
            self._auth_context,
        )

        try:
            result = await self._execute(args)
        except ToolError:
            raise
        except ToolUpstreamError as e:
            # Upstream dependency failure (connection/timeout): surface the specific message.
            _log.warning("Upstream error in MCP tool %s: %s", self.name, e)
            raise ToolError(str(e)) from e
        except Exception:
            # Catch-all for unexpected errors. Known error cases should return proper content or
            # raise a custom exception caught in a dedicated except block above this one.
            _log.exception("Error executing MCP tool %s", self.name)
            raise ToolError(f"{self.name} tool failed to execute")

        _log.info(
            "Sending MCP tool %s response: %d content block(s), structured_content=%s",
            self.name,
            len(result.content),
            "set" if result.structured_content is not None else None,
        )
        return result

    @staticmethod
    def _text_content(text: str) -> list[ContentBlock]:
        return [TextContent(type="text", text=text)] if text else []

    @staticmethod
    def _structured_only(model: BaseModel) -> ToolResult:
        """A result whose complete payload is the structured content: no text block (it would
        only duplicate the payload) and null optional fields omitted for compactness."""
        return ToolResult(
            content=[],
            structured_content=model.model_dump(mode="json", by_alias=True, exclude_none=True),
        )

    # ~~~~~~~~~~~~~ factory ~~~~~~~~~~~~~

    @classmethod
    def get_output_schema(cls) -> dict[str, Any] | None:
        model = cls.get_output_model()
        return model_to_output_schema(model) if model is not None else None

    @staticmethod
    def from_config(
        tool_config: BaseToolConfig,
        channel_config: ChannelConfig,
        inputs: dict[str, Any],
        auth_context: AuthContext,
    ) -> "StatGptMcpTool":
        cls = mcp_tool_class_for(tool_config.type)
        app_config = tool_app_config(tool_config)
        return cls(
            tool_config=tool_config,
            channel_config=channel_config,
            inputs=inputs,
            auth_context=auth_context,
            name=channel_config.mcp.tool_name_prefix + tool_config.effective_mcp_name,
            description=tool_config.effective_mcp_description,
            parameters=cls.get_args_schema(tool_config).get_public_schema(),
            output_schema=cls.get_output_schema(),
            annotations=cls.get_annotations(tool_config),
            meta={"ui": app_config_to_meta_dict(app_config)} if app_config else None,
        )


class LangChainMcpTool(StatGptMcpTool[BaseToolConfig, ToolArgs]):
    """The MCP interface for tools that have no MCP-specific result: runs the LangChain tool and
    returns its text as the single content block."""

    _langchain_tool: StatGptTool = PrivateAttr()

    def __init__(
        self,
        tool_config: BaseToolConfig,
        channel_config: ChannelConfig,
        inputs: dict[str, Any],
        auth_context: AuthContext,
        **kwargs: Any,
    ):
        super().__init__(tool_config, channel_config, inputs, auth_context, **kwargs)
        self._langchain_tool = StatGptTool.from_config(tool_config, channel_config)

    @classmethod
    def get_args_schema(cls, tool_config: BaseToolConfig) -> type[ToolArgs]:
        return StatGptTool.implementation_for(tool_config.type).get_args_schema(tool_config)

    @classmethod
    def get_annotations(cls, tool_config: BaseToolConfig) -> ToolAnnotations:
        if tool_config.type in _OPEN_WORLD_TOOL_TYPES:
            return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)
        return _DEFAULT_ANNOTATIONS

    async def _execute(self, args: ToolArgs) -> ToolResult:
        # `inputs` is re-injected as the live object: dumping it would serialize the context
        # objects it carries. LangChain re-validates the arguments, which is harmless.
        tool_call = {
            "name": self._langchain_tool.name,
            "args": {**args.model_dump(exclude={"inputs"}), "inputs": self._inputs},
            "id": str(uuid4()),
            "type": "tool_call",
        }
        result = await self._langchain_tool.ainvoke(tool_call)
        text = result.content if isinstance(result.content, str) else str(result.content)
        return ToolResult(content=self._text_content(text))


def mcp_tool_class_for(tool_type: ToolTypes) -> type[StatGptMcpTool]:
    """The MCP interface class serving `tool_type`; the LangChain-backed default when the tool has
    no MCP-specific result."""
    return _MCP_TOOL_IMPLEMENTATIONS.get(tool_type, LangChainMcpTool)
