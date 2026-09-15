from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Annotated, Any, ClassVar, Generic, Literal, TypeVar

from langchain_core.tools import BaseTool, InjectedToolArg
from pydantic import BaseModel, Field

from statgpt.app.schemas import ToolArtifact
from statgpt.common.config import multiline_logger as logger
from statgpt.common.schemas import BaseToolConfig, ChannelConfig, ToolTypes


class ToolRegistry(MutableMapping):

    def __init__(self) -> None:
        self._mapping: dict[ToolTypes, type[StatGptTool]] = {}

    def __setitem__(self, tool_type: ToolTypes, factory: type['StatGptTool']) -> None:
        if not issubclass(factory, StatGptTool):
            raise ValueError(f"{factory=} must be a subclass of {StatGptTool}")

        if tool_type in self._mapping:
            raise ValueError(
                f"Cant register {factory} factory:"
                f" {tool_type=} already registered with {self._mapping[tool_type]}"
            )

        self._mapping[tool_type] = factory

    def __delitem__(self, tool_type: ToolTypes, /):
        del self._mapping[tool_type]

    def __getitem__(self, key: ToolTypes) -> type['StatGptTool']:
        if key not in self._mapping:
            raise KeyError(
                f"Factory has not been registered for {key} agent."
                f" Please ensure that the factory is imported in the global space."
                f"\nAvailable agents: {list(self._mapping.keys())}"
            )

        return self._mapping[key]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)


class ToolUpstreamError(Exception):
    """Raised by a tool when an upstream dependency (e.g. an HTTP backend) fails.

    The message is safe to surface to the caller (e.g. translated into an MCP
    ToolError) so that connection/timeout failures yield a clear message instead
    of being masked as a generic execution failure.
    """


_TOOL_IMPLEMENTATIONS = ToolRegistry()


class GuardrailInput:
    """Marks the single free-text field that input guardrails must screen.

    Attach via ``Annotated[str, GuardrailInput]`` on a ToolArgs field. The marker
    travels with the field declaration, so renaming the field keeps the guardrail
    wired up without touching any extraction logic elsewhere. Used like langchain's
    ``InjectedToolArg`` — a bare class placed in the ``Annotated`` metadata."""


class ToolArgs(BaseModel):
    # injected tool argument. set in the code, not by the LLM.
    # LLM can't set this field because it's not added to the tool schema shown to LLM.
    # `inputs` is used to pass execution context from Supreme Agent to the tool.
    inputs: Annotated[dict, InjectedToolArg] = Field()

    @classmethod
    def get_guardrail_input(cls, arguments: dict[str, Any]) -> str | None:
        """Return the free-text user input that should be screened by input
        guardrails, or None if this schema declares no ``GuardrailInput`` field.

        The screened field is identified by the ``GuardrailInput`` marker on the
        schema, so it stays in sync with the schema definition automatically."""
        marked = [
            name for name, field in cls.model_fields.items() if GuardrailInput in field.metadata
        ]
        if len(marked) > 1:
            raise ValueError(f"{cls.__name__} marks multiple guardrail fields: {marked}")
        return arguments.get(marked[0]) if marked else None

    @classmethod
    def get_public_schema(cls) -> dict[str, Any]:
        """Return JSON Schema with injected (non-user-facing) fields removed."""
        injected_fields = {
            name for name, field in cls.model_fields.items() if InjectedToolArg in field.metadata
        }
        schema = cls.model_json_schema()
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for name in injected_fields:
            props.pop(name, None)
            if name in required:
                required.remove(name)
        return schema


ToolConfigType = TypeVar('ToolConfigType', bound=BaseToolConfig)


class StatGptTool(BaseTool, ABC, Generic[ToolConfigType]):
    response_format: Literal['content', 'content_and_artifact'] = "content_and_artifact"
    tool_type: ClassVar[ToolTypes]  # Set dynamically in __init_subclass__

    def __init_subclass__(cls, **kwargs):
        tool_type = kwargs.pop('tool_type', None)

        super().__init_subclass__(**kwargs)

        if StatGptTool in cls.__bases__:
            # we want to register tool instance (like DataQueryTool),
            # not generic itself (like StatGptTool[DataQueryToolConfig])
            return

        if tool_type is None:
            raise ValueError(f"Subclass {cls.__name__} must specify a 'tool_type' parameter")

        if not isinstance(tool_type, ToolTypes):
            raise ValueError(f"{tool_type=} must be an instance of {ToolTypes}")

        cls.tool_type = tool_type
        _TOOL_IMPLEMENTATIONS[tool_type] = cls

    def __init__(self, tool_config: ToolConfigType, channel_config: ChannelConfig, **kwargs):
        super().__init__(**kwargs)

        self._tool_config = tool_config
        self._channel_config = channel_config

    @property
    def default_stage_name(self) -> str:
        """The stage name of calling this tool when none is configured."""
        tool_name = self.name.replace('_', ' ')
        return f"Calling {tool_name} tool"

    @property
    def default_result_stage_name(self) -> str:
        """The stage name of this tool's result when none is configured."""
        tool_name = self.name.replace('_', ' ')
        return f"Result from {tool_name} tool"

    @property
    def stage_name(self) -> str:
        """Return the stage name of calling this tool."""
        return self._tool_config.details.stages_config.tool_call_name or self.default_stage_name

    @property
    def result_stage_name(self) -> str:
        """Return the stage name of showing the result of this tool."""
        return (
            self._tool_config.details.stages_config.tool_result_name
            or self.default_result_stage_name
        )

    def render_stage_name(self, args: dict, inputs: dict) -> str:
        """Return the display name of the stage of calling this tool."""
        return self._render_stage_name(self.stage_name, args, inputs)

    def render_result_stage_name(self, args: dict, inputs: dict) -> str:
        """Return the display name of the stage showing this tool's result."""
        return self._render_stage_name(self.result_stage_name, args, inputs)

    def _render_stage_name(self, name: str, args: dict, inputs: dict) -> str:
        """Fill the placeholders of a configured stage name from the tool call arguments.

        Override to expose values the LLM never passes (e.g. something carried in the conversation
        state) or to substitute a different name when a value the template needs is unavailable."""
        if not args:
            return name
        values = {
            k: ", ".join(map(str, v)) if isinstance(v, list) else str(v) for k, v in args.items()
        }
        try:
            return name.format(**values)
        except (KeyError, IndexError, ValueError) as e:
            # An unknown or malformed placeholder: keep the raw name rather than failing the call.
            logger.warning(f"Error formatting stage name: {e}")
            return name

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """This method is implemented to satisfy the BaseTool interface.
        But it raises an error since we don't want to use it."""
        raise NotImplementedError("This method should not be called. Use async version instead.")

    @abstractmethod
    async def _arun(self, *args: Any, **kwargs: Any) -> tuple[str, ToolArtifact]:
        pass

    @classmethod
    def get_args_schema(cls, tool_config: ToolConfigType) -> type[ToolArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return ToolArgs

    @staticmethod
    def implementation_for(tool_type: ToolTypes) -> type['StatGptTool']:
        """The tool class registered for `tool_type` (the implementing module must be imported)."""
        return _TOOL_IMPLEMENTATIONS[tool_type]

    @staticmethod
    def from_config(tool_config: ToolConfigType, channel_config: ChannelConfig) -> 'StatGptTool':
        cls = StatGptTool.implementation_for(tool_config.type)

        return cls(
            tool_config=tool_config,
            channel_config=channel_config,
            name=tool_config.name,
            description=tool_config.description,
            args_schema=cls.get_args_schema(tool_config),
        )
