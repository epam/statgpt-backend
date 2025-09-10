from abc import abstractmethod
from collections.abc import MutableMapping
from typing import Annotated, Any, Generic, TypeVar

from langchain_core.tools import BaseTool, InjectedToolArg
from pydantic import BaseModel, Field

from common.schemas import BaseToolConfig, ChannelConfig, ToolTypes
from statgpt.schemas import ToolArtifact


class ToolRegistry(MutableMapping):
    def __init__(self):
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


_TOOL_IMPLEMENTATIONS = ToolRegistry()


class ToolArgs(BaseModel):
    # injected tool argument. set in the code, not by the LLM.
    # LLM can't set this field because it's not added to the tool schema shown to LLM.
    # `inputs` is used to pass execution context from Supreme Agent to the tool.
    inputs: Annotated[dict, InjectedToolArg] = Field()


ToolConfigType = TypeVar('ToolConfigType', bound=BaseToolConfig)


class StatGptTool(BaseTool, Generic[ToolConfigType]):
    response_format: str = "content_and_artifact"

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
    def stage_name(self) -> str:
        """Return the stage name of calling this tool."""
        if name := self._tool_config.details.stages_config.tool_call_name:
            return name

        tool_name = self.name.replace('_', ' ')
        return f"Calling {tool_name} tool"

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
    def from_config(tool_config: ToolConfigType, channel_config: ChannelConfig) -> 'StatGptTool':
        cls = _TOOL_IMPLEMENTATIONS[tool_config.type]

        return cls(
            tool_config=tool_config,
            channel_config=channel_config,
            name=tool_config.name,
            description=tool_config.description,
            args_schema=cls.get_args_schema(tool_config),
        )
