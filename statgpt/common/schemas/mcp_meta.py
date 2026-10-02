import re
from typing import ClassVar, Self

from pydantic import AliasChoices, Field, model_validator

from statgpt.common.config.utils import replace_env

from .base import BaseYamlModel, ToggleableConfig


class McpMeta(BaseYamlModel):
    """Audience-specific payloads an MCP tool result carries in `result._meta`.

    Each payload is published under its own namespaced `_meta` key, e.g. `{namespace}/client`.
    """

    namespace_raw: str = Field(
        default="statgpt.dialx.ai",
        validation_alias=AliasChoices("namespace", "namespaceRaw"),
        serialization_alias="namespace",
        description=(
            "Reverse-DNS prefix of the `_meta` keys the result carries, as the MCP specification"
            " requires for extension keys. Supports $env:{VAR} syntax."
        ),
    )
    client: ToggleableConfig = Field(
        default_factory=lambda: ToggleableConfig(enabled_str="False"),
        description=(
            "Payload for programmatic clients. Off by default - enable it for a channel whose"
            " callers are programmatic."
        ),
    )

    _NAMESPACE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
    # Prefixes the MCP specification reserves for itself.
    _RESERVED_NAMESPACES: ClassVar[tuple[str, ...]] = ("modelcontextprotocol.io", "mcp")

    def get_namespace(self) -> str:
        return replace_env(self.namespace_raw).strip("/")

    @property
    def client_key(self) -> str:
        return f"{self.get_namespace()}/client"

    @model_validator(mode="after")
    def _validate_namespace(self) -> Self:
        # Resolve $env:{VAR} once at config-load time so a missing var or an unusable key fails
        # fast here instead of on every tool call.
        namespace = self.get_namespace()
        if not self._NAMESPACE_PATTERN.match(namespace):
            raise ValueError(
                f"Invalid `_meta` namespace {namespace!r}: expected a reverse-DNS name such as"
                " 'statgpt.dialx.ai'"
            )
        if namespace in self._RESERVED_NAMESPACES or namespace.startswith("mcp."):
            raise ValueError(f"The `_meta` namespace {namespace!r} is reserved by the MCP spec")
        return self
