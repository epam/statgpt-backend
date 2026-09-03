"""MCP interfaces of the StatGPT tools.

Importing this package registers every MCP tool class (see `base.StatGptMcpTool`); the provider
resolves a tool config to its class through `StatGptMcpTool.from_config`.
"""

from . import (  # noqa: F401
    data_query,
    datasets_meta,
    datasets_metadata_app,
    glossary,
    sdmx_query_app,
)
from .base import LangChainMcpTool, StatGptMcpTool, mcp_tool_class_for, tool_app_config

__all__ = ["LangChainMcpTool", "StatGptMcpTool", "mcp_tool_class_for", "tool_app_config"]
