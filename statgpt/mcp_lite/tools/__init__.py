from . import (  # noqa: F401  — registers tools on shared provider
    data_query,
    dataset,
    glossary,
    search,
)
from ._provider import mcp_tools

__all__ = ["mcp_tools"]
