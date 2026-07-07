from fastmcp import FastMCP

from statgpt.mcp_lite.timing import TimingMiddleware
from statgpt.mcp_lite.tools import mcp_tools

mcp = FastMCP(
    name="StatGPT MCP-Lite",
    instructions=(
        "Low-level, no-LLM data primitives for one StatGPT channel. "
        "Each connection is bound to the channel in its URL "
        "(`/api/v1/{channel}/mcp-lite/`); tools take no `channel` arg. "
        "Start with `list_glossary_terms` for channel vocabulary, "
        "then call `get_glossary_term` for full definitions."
    ),
    providers=[mcp_tools],
)

mcp.add_middleware(TimingMiddleware())
