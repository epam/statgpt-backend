from fastmcp import FastMCP

from statgpt.admin.mcp.prompts import mcp_prompts
from statgpt.admin.mcp.tools import mcp_tools

mcp = FastMCP(
    name="StatGPT MCP",
    instructions="This server provides tools to assist in the exploration and processing of new datasets.",
    providers=[mcp_prompts, mcp_tools],
)
