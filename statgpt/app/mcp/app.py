from fastmcp import FastMCP

from statgpt.app.mcp.provider import channel_tool_provider

mcp = FastMCP(
    name="StatGPT MCP",
    instructions=(
        "This server provides tools from the StatGPT platform for querying "
        "official statistics data, searching publications, looking up glossary terms, "
        "and more. Tools are channel-specific and depend on the deployment configuration."
    ),
    providers=[channel_tool_provider],
)
