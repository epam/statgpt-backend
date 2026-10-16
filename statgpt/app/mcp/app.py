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
    # Defense-in-depth: only text from an explicitly raised ToolError reaches the caller.
    # The tool adapter already funnels failures through the actionable-error taxonomy
    # (see statgpt.app.mcp.errors); masking guarantees any error escaping that path is
    # not surfaced as a bare, internals-bearing message.
    mask_error_details=True,
)
