### Local setup
To locally setup StatGPT MCP for Cursor do the following:

1) Execute `make statgpt_mcp` or `make statgpt_admin` (first command will setup local mcp server / second command will setup admin app with mounted mcp as part of it)

2) Go to Cursor settings, Tools and MCP bar and click add New MCP Server and define this MCP:
```json
{
  "mcpServers": {
    "StatGPT MCP": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```
3) Then look at the MCP status it should be green (try to disable/enable mcp button if status not changing to green for long)
