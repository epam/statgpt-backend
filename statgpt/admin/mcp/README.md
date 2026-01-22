# StatGPT Admin MCP Setup

Local setup guide for StatGPT Admin MCP server, allowing to onboard datasets into StatGPT.

It is meant to be used by coding agent (Cursor or Claude Code).

## Prerequisites

1. Enable MCP in `.env`:
   ```bash
   BETA_MCP_ENABLED=true
   ```

2. Install dependencies:
   ```bash
   poetry install -E beta-mcp
   ```
   or with make:
   ```bash
   make install_dev
   ```

## Start MCP Server

MCP server is a part of StatGPT admin app. To start it, you need to start the admin app.

```bash
make statgpt_admin
```

## Coding Agent Configuration

### Cursor

1. Go to **Settings** → **Tools and MCP** → **Add New MCP Server**
2. Add the following configuration:
   ```json
   {
     "mcpServers": {
       "statgpt-mcp": {
         "type": "http",
         "url": "http://127.0.0.1:8000/mcp"
       }
     }
   }
   ```
3. Verify the MCP status is green. If needed, toggle the MCP button to refresh.

### Claude Code

1) Add the MCP server via terminal:
   ```bash
   claude mcp add --transport http statgpt-mcp http://127.0.0.1:8000/mcp
   ```

2) Verify it’s connected
   ```bash
   claude mcp list
   claude mcp get statgpt-mcp
   ```
