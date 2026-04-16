# Chat

## Environment variables

Below are the environment variables specific to the Chat Application. Other required variables can be found in
the [common README file](../common/README.md).

| Variable                       | Required | Description                                                                                                                                                                                                                                          | Available Values                             | Default values      |
|--------------------------------|:--------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|---------------------|
| DIAL_APP_NAME                  |    No    | Name of the DIAL app                                                                                                                                                                                                                                 |                                              | `StatGPT`           |
| DIAL_SDK_LOG                   |    No    | Log level for the DIAL SDK                                                                                                                                                                                                                           | `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL` | `WARNING`           |
| DIAL_SHOW_STAGE_SECONDS        |    No    | Whether to show the stage seconds in the DIAL app                                                                                                                                                                                                    | `true`, `false`                              | `false`             |
| DIAL_SHOW_DEBUG_STAGES         |    No    | Whether to show the debug stages in the DIAL app                                                                                                                                                                                                     | `true`, `false`                              | `false`             |
| DIAL_SHOW_DEBUG_ATTACHMENTS    |    No    | Whether to show the debug attachments in the chat completion responses                                                                                                                                                                               | `true`, `false`                              | `false`             |
| ENABLE_DEV_COMMANDS            |    No    | Whether to enable developer commands in chat. Some commands, such as `!show_debug_stages`, are allowed even if this environment variable is set to False. Must be disabled in the production.                                                        | `true`, `false`                              | `false`             |
| ENABLE_DIRECT_TOOL_CALLS       |    No    | Whether to allow the user to call tools directly, bypassing the `out of scope` check and `supreme agent` orchestration.                                                                                                                              | `true`, `false`                              | `false`             |
| OFFICIAL_DATASET_LABEL         |    No    | A label for official datasets to mark them for the user in the chat                                                                                                                                                                                  |                                              | `⭐`                 |
| SKIP_OUT_OF_SCOPE_CHECK        |    No    | Whether to skip the out of scope check for the chat                                                                                                                                                                                                  | `true`, `false`                              | `false`             |
| CMD_OUT_OF_SCOPE_ONLY          |    No    | Whether to stop processing user query right after out-of-scope check                                                                                                                                                                                 | `true`, `false`                              | `false`             |
| CMD_RAG_PREFILTER_ONLY         |    No    | Whether to use pre-filters only for the RAG                                                                                                                                                                                                          | `true`, `false`                              | `false`             |
| DIAL_SYSTEM_USER_CONTEXT_ROLES |    No    | Comma-separated list of DIAL roles that can receive system user context when no bearer token is present. Users with these roles can access channels with `bearer_token_required=true` without providing a bearer token. Use carefully in production. |                                              |                     |
| DIAL_RAG_DEPLOYMENT_ID         |    No    | Deployment ID for the RAG                                                                                                                                                                                                                            |                                              | `dial-rag-pgvector` |
| DIAL_RAG_PGVECTOR_URL          |    No    | URL for the RAG with pgvector, only for local development                                                                                                                                                                                            |                                              |                     |
| DIAL_RAG_PGVECTOR_API_KEY      |    No    | API key for the RAG with pgvector, only for local development                                                                                                                                                                                        |                                              |                     |
| TTYD_TOOL_PLAIN_CONTENT_*      |    No    | Environment variables for the Plain Content tool to replace in the files content. Replace `*` with the variable name.                                                                                                                                |                                              |                     |
| STATGPT_MCP_PATH               |    No    | Path to mount the MCP server at (see [MCP Server](#mcp-server) section)                                                                                                                                                                              |                                              | `/api/v1/mcp`       |

## MCP Server

StatGPT can expose its tools via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing external clients (Claude Code, Cursor, etc.) to use StatGPT tools directly.

The MCP server is mounted at the path configured by `STATGPT_MCP_PATH` (default: `/api/v1/mcp`).

### DIAL Core Configuration

To register the MCP server in DIAL Core, two config entries are needed:

**1. Application type schema** — defines the MCP transport and endpoint:

```json
"applicationTypeSchemas": [
  {
    "$schema": "https://dial.epam.com/application_type_schemas/schema#",
    "$id": "https://dial.epam.com/application_type_schemas/<statgpt-mcp>",
    "dial:applicationTypeDisplayName": "<StatGPT MCP>",
    "dial:applicationTypeMcp": {
      "dial:endpoint": "http://<host>/<path>",
      "dial:transport": "HTTP",
      "dial:mcpConfigDelivery": "HEADER",
      "dial:forwardPerRequestKey": true
    },
    "dial:appendApplicationPropertiesHeader": false
  }
]
```

> `dial:endpoint` should point to the MCP app URL, e.g. if the app runs on port 5000 with the default path, the endpoint is `http://localhost:5000/api/v1/mcp`.

**2. Application instance** — registers the application using the schema:

```json
"applications": {
  "<statgpt-mcp-1>": {
    "displayName": "<StatGPT MCP Application>",
    "description": "<Test application for StatGPT MCP toolset>",
    "application_type_schema_id": "https://dial.epam.com/application_type_schemas/<statgpt-mcp>",
    "application_properties": {},
    "forwardAuthToken": true
  }
}
```

> Fields in `<>` are intended for updating, other fields are optionally updated.

**MCP endpoint URL pattern:** `http(s)://<dial-core>/v1/toolset/<app-name>/mcp`

For example, if the application name is `statgpt-mcp-1` and DIAL Core runs on `localhost:8080`, the MCP URL for clients is: `http://localhost:8080/v1/toolset/statgpt-mcp-1/mcp`

### Example: Claude Code Configuration

Add the following to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "statgpt": {
      "type": "streamable-http",
      "url": "http://localhost:8080/v1/toolset/statgpt-mcp-1/mcp",
      "headers": {
        "api-key": "<your-dial-api-key>"
      }
    }
  }
}
```
