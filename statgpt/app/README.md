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

The MCP server is mounted at the path configured by `STATGPT_MCP_PATH` (default: `/api/v1/{deployment_id}/mcp`). The `{deployment_id}` placeholder is resolved per request from the URL path and is used to look up the channel configuration.

### DIAL Core Configuration

To expose an existing StatGPT application's tools over MCP, add an `mcp` section to its entry in DIAL Core's `applications` config. The rest of the application fields (`endpoint`, `features`, etc.) are expected to already exist.

In the example below, `statgpt-sample` is the application's `deployment_id` — the same id is embedded in both the chat completion endpoint and the MCP endpoint.

```json
{
  "applications": {
    "statgpt-sample": {
      "displayName": "StatGPT Sample",
      "displayVersion": "default",
      "description": "<description>",
      "endpoint": "http://<host>/openai/deployments/statgpt-sample/chat/completions",
      "forwardAuthToken": true,
      "features": {
        "configurationEndpoint": "http://<host>/openai/deployments/statgpt-sample/configuration"
      },
      "mcp": {
        "endpoint": "http://<host>/api/v1/statgpt-sample/mcp/",
        "transport": "http",
        "allowedTools": [],
        "configDelivery": "header",
        "forwardPerRequestKey": true
      }
    }
  }
}
```

> `mcp.endpoint` must point to the StatGPT app's MCP URL with the application's `deployment_id` substituted into the path, followed by a trailing `/`. For the default `STATGPT_MCP_PATH`, this is `http://<host>/api/v1/<deployment_id>/mcp/`.

### Example: Claude Code Configuration

Add the following to your Claude Code MCP settings (replace `<dial-core>` and `<app-name>` with your DIAL Core host and the application id, e.g. `statgpt-sample`):

```json
{
  "mcpServers": {
    "statgpt": {
      "type": "streamable-http",
      "url": "http://<dial-core>/v1/toolset/<app-name>/mcp",
      "headers": {
        "api-key": "<your-dial-api-key>"
      }
    }
  }
}
```
