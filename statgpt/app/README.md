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
| TRACK_LLM_CALL_DURATIONS       |    No    | Track and report LLM and embedding call durations per model. Durations are written to DIAL response state and shown in the debug performance stage (requires `DIAL_SHOW_DEBUG_STAGES=true`)                                                          | `true`, `false`                              | `false`             |
| DIAL_SHOW_DEBUG_ATTACHMENTS    |    No    | Whether to show the debug attachments in the chat completion responses                                                                                                                                                                               | `true`, `false`                              | `false`             |
| ENABLE_DEV_COMMANDS            |    No    | Whether to enable developer commands in chat. Some commands, such as `!show_debug_stages`, are allowed even if this environment variable is set to False. Must be disabled in the production.                                                        | `true`, `false`                              | `false`             |
| ENABLE_DIRECT_TOOL_CALLS       |    No    | Whether to allow the user to call tools directly, bypassing the `out of scope` check and `supreme agent` orchestration.                                                                                                                              | `true`, `false`                              | `false`             |
| OFFICIAL_DATASET_LABEL         |    No    | A label for official datasets to mark them for the user in the chat                                                                                                                                                                                  |                                              | `⭐`                 |
| SKIP_OUT_OF_SCOPE_CHECK        |    No    | Whether to skip the out of scope check. Gates both the chat completion guardrail and the MCP tool input guardrail.                                                                                                                                   | `true`, `false`                              | `false`             |
| CMD_OUT_OF_SCOPE_ONLY          |    No    | Whether to stop processing user query right after out-of-scope check                                                                                                                                                                                 | `true`, `false`                              | `false`             |
| CMD_RAG_PREFILTER_ONLY         |    No    | Whether to use pre-filters only for the RAG                                                                                                                                                                                                          | `true`, `false`                              | `false`             |
| DIAL_SYSTEM_USER_CONTEXT_ROLES |    No    | Comma-separated list of DIAL roles that can receive system user context when no bearer token is present. Users with these roles can access channels with `bearer_token_required=true` without providing a bearer token. Use carefully in production. |                                              |                     |
| DIAL_RAG_DEPLOYMENT_ID         |    No    | Deployment ID for the RAG                                                                                                                                                                                                                            |                                              | `dial-rag-pgvector` |
| DIAL_RAG_PGVECTOR_URL          |    No    | URL for the RAG with pgvector, only for local development                                                                                                                                                                                            |                                              |                     |
| DIAL_RAG_PGVECTOR_API_KEY      |    No    | API key for the RAG with pgvector, only for local development                                                                                                                                                                                        |                                              |                     |
| TTYD_TOOL_PLAIN_CONTENT_*      |    No    | Environment variables for the Plain Content tool to replace in the files content. Replace `*` with the variable name.                                                                                                                                |                                              |                     |
| INDICATORS_TOTAL_CACHE_TTL     |    No    | TTL in seconds for the in-process cache of the per-channel indicators total (used to substitute the `{indicators_total}` token in conversation-starter texts). The figure is non-transactional; staleness within this window is acceptable.          | integer (seconds)                            | `60`                |

## MCP Server

StatGPT can expose its tools via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing external clients (Claude Code, Cursor, etc.) to use StatGPT tools directly.

The MCP server is mounted at `/api/v1/{deployment_id}/mcp`. The `{deployment_id}` placeholder is resolved per request from the URL path and is used to look up the channel configuration.

### Tool Names and Descriptions

By default, tools are exposed via MCP under the same names and descriptions the internal agent uses. The channel configuration can customize the MCP-facing values without affecting the chat flow:

- `mcp.tool_name_prefix` — a prefix prepended to all tool names exposed via MCP (empty by default, i.e. no prefixing).
- Per tool, `mcp_name` / `mcp_description` — overrides for the name/description exposed via MCP. If unset, the tool's regular `name` / `description` is used.

```yaml
details:
  mcp:
    tool_name_prefix: "statgpt__"
  data_query:
    name: "query_data"
    mcp_name: "data_query"            # exposed via MCP as "statgpt__data_query"
    mcp_description: "Query official statistics data using natural language."
    # ...
```

### MCP Apps (UI widgets)

StatGPT can advertise an [MCP App](https://github.com/modelcontextprotocol/ext-apps) UI widget for a tool, so an MCP host renders custom UI around the tool's result. This is **disabled by default** and configured per channel.

The widget HTML is produced by a separate frontend service — **the backend stores no HTML**. On `resources/read`, the backend does a server-to-server GET against the frontend's internal endpoint, returns the body verbatim, and caches it for a short TTL.

Two pieces of configuration are required:

- `mcp.resources` — a list of resources served by the MCP server. Each entry (type `PROXIED`):
  - `uri` — the `ui://` resource URI (required), e.g. `ui://statgpt/data-widget.html`.
  - `origin` — origin the widget loads its JS/CSS/fonts from; exposed to the host as `_meta.ui.csp.resourceDomains` (required). Supports `$env:{VAR}`.
  - `html_url` — internal endpoint the backend fetches the HTML from (required). Supports `$env:{VAR}`.
  - `cache_ttl_seconds` — TTL for the in-process HTML cache (default `60`).
  - `mime_type` — MIME type reported for the content (default `text/html`).
- Per tool, `mcp_app_resource_uri` — binds the tool to a `uri` declared in `mcp.resources` (added to the tool's `_meta.ui.resourceUri`). Must reference a declared resource.

```yaml
details:
  mcp:
    resources:
      - type: PROXIED
        uri: "ui://statgpt/data-widget.html"
        origin: "https://widget.statgpt.example"
        html_url: "$env:{WIDGET_HTML_URL}"   # e.g. http://widget-internal.svc/_mcp-app/index.html
        cache_ttl_seconds: 60
  data_query:
    # ...
    mcp_app_resource_uri: "ui://statgpt/data-widget.html"
```

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

> `mcp.endpoint` must point to the StatGPT app's MCP URL with the application's `deployment_id` substituted into the path, followed by a trailing `/`: `http://<host>/api/v1/<deployment_id>/mcp/`.

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
