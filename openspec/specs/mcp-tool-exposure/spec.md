# mcp-tool-exposure Specification

## Purpose

Exposes each channel's configured tools to external clients over the Model Context Protocol (MCP): which tools a caller sees and under which names, how calls are authorized, screened, bounded and turned into MCP results or errors, and how MCP App UI widgets are advertised and served. The content of individual tool responses belongs to the tools' own capabilities (`data-query-mcp-response`, `available-datasets-tool`, `dataset-structure-tool`).

## Requirements

### Requirement: One MCP server per channel
The chat backend SHALL serve MCP over stateless streamable HTTP at `/api/v1/{deployment_id}/mcp`, resolving the channel from `deployment_id` on every request. `tools/list` SHALL return one MCP tool per enabled tool of the channel's tool list, including `mcp_only` tools. Deep Research and discovery datasets are not in the channel's tool list, so they SHALL never be exposed over MCP. A tool that fails to build SHALL be left out of the list and logged, without failing the list.

#### Scenario: Enabled tools are listed
- **WHEN** an MCP client lists tools at `/api/v1/alpha/mcp` and channel `alpha` has `data_query` enabled and `web_search` with `enabled: false`
- **THEN** the list contains the Data Query tool and no Web Search tool

### Requirement: Unresolvable requests degrade to nothing
When the channel context of a request cannot be resolved (unknown channel, missing `deployment_id`, authentication or authorization failure), `tools/list` and `resources/list` SHALL return an empty list, and `tools/call` and `resources/read` SHALL treat the requested tool or resource as unknown. The failure SHALL be logged, not raised to the client.

#### Scenario: Unknown channel
- **WHEN** an MCP client lists tools at `/api/v1/missing/mcp` and no channel has deployment id `missing`
- **THEN** the list is empty and the failure is logged

### Requirement: Calls are authorized like chat requests
The MCP endpoint SHALL build the caller's auth context from the request headers with the channel's `bearer_token_required` flag, as the chat completion endpoint does. A request carrying a bearer token SHALL run as that user, and so SHALL any request to a channel whose `bearer_token_required` is false.

#### Scenario: Channel without the flag
- **WHEN** channel `beta` has `bearer_token_required: false` and an MCP request carries only an API key
- **THEN** the request runs as that user

### Requirement: System-user access without a bearer token
When `bearer_token_required` is true and no bearer token is present, a request whose API key has one of the DIAL roles listed in `DIAL_SYSTEM_USER_CONTEXT_ROLES` SHALL run as the system user. Any other such request SHALL be refused, so it sees an empty tool list and cannot call or read anything. Roles that cannot be resolved SHALL deny system-user access.

#### Scenario: API key with the system-access role
- **WHEN** channel `alpha` has `bearer_token_required: true`, `DIAL_SYSTEM_USER_CONTEXT_ROLES` is `system-access`, and an MCP request carries only an API key whose user has the `system-access` role
- **THEN** the tools are listed and calls run as the system user

#### Scenario: API key without the role
- **WHEN** the same channel receives an MCP request with only an API key whose user lacks the `system-access` role
- **THEN** `tools/list` returns an empty list

### Requirement: Host header is not validated
The MCP server SHALL accept any `Host` header. DNS rebinding protection defends servers bound to localhost from web pages, while StatGPT runs behind a gateway that authenticates every request; validating the host would reject every request that reaches a deployed environment through its public domain.

#### Scenario: Request through a public domain
- **WHEN** an MCP request arrives with `Host: statgpt.example.com`
- **THEN** it is processed and not rejected with `421 Misdirected Request`

### Requirement: MCP-facing tool names and descriptions
Each tool SHALL be exposed under `mcp.tool_name_prefix` followed by its `mcp_name`, or its `name` when `mcp_name` is unset, and described by its `mcp_description`, or its `description` when unset. The prefix SHALL default to the empty string, which disables prefixing. These settings SHALL affect MCP only: the Supreme Agent keeps the tool's `name` and `description`. The prefix lets a client connected to several MCP servers tell StatGPT tools from same-named ones.

#### Scenario: Prefix and name override
- **WHEN** channel `alpha` has `mcp.tool_name_prefix` = `statgpt__` and a Data Query tool with `name` = `query_data` and `mcp_name` = `data_query`
- **THEN** MCP clients see the tool as `statgpt__data_query` and the Supreme Agent still calls it `query_data`

#### Scenario: Defaults
- **WHEN** a tool has no `mcp_name` or `mcp_description` and the channel has no prefix
- **THEN** MCP clients see the tool's `name` and `description` unchanged

### Requirement: Calls resolve tools by their MCP name
`tools/call` SHALL find the tool whose MCP name (prefix plus `mcp_name` or `name`) equals the requested name. When the channel has a prefix, a requested name that does not start with it SHALL be treated as an unknown tool, and the miss SHALL be logged.

#### Scenario: Unprefixed call on a prefixed channel
- **WHEN** channel `alpha` has the prefix `statgpt__` and a client calls `data_query`
- **THEN** the call fails as an unknown tool

### Requirement: MCP-only tools
A tool with `mcp_only: true` SHALL be served over MCP and SHALL NOT be given to the Supreme Agent or described to the out-of-scope checker. The SDMX query app tool (`SDMX_QUERY_APP`) and the datasets metadata app tool (`DATASETS_METADATA_APP`) SHALL always be MCP-only, and a channel config SHALL NOT be able to turn that off, because the MCP App widget, not the model, builds and invokes their requests.

#### Scenario: SDMX query app tool stays out of the agent
- **WHEN** channel `alpha` configures an `SDMX_QUERY_APP` tool named `sdmx_proxy`
- **THEN** the Supreme Agent's tool list does not contain it and the MCP tool list does

### Requirement: MCP App fields are advertised in tool metadata
A tool's `mcp_visibility` (a list of `model` and/or `app`) and `mcp_app_resource_uri` SHALL be advertised as its `_meta.ui.visibility` and `_meta.ui.resourceUri`. When both are unset, `_meta.ui` SHALL be omitted, so the host applies the MCP Apps default `["model", "app"]`. `mcp_visibility` SHALL default to `["app"]` for the SDMX query app and datasets metadata app tools and to unset for every other tool.

#### Scenario: No MCP App fields
- **WHEN** a tool sets neither `mcp_visibility` nor `mcp_app_resource_uri`
- **THEN** its listing carries no `_meta.ui`

### Requirement: Visibility is a hint for the host
`mcp_visibility` SHALL NOT filter anything on the server: a tool with `mcp_visibility: ["app"]` SHALL stay in `tools/list` and be callable by its plain MCP name, since the MCP Apps specification makes visibility an advisory hint for the host. `mcp_visibility` is independent of `mcp_only`, which controls only the Supreme Agent.

#### Scenario: App-only tool is callable
- **WHEN** an MCP App widget calls `sdmx_proxy`, whose `mcp_visibility` is the default `["app"]`
- **THEN** the call runs, and `tools/list` lists `sdmx_proxy` with `_meta.ui.visibility` = `["app"]`

### Requirement: Tool annotations
Every MCP tool SHALL be annotated `readOnlyHint: true` and `destructiveHint: false`. Web Search and Web Search Agent tools SHALL be annotated `openWorldHint: true`, and every other tool `openWorldHint: false`, since they read only the channel's own data. The annotations are advisory hints for client UX.

#### Scenario: Web search tool
- **WHEN** a channel exposes a `WEB_SEARCH` tool
- **THEN** its listing carries `readOnlyHint: true`, `destructiveHint: false` and `openWorldHint: true`

### Requirement: Arguments are validated before anything runs
A call's arguments SHALL be validated against the tool's advertised input schema before the out-of-scope check or the tool runs. Invalid arguments SHALL be rejected with an MCP tool error (`isError`) whose message is `Invalid arguments for <tool name>: ` followed by the field-specific validation error.

#### Scenario: Missing required argument
- **WHEN** a client calls the Data Query tool without `query`
- **THEN** the call returns a tool error starting with `Invalid arguments for` and naming `query`, and the tool does not run

### Requirement: Errors surface as tool errors
A failed call SHALL return an MCP tool error, never a successful result carrying error text. A tool error a tool raises (such as an out-of-scope rejection) SHALL reach the client unchanged. A tool that reports an upstream failure (today only the SDMX query app tool, on a connection error or timeout) SHALL return its specific message. Any other exception SHALL be logged with its traceback and return `<tool name> tool failed to execute`, without internal details.

#### Scenario: Unexpected failure
- **WHEN** a tool raises an unexpected exception while running
- **THEN** the client receives the tool error `<tool name> tool failed to execute` and the traceback is logged

#### Scenario: Upstream unreachable
- **WHEN** the SDMX query app tool's backend does not respond in time
- **THEN** the client receives the tool error `The SDMX backend did not respond in time (timeout).`

### Requirement: Every call is bounded by a deadline
The out-of-scope check and the tool run of one call SHALL together be bounded by `MCP_TOOL_TIMEOUT_SECONDS` (default `240`). On expiry the downstream work SHALL be cancelled and the call SHALL return a tool error stating the limit and advising a narrower request. The default stays under the 300-second tool-call timeout of common MCP hosts, so the model gets an error it can act on instead of a host cancellation.

#### Scenario: Slow data query
- **WHEN** a Data Query call is still running after 240 seconds with the default setting
- **THEN** its downstream work is cancelled and the client receives a tool error stating the 240-second limit and suggesting fewer indicators, a shorter time range, or a single dataset

### Requirement: Free-text input is screened for scope
Before a tool runs, a non-empty free-text `query` argument of the LLM-backed tools (Data Query, Datasets Metadata, File RAG, Web Search, Web Search Agent) SHALL be screened by the channel's out-of-scope checker, the check the chat endpoint runs before the Supreme Agent, applied to that argument alone without chat history. Tools without such an argument SHALL NOT be screened. Screening SHALL run only when the channel has an `out_of_scope` config and `SKIP_OUT_OF_SCOPE_CHECK` is false.

#### Scenario: Screening disabled
- **WHEN** a channel has no `out_of_scope` config, or `SKIP_OUT_OF_SCOPE_CHECK` is true
- **THEN** MCP calls run without screening

#### Scenario: Tool without free text
- **WHEN** a client calls the Available Terms tool
- **THEN** no screening runs

### Requirement: Out-of-scope requests are rejected, fail closed
An out-of-scope request SHALL be rejected with a tool error carrying the checker's user-facing response, or `This request is out of scope. Reason: <reasoning>` when that response cannot be generated. When the check itself cannot be completed, the call SHALL be rejected with `Request blocked: the safety check could not be completed.` A tool error, rather than a result with refusal text, keeps a client from treating the refusal as data.

#### Scenario: Off-topic query
- **WHEN** channel `alpha` has an `out_of_scope` config and a client calls Data Query with `query` = `write me a poem about cats`, which the checker judges out of scope
- **THEN** the call returns a tool error with the checker's response and the pipeline does not run

#### Scenario: Checker failure
- **WHEN** the checker's LLM call fails while screening a Data Query call
- **THEN** the call returns the tool error `Request blocked: the safety check could not be completed.`

### Requirement: Output schemas match structured content
A tool whose result shape is declared SHALL advertise an `outputSchema` built from the same model and serialization (aliases, camelCase where the model uses it) as the `structuredContent` it returns, so the two cannot drift. A tool without a declared shape, such as the SDMX query app and datasets metadata app tools, SHALL advertise no output schema even when it returns structured content.

#### Scenario: Tool without a declared shape
- **WHEN** a client lists a File RAG tool
- **THEN** it has no `outputSchema` and its result is the tool's text as one text block

### Requirement: Structured-only results are also sent as JSON text
A structured-only result of a tool the model can see, whose whole payload is the structured content, SHALL omit null optional fields and SHALL also send the same payload serialized as JSON in a single text block. The datasets metadata app tool is excluded: its payload is for the widget only. Clients that read only `content`, such as LangChain-based clients that treat `structuredContent` as program data and never show it to the model, would otherwise see nothing.

#### Scenario: Structured-only result
- **WHEN** a client calls the Available Terms tool
- **THEN** the result carries the payload as `structuredContent` and the identical JSON as its one text block, and the listing advertises the matching `outputSchema`

### Requirement: Available Terms result
The Available Terms tool SHALL return the structured-only payload `{terms: [{term, domain?, source?}], count}`, with `domain` and `source` present only when the tool's `include_domain` / `include_source` is true and the glossary has a non-empty value.

#### Scenario: Domain not exposed
- **WHEN** the tool has `include_domain: false` and the glossary has 3 terms
- **THEN** the result has `count` = 3 and no record carries `domain`

### Requirement: Term Definitions result
The Term Definitions tool SHALL return the structured-only payload `{definitions: [{term, definition, domain?, source?}], notFound?}`, where `notFound` lists the requested terms the glossary lacks and is omitted when every term was found. A request with more terms than the tool's limit SHALL fetch nothing and return a tool error asking for fewer terms, so it is not read as "none of these terms exist".

#### Scenario: Unknown term
- **WHEN** a client requests definitions of `Bid price` and `Foo` and the glossary has only `Bid price`
- **THEN** the result has one entry in `definitions` and `notFound` = `["Foo"]`

#### Scenario: Over the limit
- **WHEN** the tool's limit is 10 and a client requests 11 terms
- **THEN** the call returns a tool error stating the limit of 10

### Requirement: SDMX query app passthrough
The SDMX query app tool SHALL forward a request built by the MCP App widget to the base URL configured in the tool's details, appending the caller's `path` verbatim. It SHALL return the upstream body as the text block whatever the status, with `{statusCode, contentType}` as structured content so the widget can tell success from upstream errors. `method` SHALL be `GET` (default) or `POST`.

#### Scenario: Upstream error status
- **WHEN** the backend answers `/data/ECB,EXR/D.USD.EUR.SP00.A` with HTTP 404
- **THEN** the result's text block is the 404 body and its structured content has `statusCode` = 404

### Requirement: SDMX query app input rules
`path` SHALL start with a single `/` and SHALL contain no scheme or host (a `//` prefix or `://` anywhere is rejected), so the request can reach only the configured host. A `body` SHALL be rejected with `GET` rather than silently ignored. These rules SHALL be enforced during argument validation, so a rejected call returns a field-specific `Invalid arguments` tool error and sends no request.

#### Scenario: Absolute URL in path
- **WHEN** the widget calls `sdmx_proxy` with `path` = `https://evil.example/data`
- **THEN** the call returns an `Invalid arguments` tool error naming `path` and no request is sent

#### Scenario: Body with GET
- **WHEN** the widget calls `sdmx_proxy` with `method` = `GET` and a `body`
- **THEN** the call returns an `Invalid arguments` tool error and no request is sent

### Requirement: Datasets metadata app tool
The datasets metadata app tool SHALL take no arguments and SHALL return, as structured content, the same channel datasets metadata payload as the `/metadata/datasets` service endpoint, computed for the caller's MCP auth context, so the MCP App widget gets dataset metadata inside the MCP flow while the model does not see the tool.

#### Scenario: Widget reads datasets metadata
- **WHEN** the widget calls the datasets metadata app tool on channel `alpha`
- **THEN** its structured content equals the `/metadata/datasets` response for `alpha` computed with the same auth context

### Requirement: Widget resources are configured per channel
A channel SHALL declare widget resources in `mcp.resources`, empty by default, which turns the feature off. An enabled tool's `mcp_app_resource_uri` SHALL reference a `uri` declared there, or the channel config SHALL be rejected.

#### Scenario: Unknown resource binding
- **WHEN** a channel config binds Data Query to `mcp_app_resource_uri` = `ui://statgpt/data-widget.html` and `mcp.resources` does not declare it
- **THEN** the channel config is rejected as invalid

### Requirement: Proxied resource fields
Each `mcp.resources` entry of type `PROXIED` SHALL have a `uri` starting with `ui://` and unique within the channel; an `origin` that is a bare `http`/`https` origin (no path, query or fragment); an `html_url` starting with `http://` or `https://`; `cache_ttl_seconds` (default `60`, at least 0); `mime_type` (default `text/html;profile=mcp-app`); and `prefers_border` (default `true`). `origin` and `html_url` SHALL support `$env:{VAR}` and SHALL be resolved and validated when the config loads.

#### Scenario: Origin with a path
- **WHEN** a `PROXIED` resource has `origin` = `https://widget.example.com/app`
- **THEN** the channel config is rejected as invalid

### Requirement: Widget HTML is proxied from the frontend
The backend SHALL store no widget HTML: on `resources/read` it SHALL GET `html_url` server to server, following redirects, and return the body verbatim, cached in process per `html_url` for `cache_ttl_seconds`. A failed fetch (timeout, connection error or non-2xx status) SHALL fail the read. Keeping the HTML in the separately deployed frontend avoids coupling backend releases to frontend assets.

#### Scenario: Cached read
- **WHEN** two clients read `ui://statgpt/data-widget.html` within 60 seconds with the default TTL
- **THEN** the frontend endpoint is fetched once and both receive the same HTML

#### Scenario: Frontend down
- **WHEN** the `html_url` endpoint refuses connections
- **THEN** the `resources/read` call fails

### Requirement: Widget resource metadata
Each listed resource SHALL report its `mime_type` and carry `_meta.ui.csp.resourceDomains` = `[origin]` and `_meta.ui.prefersBorder` = `prefers_border`, so the host lets the widget load its scripts and styles from the frontend origin.

#### Scenario: Resource listing
- **WHEN** channel `alpha` declares a `PROXIED` resource with `origin` = `https://widget.example.com` and default fields
- **THEN** `resources/list` reports it with MIME type `text/html;profile=mcp-app`, `_meta.ui.csp.resourceDomains` = `["https://widget.example.com"]` and `_meta.ui.prefersBorder` = true
