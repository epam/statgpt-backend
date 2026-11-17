# data-query-mcp-response Specification

## Purpose

Defines what the Data Query tool returns to an MCP client (response version 3): one result split by reader, with the outcome and the queries in `structuredContent` for the calling model, the same payload plus inline data resources in `content`, and one namespaced payload per client audience (the MCP-App widget, programmatic clients) in `result._meta`. Naming, auth, the out-of-scope guardrail and the generic structured-to-text fallback belong to `mcp-tool-exposure`; the agent-facing response belongs to `data-query-agent-response`; the discovery block appended to `content` is specified in `discovery-search`. `statgpt/app/MCP_DATA_QUERY_RESPONSE.md` is the consumer reference with the full field list and an example per status.

## Requirements

### Requirement: Structured content is the model's payload
`structuredContent` SHALL be the complete payload for the calling model, validated against the tool's advertised output schema, with null fields omitted. It SHALL NOT carry the SDMX query model, the Python snippet, the companion tool names or the response version: those are for clients and go to `result._meta`.

#### Scenario: Null fields are omitted
- **WHEN** a query returned no data
- **THEN** its record in `structuredContent` has no `seriesCount` key

#### Scenario: No client wiring for the model
- **WHEN** a query returns data
- **THEN** `structuredContent` has no `pythonCode`, `tools` or `version` key

### Requirement: Content block order
`content` SHALL hold, in order: the `structuredContent` serialized as JSON text, then the inline data resources, then the discovery datasets block as a text block of its own when there is one. The rendered human-readable pipeline response SHALL NOT be a `content` block.

#### Scenario: Executed query with a discovery block
- **WHEN** an MCP client calls Data Query with `GDP of France since 2020`, the query returns data, `mcpResources` is left at its defaults and the discovery lookup finds a relevant record
- **THEN** `content` holds the JSON text of `structuredContent`, one `text/csv` resource for the dataset, and the discovery block, in that order

### Requirement: Query id
Every reported query SHALL carry a `queryId` of `dq_` plus 10 hex characters, hashed from the canonical query (dataset URN and filters, order-insensitive) plus the UTC date its data was received. The same query on the same UTC date SHALL get the same id, and on another date a different one, since it may return different data. A query that never ran SHALL get an id from the query alone.

#### Scenario: Same query, same day
- **WHEN** the same query on `IMF.RES:WEO(9.0.0)` is executed twice on 2026-09-23 UTC
- **THEN** both responses report the same `queryId`, e.g. `dq_333e6e65fc`

#### Scenario: Same query, another day
- **WHEN** the same query is executed again on 2026-10-23 UTC
- **THEN** it gets a different `queryId`

#### Scenario: Filter order does not matter
- **WHEN** two queries differ only in the order of their filters or of a filter's values
- **THEN** they get the same `queryId` on the same date

### Requirement: Query id joins the surfaces
A query's `queryId` SHALL be the same in `structuredContent`, in both `_meta` payloads and as the file stem of its resource URIs, so a client can join a table to the query that produced it.

#### Scenario: Join by id
- **WHEN** a query on `IMF.RES:WEO(9.0.0)` with id `dq_333e6e65fc` returns data
- **THEN** its CSV resource URI ends in `/dq_333e6e65fc.csv` and both `_meta` payloads list the query with `queryId` = `dq_333e6e65fc`

### Requirement: Response version
Both `_meta` payloads SHALL carry `version: 3`, one number for the whole response, bumped together. `structuredContent` SHALL NOT carry a version: the model cannot act on it.

#### Scenario: Version in the payloads
- **WHEN** a result carries both `_meta` payloads
- **THEN** each has `version` = 3 and `structuredContent` has no `version` key

### Requirement: Outcome fields
`structuredContent` SHALL carry `status` (`data_available`, `executed_no_data`, `failed`, `not_executed`, `invalid_time_period`, `dataset_selection_required`, `missing_dimensions` or `no_data`) and a `message` telling the model what to know about the outcome. When queries were executed (`data_available`, `executed_no_data`, or `failed` after execution) and `mcpStructuredContent.executedAt` is on, it SHALL also carry `executedAt`, the ISO 8601 execution time, in UTC on the MCP path.

#### Scenario: Executed query
- **WHEN** a query returns data
- **THEN** `status` is `data_available` and `executedAt` is set

### Requirement: Message resolution
The `message` SHALL use the tool's `*McpOnly` message, falling back to its non-MCP twin, then to a built-in text where one exists; an empty `*McpOnly` message counts as unset. Executed statuses use `dataQueryExecutedMcpOnly` / `dataQueryExecutedAgentOnly` with no default; `invalid_time_period` uses `invalidTimePeriodMcpOnly` / `invalidTimePeriod`; `no_data` uses `noDataMcpOnly` / `noData` or the country variants. `missing_dimensions` carries the question asked about the missing dimensions.

#### Scenario: MCP message configured
- **WHEN** a query returns data and the tool sets both `dataQueryExecutedAgentOnly` and `dataQueryExecutedMcpOnly`
- **THEN** `message` is the `dataQueryExecutedMcpOnly` text

#### Scenario: Only the non-MCP message configured
- **WHEN** the tool sets only `invalidTimePeriod` and every built query's period is out of range
- **THEN** `message` is the `invalidTimePeriod` text

#### Scenario: Nothing configured for an executed query
- **WHEN** a query returns data and neither executed message is set
- **THEN** `structuredContent` has no `message` key

#### Scenario: Dataset selection message
- **WHEN** status is `dataset_selection_required` and `multipleDatasetsMcpOnly` is set
- **THEN** `message` is the built-in instruction on how to proceed followed by the `multipleDatasetsMcpOnly` text

### Requirement: Executed and constructed queries
`structuredContent.queries` SHALL list one record per dataset query: the executed queries, with `executed: true` and in response order, for `data_available`, `executed_no_data` and `failed`; the constructed queries, with `executed: false`, for `not_executed` and `invalid_time_period`. For `no_data`, `missing_dimensions` and `dataset_selection_required`, `queries` SHALL be empty.

#### Scenario: Post-processing disabled
- **WHEN** status is `not_executed`
- **THEN** every query record has `executed` = false and no `execution`

### Requirement: Invalid time period reports the constructed queries
`invalid_time_period` occurs only with `timePeriodStrategy: AFTER` (the default `BEFORE` never reports it), when no query is valid and at least one was rejected for its period. Every constructed query SHALL then carry an `invalidity` with an `explanation` and a `reason`; a rejected one SHALL have reason `invalid_time_period` and the `rejectedPeriod` (rejected bound, requested value, available period). A rejected period SHALL NOT be applied, so `requestedPeriod` SHALL NOT carry it.

#### Scenario: Rejected start period
- **WHEN** a query asks for data from 2030 and the dataset ends in 2025
- **THEN** the query's `invalidity` has reason `invalid_time_period`, `rejectedPeriod.rejectedBound` = `startPeriod`, `requestedValue` = 2030 and `availablePeriod.endPeriod` = 2025, and `requestedPeriod` does not contain 2030

### Requirement: Invalid time period with a query missing dimensions
With status `invalid_time_period`, a constructed query that is invalid because it does not specify every required dimension SHALL have `invalidity.reason` = `missing_dimensions` and `invalidity.missingDimensions`: the dimensions in the bounded shape of the top-level `missingDimensions` (see "Codelist-derived values are bounded").

#### Scenario: Another dataset misses a dimension
- **WHEN** `timePeriodStrategy` is `AFTER`, the query on `ECB:EXR` is rejected for its period and the query on `IMF:ER` does not specify its required `CURRENCY` dimension
- **THEN** status is `invalid_time_period`, the `ECB:EXR` record has `invalidity.reason` = `invalid_time_period`, and the `IMF:ER` record has `invalidity.reason` = `missing_dimensions` with `CURRENCY` in `invalidity.missingDimensions`

### Requirement: Dataset selection and missing dimensions
For `dataset_selection_required`, each entry of `candidateDatasets` SHALL carry the dataset URN as `id`, its name, and the constructed `query` that would run against it. For `missing_dimensions`, `missingDimensions` SHALL carry the dataset URN and the dimensions a follow-up query must specify.

#### Scenario: Several datasets match
- **WHEN** the query matches `ECB:EXR` and `IMF:ER` and the tool asks to clarify multiple datasets
- **THEN** `candidateDatasets` lists both, each with the `query` that would run against it, and `queries` is empty

### Requirement: Filter values are complete
Each query record SHALL list one filter per filtered dimension, with its id, display name when known, operator, `isIndicator`, `isDefault` (applied because the user did not specify it) and `values` as id/name pairs. `values` SHALL be complete, with `valueCount` equal to its length: it is what the query asked for, already bounded by the request. A dimension without a filter SHALL NOT be listed: all of its values are included.

#### Scenario: Twenty-five indicators
- **WHEN** a query filters on 25 indicators
- **THEN** that filter's `values` lists all 25 and `valueCount` is 25

### Requirement: Time period is reported apart from filters
The time period SHALL NOT be a filter in `structuredContent`. It SHALL be reported once as `requestedPeriod` (`startPeriod` / `endPeriod`, named after the SDMX REST parameters, with `isDefault` when the dataset's default period was applied), and for a query that returned data also as `factualPeriod`, the period the returned data covers.

#### Scenario: Default period
- **WHEN** the user asks `GDP of France` without a period and the dataset's default period is applied
- **THEN** `requestedPeriod.isDefault` is true and no filter has the time period dimension id

### Requirement: Codelist-derived values are bounded
Values taken from a codelist rather than from the query, which can hold thousands of entries, SHALL be reported in `structuredContent` as `totalValues` plus at most 10 `sampleValues` (all of them when there are at most 10, otherwise the first 10). This applies to `missingDimensions` and to `invalidity.missingDimensions`. The full lists SHALL stay in the MCP-App payload.

#### Scenario: Missing reference area
- **WHEN** a query misses the reference area dimension with 196 available values
- **THEN** `missingDimensions.dimensions[0]` has `totalValues` = 196 and 10 `sampleValues`, while the MCP-App payload lists all 196

### Requirement: Execution result per query
Each executed query SHALL carry `execution.result`: `data_received` (rows, fully parsed), `partially_parsed` (rows, part failed to parse), `request_failed` (no rows, request failed), `parsing_failed` (no rows, parsing failed fully or partly) or `no_data` (no rows otherwise). Every result but `data_received` SHALL carry a `reason` and an `advice`. `seriesCount` SHALL be reported only for a query that returned data.

#### Scenario: Zero rows with a partial parse failure
- **WHEN** a response has no rows and its parsing partly failed
- **THEN** its `execution.result` is `parsing_failed` and the response status is `failed`, not `executed_no_data`

### Requirement: Widget-aware advice
The advice for `partially_parsed` and `parsing_failed` SHALL tell the model that the user can still see the data in the widget only when the tool binds a widget through `mcp_app_resource_uri`, because the widget fetches the data itself; otherwise it SHALL NOT mention a widget.

#### Scenario: No widget
- **WHEN** a response returns rows that only partly parsed and the tool has no `mcp_app_resource_uri`
- **THEN** `execution.result` is `partially_parsed` and the advice does not mention a widget

### Requirement: Structured content field toggles
The tool's `details.mcpStructuredContent` block SHALL toggle optional `structuredContent` fields: `executedAt`, `provider` and `datasetUrl`, reported by default, and `isOfficial`, not reported by default since not every channel marks its official datasets. When on, `isOfficial` SHALL be reported on query records and candidate datasets.

#### Scenario: Official mark off by default
- **WHEN** `mcpStructuredContent` is not configured
- **THEN** no query record or candidate dataset carries `isOfficial`, and executed query records carry `provider` and `datasetUrl` when known

### Requirement: Data explorer link policy
`details.mcpStructuredContent.dataExplorerUrl` SHALL be `always` (default), `only_when_no_data` or `never`, and SHALL decide per dataset response whether its executed query record carries the data explorer deep link. The same policy SHALL govern the links in the MCP-App `message`. `explorerLink` SHALL NOT affect MCP results, and a leftover `explorerLink.mcp` key SHALL be ignored.

#### Scenario: Link only when there is no data
- **WHEN** `dataExplorerUrl` is `only_when_no_data` and a query on two datasets returns rows for `ECB:EXR` and none for `IMF:ER`
- **THEN** only the `IMF:ER` record carries `dataExplorerUrl`

#### Scenario: Legacy key
- **WHEN** a tool config sets `explorerLink.mcp: never` and no `mcpStructuredContent`
- **THEN** executed query records still carry `dataExplorerUrl`

### Requirement: Client payload ignores the link policy
The client payload SHALL carry the data explorer deep link of every executed query whatever `dataExplorerUrl` is. The policy exists because the model copies links into its answer, and a programmatic client is not that model.

#### Scenario: Policy `never`
- **WHEN** `dataExplorerUrl` is `never`, the client payload is enabled and a query returns data
- **THEN** the `structuredContent` record has no `dataExplorerUrl` and the client payload's query record carries it

### Requirement: `_meta` namespace
Each `_meta` payload SHALL be keyed `{namespace}/mcp-app` or `{namespace}/client`, where the namespace is `details.mcpMeta.namespace` (default `statgpt.dialx.ai`, `$env:{VAR}` supported). The namespace SHALL be resolved and validated at config load as a reverse-DNS name not reserved by the MCP specification (`mcp`, `mcp.*`, `modelcontextprotocol.io`).

#### Scenario: Custom namespace
- **WHEN** `mcpMeta.namespace` is `$env:{MCP_META_NAMESPACE}` and the variable is `stats.example.org`
- **THEN** the payload keys are `stats.example.org/mcp-app` and `stats.example.org/client`

#### Scenario: Reserved namespace
- **WHEN** `mcpMeta.namespace` is `mcp.tools`
- **THEN** the channel config is rejected

### Requirement: `_meta` audiences are toggled per tool
The MCP-App payload SHALL be carried exactly when the tool binds a widget through `mcp_app_resource_uri`, with no toggle of its own, since nothing else can render it; it SHALL keep null fields so its shape does not change with the outcome. The client payload SHALL be carried when `details.mcpMeta.client.enabledStr` is true (default false) and SHALL omit null fields. `_meta` SHALL be omitted when neither applies.

#### Scenario: No widget and client payload off
- **WHEN** the tool has no `mcp_app_resource_uri` and `mcpMeta.client` is left at its default
- **THEN** the result has no `_meta`

#### Scenario: Widget only
- **WHEN** the tool binds `ui://statgpt/data-widget.html` and `mcpMeta.client` is off
- **THEN** `_meta` holds only `statgpt.dialx.ai/mcp-app`

### Requirement: MCP-App payload contents
The MCP-App payload SHALL carry the `status`; a `message` with the pipeline's full response text; the SDMX query model of each query (dataset URN, filters including the time period, metadata, `sdmx1Source`, `disabled`) with its `queryId`, for the executed statuses and `not_executed`; the full `candidateDatasets` or `missingDimensions` for their statuses; a reproducible sdmx1 `pythonCode` only when there are queries; and `tools.sdmxProxy`, the channel's SDMX proxy tool name.

#### Scenario: Failure before any query
- **WHEN** the pipeline ends with status `failed` and no query
- **THEN** the MCP-App payload has empty `queries` and `pythonCode` null

### Requirement: Client payload contents
The client payload SHALL carry the `status` and, per query, the `queryId` and dataset URN; for executed queries also the dataset name, the data explorer deep link, the dataset link, the URIs of the resources `content` carries for that query and `seriesCount` when it returned data; for `not_executed` queries the dataset link.

#### Scenario: Resource URIs match content
- **WHEN** CSV and Markdown resources are both enabled and a query returns data
- **THEN** the query's `resourceUris` lists the `.csv` and the `.md` URI exactly as the resources in `content` have them

### Requirement: Inline data resources
Per dataset response with rows, in response order, `content` SHALL carry a `text/csv` resource when `details.mcpResources.csv.enabledStr` is true (default true) and a `text/markdown` resource when `details.mcpResources.markdownTable.enabledStr` is true (default false); a response without rows SHALL get none. Both SHALL carry the values the model reads, transformed as the dataset config sets.

#### Scenario: Markdown only
- **WHEN** `mcpResources.csv.enabledStr` is `False` and `mcpResources.markdownTable.enabledStr` is `True`
- **THEN** each dataset with rows gets one `text/markdown` resource and no CSV resource

#### Scenario: Empty response
- **WHEN** a query returns no rows
- **THEN** `content` carries no resource for it and its client `resourceUris` is empty

### Requirement: Resource URIs
A resource URI SHALL be `statgpt://data_query/{dataset}/{queryId}.csv` or `.md`, with the dataset path URL-encoded. A response that carries no query SHALL use its UTC timestamp as the stem instead.

#### Scenario: Default CSV resource
- **WHEN** `mcpResources` is not configured and a query on `IMF.RES:WEO(9.0.0)` returns data
- **THEN** `content` carries one resource, `statgpt://data_query/IMF.RES%3AWEO%289.0.0%29/dq_333e6e65fc.csv`, with MIME type `text/csv`

### Requirement: CSV for machine reading
The CSV resource SHALL be observation-level, one row per observation, for the model and programmatic readers. Showing the data to the user is the Markdown table's job (see "Markdown table for the user").

#### Scenario: Two periods
- **WHEN** a query returns one series with values for 2023 and 2024
- **THEN** the CSV resource has two data rows, one per observation

### Requirement: Markdown table for the user
The Markdown resource SHALL be one table per dataset headed by the dataset name, with time periods as columns, component display names as headers (the coded id kept when the name is blank or would collide), code columns collapsed into their names, no row cap, no rounding of its own, pipes and line breaks in values neutralized, and `NA` for missing values. It SHALL be annotated `audience: ["user"]`.

#### Scenario: Display-name headers
- **WHEN** a Markdown table is built for a dataset whose `REF_AREA` dimension is named `Reference area`
- **THEN** the table has a `Reference area` column holding the area names and no separate code column

### Requirement: Markdown table needs a model instruction
Because clients may ignore the `audience` annotation, a channel that enables the Markdown table SHALL also set `dataQueryExecutedMcpOnly` to tell the model to reproduce the table for the user; the response itself SHALL NOT add such an instruction.

#### Scenario: Instruction comes from config
- **WHEN** `markdownTable` is enabled and `dataQueryExecutedMcpOnly` is not set
- **THEN** the Markdown resource is still carried, and `structuredContent.message` carries no instruction about it
