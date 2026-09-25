# Data Query MCP response: migrating from version 2 to version 3

Version 3 splits the Data Query result by reader. `structuredContent` is now written for the
calling model alone; everything a client reads moved into `result._meta`, under one namespaced key
per audience. This is a breaking change with no dual-write period: a client that reads
`structuredContent` must be updated in the same release.

`content` no longer carries the rendered text response: its first block is the `structuredContent`
serialized as JSON, which is everything the model needs. The `text/csv` / `text/markdown` resources
are unchanged except for their URIs (see below), and the discovery datasets block, when there is
one, is still a text block of its own after them.

See [the response reference](MCP_DATA_QUERY_RESPONSE.md) for the full contract and an example of
every pipeline status.

## Where each version 2 field went

Version 2 carried everything in `structuredContent`:

| Version 2 (`structuredContent`) | Version 3 |
|---|---|
| `status` | `structuredContent.status`, and `_meta["{ns}/mcp-app"].status` |
| `message` | `_meta["{ns}/mcp-app"].message`; `structuredContent.message` now carries what the model is told about the outcome instead |
| `queries[]` (SDMX query model) | `_meta["{ns}/mcp-app"].queries[]`, unchanged except for the added `queryId` |
| `queries[].urn` | `_meta["{ns}/mcp-app"].queries[].urn` |
| `queries[].filters[].componentCode` | unchanged in `mcp-app` |
| `queries[].metadata`, `sdmx1Source`, `disabled` | `_meta["{ns}/mcp-app"].queries[]` |
| `pythonCode` | `_meta["{ns}/mcp-app"].pythonCode` |
| `candidateDatasets[]` (with `description`) | `_meta["{ns}/mcp-app"].candidateDatasets[]` |
| `missingDimensions` (every available value) | `_meta["{ns}/mcp-app"].missingDimensions` |
| `tools.sdmxProxy` | `_meta["{ns}/mcp-app"].tools.sdmxProxy` |
| `version: 2` | `version: 3`, in both `_meta` payloads; `structuredContent` is no longer versioned |

`{ns}` is the configured namespace, `statgpt.dialx.ai` by default.

## What is new

- **`queryId`** on every query, in `structuredContent`, in both `_meta` payloads and as the stem of
  the resource URIs. It is derived from the query plus the date it ran, so the same query on the
  same date yields the same id.
- **`structuredContent.queries[].executed`**, telling a constructed query apart from one that ran.
- **`requestedPeriod` / `factualPeriod`** (`startPeriod` / `endPeriod`), replacing the
  `TIME_PERIOD` entry in the model-facing `filters`. `mcp-app` still carries it as a filter.
- **`_meta["{ns}/client"]`**, a payload for programmatic clients: the `status` and, per query,
  the `dataExplorerUrl`, the `datasetUrl`, the `resourceUris` and the `seriesCount`. It is off by
  default - set `details.mcpMeta.client.enabledStr` to `"True"` on the channels that need it.
- **Dimension and value display names** in `structuredContent.queries[].filters[]`, so the model
  reads `United States` rather than `USA` alone.
- **What the text response told the model**, now in `structuredContent`: the `message` and
  `executedAt`, and per query the `querySummary`, `provider`, `datasetLastUpdated`, `datasetUrl`,
  `dataExplorerUrl`, the `execution` result with its reason and advice, and `isIndicator` /
  `isDefault` on the filters and the requested period. The tools hint and its example are gone
  from the model's view unless the channel keeps them in `dataQueryExecutedMcpOnly`.
- **`invalid_time_period` and `dataset_selection_required` report their queries**: the former with
  the `invalidity` that keeps each from running, the latter as `candidateDatasets[].query`.
- **`details.mcpStructuredContent`** turns `executedAt`, `provider` and `datasetUrl` off, sets
  when `dataExplorerUrl` is reported (`always`, `only_when_no_data` or `never`), and turns
  `isOfficial` on (it is off by default, since not every channel marks official datasets).
- **`details.explorerLink.mcp` moved** to `details.mcpStructuredContent.dataExplorerUrl`, which
  governs the link across the whole MCP response.

## Resource URIs

The URI stem is now the query id instead of the execution timestamp:

```
v2: statgpt://data_query/IMF.RES%3AWEO%289.0.0%29/20260918T131218Z.csv
v3: statgpt://data_query/IMF.RES%3AWEO%289.0.0%29/dq_333e6e65fc.csv
```

A client that parsed the timestamp out of the URI should read the execution time from
`structuredContent.executedAt` instead, and use the stem to match a resource to a query. A response that carries no query
(the dataset answered without one) keeps the timestamp stem.

## Checklist for a client

1. Stop parsing the text block: read the outcome from `structuredContent` (or the JSON text block
   that repeats it), and `status` from your audience's `_meta` payload.
2. Read the SDMX query model (`urn`, `filters`, `metadata`, `sdmx1Source`, `disabled`) and
   `pythonCode` from `_meta["{ns}/mcp-app"]`.
3. Join tables to queries by the `queryId` in the resource URI.
4. Confirm the namespace your deployment is configured with (`details.mcpMeta.namespace`) and that
   your audience's payload is carried: the `mcp-app` one whenever the tool binds a widget through
   `mcp_app_resource_uri`, the `client` one when `details.mcpMeta.client.enabledStr` is on.
5. Read `version` from your audience's `_meta` payload. It versions the whole response, so both
   payloads are bumped together; `structuredContent` no longer carries it.
