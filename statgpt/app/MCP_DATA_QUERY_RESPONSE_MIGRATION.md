# Data Query MCP response: migrating from version 2 to version 3

Version 3 splits the Data Query result by reader. `structuredContent` is now written for the
calling model alone; everything a client reads moved into `result._meta`, under one namespaced key
per audience. This is a breaking change with no dual-write period: a client that reads
`structuredContent` must be updated in the same release.

`content` is unchanged - the same text block and the same `text/csv` / `text/markdown` resources,
except for their URIs (see below).

See [the response reference](MCP_DATA_QUERY_RESPONSE.md) for the full contract and an example of
every pipeline status.

## Where each version 2 field went

Version 2 carried everything in `structuredContent`:

| Version 2 (`structuredContent`) | Version 3 |
|---|---|
| `status` | `_meta["{ns}/mcp-app"].status`, `_meta["{ns}/client"].status` |
| `message` | `_meta["{ns}/mcp-app"].message`, `_meta["{ns}/client"].message` |
| `queries[]` (SDMX query model) | `_meta["{ns}/mcp-app"].queries[]`, unchanged except for the added `queryId` |
| `queries[].urn` | `_meta["{ns}/mcp-app"].queries[].urn`; in `structuredContent` the model sees `queries[].datasetUrn` |
| `queries[].filters[].componentCode` | unchanged in `mcp-app`; in `structuredContent` the model sees `queries[].filters[].dimensionId` with the dimension and value names resolved |
| `queries[].metadata`, `sdmx1Source`, `disabled` | `_meta["{ns}/mcp-app"].queries[]` |
| `pythonCode` | `_meta["{ns}/mcp-app"].pythonCode` |
| `candidateDatasets[]` (with `description`) | `_meta["{ns}/mcp-app"].candidateDatasets[]`; `structuredContent.candidateDatasets[]` carries `id` / `name` / `isOfficial` only |
| `missingDimensions` (every available value) | `_meta["{ns}/mcp-app"].missingDimensions`; `structuredContent.missingDimensions` carries `totalValues` plus up to 10 `sampleValues` per dimension |
| `tools.sdmxProxy` | `_meta["{ns}/mcp-app"].tools.sdmxProxy` |
| `version: 2` | `version: 3`, in `structuredContent` and in both `_meta` payloads |

`{ns}` is the configured namespace, `statgpt.dialx.ai` by default.

## What is new

- **`queryId`** on every query, in `structuredContent`, in both `_meta` payloads and as the stem of
  the resource URIs. It is derived from the query plus the date it ran, so the same query on the
  same date yields the same id.
- **`structuredContent.queries[].executed`**, telling a constructed query apart from one that ran.
  With `status` gone from `structuredContent`, this is what stops the model from reading an
  unexecuted query as returned data.
- **`requestedPeriod` / `factualPeriod`** (`startPeriod` / `endPeriod`), replacing the
  `TIME_PERIOD` entry in the model-facing `filters`. `mcp-app` still carries it as a filter.
- **`_meta["{ns}/client"]`**, a payload for programmatic clients: per query the
  `dataExplorerUrl`, the `datasetUrl`, the `resourceUris` and the `seriesCount`. It is off by
  default - set `details.mcpMeta.client.enabledStr` to `"True"` on the channels that need it.
- **Dimension and value display names** in `structuredContent.queries[].filters[]`, so the model
  reads `United States` rather than `USA` alone.

## Resource URIs

The URI stem is now the query id instead of the execution timestamp:

```
v2: statgpt://data_query/IMF.RES%3AWEO%289.0.0%29/20260918T131218Z.csv
v3: statgpt://data_query/IMF.RES%3AWEO%289.0.0%29/dq_333e6e65fc.csv
```

A client that parsed the timestamp out of the URI should read the execution time from the text
block instead, and use the stem to match a resource to a query. A response that carries no query
(the dataset answered without one) keeps the timestamp stem.

## Checklist for a client

1. Read `status` and `message` from your audience's `_meta` payload, not from `structuredContent`.
2. Read the SDMX query model (`urn`, `filters`, `metadata`, `sdmx1Source`, `disabled`) and
   `pythonCode` from `_meta["{ns}/mcp-app"]`.
3. Join tables to queries by the `queryId` in the resource URI.
4. Confirm the namespace your deployment is configured with (`details.mcpMeta.namespace`) and that
   your audience's payload is carried: the `mcp-app` one whenever the tool binds a widget through
   `mcp_app_resource_uri`, the `client` one when `details.mcpMeta.client.enabledStr` is on.
5. Treat `version` as the single version of the whole response: `structuredContent` and both
   `_meta` payloads are bumped together.
