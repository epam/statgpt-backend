# available-datasets-tool Specification

## Purpose

Lists the datasets a channel can query, with their providers and channel-wide totals, for the agent (as text) and for MCP clients (as structured content), optionally counting the channel's metadata-only (Grade C) datasets and carrying each dataset's links to programmatic clients.

## Requirements

### Requirement: Lists the caller's queryable datasets
The Available Datasets tool (`type: AVAILABLE_DATASETS`) SHALL take no arguments and SHALL list the channel's datasets that have a completed version, are online, and that the caller's auth context can access, the same list for the agent tool and the MCP tool. A dataset the caller cannot access SHALL NOT be listed or counted.

#### Scenario: Two interfaces, one list
- **WHEN** channel `alpha` has datasets `IMF:CPI(1.0.0)` and `ECB:EXR(1.0)` and the agent and an MCP client call the tool as the same user
- **THEN** both results describe exactly these two datasets

### Requirement: MCP dataset records
The MCP tool SHALL return its whole result as structured content holding `providers`, `datasets`, `totalDatasets`, `totalIndicators` and `totalAgencies`, with optional fields omitted when unknown. Each `datasets` record SHALL carry `id` (the dataset URN, the id the dataset-structure, availability and data-query tools accept), `name`, `description` (the citation's description when set, else the dataset's), `provider` (the citation's provider), `lastUpdated`, `url` and `numberOfIndicators`.

#### Scenario: Record keyed by URN
- **WHEN** an MCP client calls the tool and dataset `IMF:CPI(1.0.0)` has a citation with provider `IMF` and no last-updated date
- **THEN** its record has `id` = `IMF:CPI(1.0.0)` and `provider` = `IMF`, and has no `lastUpdated` key

### Requirement: MCP providers and totals
`providers` SHALL list each distinct citation provider with its `datasetCount`, sorted by name. `totalDatasets` SHALL be the number of listed datasets and `totalAgencies` the number of distinct provider agencies across them (a citation's provider agency names, falling back to its provider).

#### Scenario: Provider counts
- **WHEN** the channel lists `IMF:CPI(1.0.0)` and `IMF:BOP(1.0.0)` from provider `IMF` and `ECB:EXR(1.0)` from provider `ECB`
- **THEN** `providers` is `[{"name": "ECB", "datasetCount": 1}, {"name": "IMF", "datasetCount": 2}]` and `totalDatasets` is 3

### Requirement: Indicator counts are opt-in
The tool SHALL report indicator counts only when `details.include_indicator_count` is true (default false): then each MCP record of a dataset with indexed indicators SHALL carry `numberOfIndicators`, and the result `totalIndicators`, their sum; otherwise both SHALL be omitted. A dataset without indexed indicators SHALL carry no count. The agent text SHALL show the counts under the same setting.

#### Scenario: Counts off
- **WHEN** `include_indicator_count` is false
- **THEN** no record has `numberOfIndicators` and the result has no `totalIndicators`

### Requirement: Dataset last-updated date
`lastUpdated` SHALL be resolved live on each call through the dataset's provider-specific source (for SDMX datasets the citation's last-updated value; for providers configured with `updatedAt` property sources, the first source that yields a parsable date) and SHALL be reported as an ISO 8601 date. A dataset whose date cannot be resolved or parsed SHALL be listed without it.

#### Scenario: Unparsable date
- **WHEN** a dataset's citation last-updated value is `sometime in May`
- **THEN** its record has no `lastUpdated`, and the dataset is still listed

### Requirement: `mcpMeta` configuration
The tool config SHALL accept `details.mcpMeta` with `namespace` (default `statgpt.dialx.ai`, `$env:{VAR}` supported) and `client.enabledStr` (default `False`). The namespace SHALL be validated at config load as a reverse-DNS name and rejected when it is `mcp`, `modelcontextprotocol.io` or starts with `mcp.`, names the MCP specification reserves. The agent tool SHALL ignore `mcpMeta`.

#### Scenario: Off by default
- **WHEN** a tool config has no `mcpMeta` block
- **THEN** the MCP result has no `_meta`

#### Scenario: Reserved namespace
- **WHEN** `mcpMeta.namespace` is `mcp.example`
- **THEN** the channel config is rejected as invalid

### Requirement: Client links in `_meta`
When `mcpMeta.client` is enabled, the MCP result SHALL carry `_meta["{namespace}/client"]` with `datasets`: one record per listed dataset, in `structuredContent.datasets` order, each with `id` (the same URN), `dataExplorerUrl` and `citationUrl`; otherwise it SHALL carry no `_meta`. Each link SHALL be resolved on its own, whichever one the record's single `url` uses (per the data source's `use_data_explorer_for_dataset_url`), and omitted when unresolved.

#### Scenario: Both links
- **WHEN** `mcpMeta.client.enabledStr` is `True` with the default namespace and dataset `IMF:CPI(1.0.0)` has a data explorer base URL and a citation URL
- **THEN** `_meta["statgpt.dialx.ai/client"].datasets` contains `{"id": "IMF:CPI(1.0.0)", "dataExplorerUrl": "...", "citationUrl": "..."}`

#### Scenario: Missing link
- **WHEN** the client payload is enabled and dataset `ECB:EXR(1.0)` has a citation URL but no data explorer base URL on the dataset or its data source
- **THEN** its client record has `id` and `citationUrl` and no `dataExplorerUrl` key

### Requirement: Client links stay out of the structured content
The MCP tool SHALL NOT add the client links to `structuredContent`, whose records keep a single `url`: the model reads the structured content, and the links are meant for programmatic clients.

#### Scenario: Structured content unchanged
- **WHEN** the client payload is enabled
- **THEN** no `structuredContent.datasets` record has `dataExplorerUrl` or `citationUrl`

### Requirement: Counting metadata-only datasets is opt-in
The tool SHALL count the channel's metadata-only (Grade C) datasets only when `details.include_metadata_only_datasets` is true (default false) and the channel's discovery lookup is available (see `discovery-search`), because only that lookup refers users to them. The counts SHALL be channel-wide and SHALL NOT depend on the caller. The same decision SHALL drive both the result and the MCP output schema the tool advertises.

#### Scenario: Lookup disabled
- **WHEN** `include_metadata_only_datasets` is true and the channel's `discoveryDatasets.enabled` is false
- **THEN** the tool counts no metadata-only datasets, and the MCP tool advertises and returns the plain shape

### Requirement: Metadata-only datasets are counted, not listed
When counting them, the tool SHALL count the channel's published discovery records (status `INDEXED` or `OUTDATED`, see `discovery-dataset-records`) per agency, channel-wide, spellings differing only in case counting as one. Each agency's count SHALL be added to the provider whose name matches it exactly (or as a new provider) and to `totalDatasets`; each agency counts towards `totalAgencies`. `totalIndicators` SHALL stay queryable-only. They SHALL NOT be listed: a channel can have hundreds.

#### Scenario: Merged counts
- **WHEN** the tool counts metadata-only datasets, the channel lists 2 queryable datasets from provider `IMF`, and has 3 published discovery records from agency `IMF` and 1 from agency `OECD`
- **THEN** `providers` is `IMF` with `datasetCount` 5 and `OECD` with 1, `totalDatasets` is 6, and `datasets` holds the 2 queryable records

#### Scenario: Unpublished records are not counted
- **WHEN** a discovery record of the channel has indexing status `NEW` or `FAILED`
- **THEN** it is not counted

### Requirement: Metadata-only note
When at least one metadata-only dataset is counted, the MCP result SHALL start with a `note` saying that some counted datasets are metadata-only, are not listed in `datasets`, cannot be passed to the dataset-structure, availability or data-query tools, appear in data query responses with a link to their official source, and that `totalIndicators` covers queryable datasets only. With none counted, `note` SHALL be omitted.

#### Scenario: Note present
- **WHEN** the tool counts 4 metadata-only datasets
- **THEN** the MCP result's first key is `note`

#### Scenario: No metadata-only datasets
- **WHEN** the tool counts metadata-only datasets and the channel has no published discovery records
- **THEN** the result has no `note`

### Requirement: Advertised MCP output schema follows the configuration
The MCP tool SHALL advertise an output schema with a leading optional `note` and totals described as covering queryable and metadata-only datasets only when it counts metadata-only datasets. Otherwise it SHALL advertise and return the plain shape without `note`, so a channel without the feature does not advertise a field it never sends.

#### Scenario: Feature off
- **WHEN** `include_metadata_only_datasets` is false
- **THEN** the tool's advertised output schema has no `note` property

### Requirement: Agent text output
The agent tool SHALL render the datasets as localized text grouped by provider (datasets without a provider under an unknown provider), official datasets first and then by name, each provider headed by its dataset count, and the whole list headed by statistics. `details.version` (default `short`) SHALL select the per-dataset detail: `short` lists the name and URN, `full` adds the description and a citation with provider, last-updated date and link.

#### Scenario: Full version
- **WHEN** `version` is `full`
- **THEN** each dataset entry carries its description and citation

### Requirement: Agent statistics header
`details.stats_header_format` (default `totals`) SHALL select the agent text header: `totals` gives the total datasets, total providers and, when counted, total indicators; `agencies` gives one sentence with the dataset count, the indicator count when counted, and the number of agencies with up to three of them named.

#### Scenario: Agencies header
- **WHEN** `stats_header_format` is `agencies` and the datasets come from 5 agencies
- **THEN** the header names 3 of them followed by "and others"

### Requirement: Agent text with metadata-only datasets
When metadata-only datasets are counted, the agent text SHALL carry the same merged counts as the MCP result, in both header formats and per provider, and SHALL start with a localized note equivalent to the MCP `note` when at least one is counted. Providers that have only metadata-only datasets SHALL go into one `Other providers` section, each with its dataset count, sorted by count descending, instead of a heading each, since there can be hundreds of them.

#### Scenario: Provider with only metadata-only datasets
- **WHEN** the agent tool counts metadata-only datasets and agency `OECD` has 4 published discovery records and no queryable dataset
- **THEN** the text has no `OECD` provider heading and lists `OECD` with 4 datasets under `Other providers`
