# dataset-structure-tool Specification

## Purpose

Describes one queryable dataset's structure, its dimensions and attributes with a bounded sample of each categorical dimension's values, so the agent or an MCP client can build availability and data queries against it.

## Requirements

### Requirement: Looks up a dataset by URN
The Dataset Structure tool (`type: DATASET_STRUCTURE`) SHALL take one argument, `dataset_id`, the dataset URN in the format `agency_id:resource_id(version)` as the available-datasets tool reports it, and SHALL look it up among the datasets the caller can access on the channel. Metadata-only (Grade C) datasets SHALL NOT be found by this tool (see `available-datasets-tool`).

#### Scenario: Known dataset
- **WHEN** an MCP client calls the tool with `dataset_id` = `IMF:CPI(1.0.0)` and the caller can access that dataset
- **THEN** the result describes `IMF:CPI(1.0.0)`

### Requirement: Unknown dataset is an MCP tool error
When the MCP tool cannot find `dataset_id`, it SHALL fail with an MCP tool error naming the id, not a success-shaped result, since a successful empty result invites the model to carry on. The agent tool SHALL instead return a text message saying the dataset was not found among the available datasets.

#### Scenario: MCP not found
- **WHEN** an MCP client asks for `IMF:XYZ(1.0.0)`, which the channel does not have
- **THEN** the call fails with a tool error whose message contains `IMF:XYZ(1.0.0)`

### Requirement: Not-found hint
The MCP not-found error SHALL name the channel's available-datasets tool exactly as MCP exposes it, channel tool-name prefix included (see `mcp-tool-exposure`), as the way to list valid ids, when the channel has that tool enabled. Otherwise it SHALL state the expected URN format `agency_id:resource_id(version)`.

#### Scenario: Hint names the prefixed tool
- **WHEN** channel `alpha` has `mcp.toolNamePrefix` = `alpha__` and an enabled available-datasets tool named `Available_Datasets`, and an MCP client asks for an unknown id
- **THEN** the tool error message contains `alpha__Available_Datasets`

#### Scenario: Channel without an available-datasets tool
- **WHEN** the channel has no enabled available-datasets tool and an MCP client asks for an unknown id
- **THEN** the tool error states the format `agency_id:resource_id(version)`

### Requirement: MCP structured content is the dataset's identity and structure
The MCP tool SHALL return its whole result as structured content with `datasetId` (the URN), `name` (always present), `lastUpdated` (an ISO 8601 date resolved as `available-datasets-tool` resolves it, omitted when unknown), `dimensions` and `attributes`, and nothing else. It SHALL carry no `found` flag, since a returned result always describes a found dataset, and no provenance (description, provider, link, provider agencies), which the caller has from the available-datasets tool.

#### Scenario: Provenance is left out
- **WHEN** an MCP client gets the structure of `IMF:CPI(1.0.0)`, whose citation has a provider and a URL
- **THEN** the result has no `description`, `provider`, `url`, `providerAgencies` or `found` key

### Requirement: Component records
Each MCP dimension and attribute record SHALL carry `id`, `name`, `type` (the component's data type, e.g. `category` or `datetime`) and, when it has one, `description`.

#### Scenario: Attribute record
- **WHEN** a dataset has attribute `UNIT_MULT` of type `string` without a description
- **THEN** its record is `{"id": "UNIT_MULT", "name": "Unit multiplier", "type": "string"}`

### Requirement: Bounded sample of categorical values
Each categorical dimension record SHALL carry `totalValues` (the number of its available values), `sampleValues` (each value's query `id` and `name`: all of them when there are at most 10, otherwise a random sample of 10, so repeated calls do not always show the head of a long list) and `sampleValuesCount` (the number of values in `sampleValues`). Non-categorical dimensions and all attributes SHALL omit all three.

#### Scenario: Long code list
- **WHEN** dimension `REF_AREA` has 200 available values
- **THEN** its record has `totalValues` 200, 10 entries in `sampleValues` and `sampleValuesCount` 10

#### Scenario: Short code list
- **WHEN** dimension `FREQ` has 3 available values
- **THEN** its record lists all 3 in `sampleValues` with `sampleValuesCount` 3

#### Scenario: Time dimension
- **WHEN** a dataset has a `TIME_PERIOD` dimension of type `datetime`
- **THEN** its record has no `totalValues`, `sampleValues` or `sampleValuesCount` key

### Requirement: Agent text output
The agent tool SHALL render the dataset as localized text with its name, URN, description and a citation with provider, last-updated date and link, then its dimensions (sampled as in the MCP tool) and attributes, and a closing note telling the model not to assume anything beyond the structure shown. The citation SHALL include the provider agencies only when `details.include_provider_agencies` is true (default false); the MCP tool SHALL ignore this setting.

#### Scenario: Provider agencies on
- **WHEN** `include_provider_agencies` is true
- **THEN** the agent text lists the dataset's provider agencies, and the MCP result is unchanged
