# availability-query-tool Specification

## Purpose

Reports which dimension values (codes) actually carry observations in one dataset for an empty or partial key, with a bounded list of codes per dimension and the available time range, so coverage questions are answered without running a full data query.

## Requirements

### Requirement: Arguments
The Availability Query tool (`type: AVAILABILITY_QUERY`) SHALL take `dataset_id` (the dataset URN in the format `agency_id:resource_id(version)`), `partial_query` (a list of `{dimension_id, codes}` filters, default `[]`), `codes_per_dimension` (an integer of at least 1 or null, default 10) and `include_dimensions` (a list of dimension ids or null, default null). The agent tool and the MCP tool SHALL share this argument model. An empty `partial_query` SHALL query the whole dataset.

#### Scenario: Partial key
- **WHEN** the tool is called with `partial_query` = `[{"dimension_id": "SERIES", "codes": ["51401"]}]`
- **THEN** availability is queried with `SERIES` fixed to `51401` and every other dimension open

#### Scenario: Whole dataset
- **WHEN** the tool is called with `partial_query` = `[]`
- **THEN** availability is queried across the whole dataset

### Requirement: Argument schema is valid under strict function calling
Every object in the tool's argument schema SHALL declare its properties, so `partial_query` SHALL be a list of typed filters rather than a map from dimension id to codes. OpenAI strict function calling, which the agent binds its tools with, rejects objects without declared properties, and that rejection fails every request on the channel, not only those that would use the tool.

#### Scenario: Strict schema
- **WHEN** the agent binds the tool with OpenAI strict function calling
- **THEN** the request is accepted

### Requirement: Repeated dimensions are merged
Filters that repeat a dimension SHALL be merged into one filter holding the union of their codes, in first-seen order without duplicates, before any validation or query.

#### Scenario: Repeated dimension
- **WHEN** `partial_query` is `[{"dimension_id": "REF_AREA", "codes": ["FR"]}, {"dimension_id": "REF_AREA", "codes": ["DE", "FR"]}]`
- **THEN** the query fixes `REF_AREA` to `FR` and `DE`

### Requirement: Input is validated before querying
Before querying, the tool SHALL reject the call when the dataset is not among the datasets the caller can access; when a dimension id in `partial_query` or `include_dimensions` is not a dimension of the dataset (the message lists the unknown ids, sorted, and the valid ones); or when a code given for a categorical dimension is not in its codelist (the message lists the unknown codes per dimension and says to look them up first). Codes for non-categorical dimensions SHALL NOT be checked.

#### Scenario: Unknown code
- **WHEN** a caller fixes `REF_AREA` to `XX`, which is not in its codelist
- **THEN** the call is rejected naming `XX` for `REF_AREA`, and no availability query is sent

### Requirement: How rejections are reported
The MCP tool SHALL report each rejection as an MCP tool error rather than a result, so invalid input is never silently ignored; its unknown-dataset error SHALL carry the same hint as the dataset-structure tool's (see `dataset-structure-tool`). The agent tool SHALL return the rejection message as its text result.

#### Scenario: Unknown dimension over MCP
- **WHEN** an MCP client passes `include_dimensions` = `["COUNTRY"]` for a dataset whose dimensions are `FREQ`, `REF_AREA` and `TIME_PERIOD`
- **THEN** the call fails with a tool error naming `COUNTRY` and listing `FREQ`, `REF_AREA` and `TIME_PERIOD` as valid

### Requirement: Per-dimension coverage is bounded
For each dimension in the availability result, the tool SHALL report its name, `total_available` (codes available under the query), `returned`, `truncated` (whether codes were left out) and `values`: the first codes up to the effective limit, each with its id and, for a categorical dimension, its name. The limit SHALL be `codes_per_dimension` capped by `details.hard_limit` (default 500), with null meaning `hard_limit`, so no call is unbounded.

#### Scenario: Truncated list
- **WHEN** 40 reference areas are available and `codes_per_dimension` is 10
- **THEN** `REF_AREA` reports `total_available` 40, `returned` 10, `truncated` true and 10 values

#### Scenario: No per-call limit
- **WHEN** `codes_per_dimension` is null, `hard_limit` is 500 and 800 codes are available
- **THEN** 500 codes are returned and `truncated` is true

### Requirement: Dimension filter
When `include_dimensions` is a non-empty list, only those dimensions SHALL be reported, and the time coverage only if the time dimension is among them. Otherwise every dimension in the availability result SHALL be reported.

#### Scenario: One dimension
- **WHEN** `include_dimensions` is `["REF_AREA"]`
- **THEN** the result reports only `REF_AREA` and no time coverage

### Requirement: Time coverage
When the availability result has a start or an end time period, the tool SHALL report the time dimension's id, name, `start` and `end`; otherwise the time coverage SHALL be omitted.

#### Scenario: Range known
- **WHEN** the availability result spans `2000` to `2024`
- **THEN** the time coverage reports `TIME_PERIOD` with `start` `2000` and `end` `2024`

### Requirement: Output shapes
The MCP tool SHALL return its whole result as structured content with `datasetId`, `dimensions` as a list of records each carrying its `id` with `name`, `totalAvailable`, `returned`, `truncated` and `values`, and `timeCoverage` with `dimensionId`, `name`, `start` and `end`, omitting optional fields that are unknown. The agent tool SHALL return the same data as JSON with snake_case keys, `dimensions` keyed by dimension id, and `time_coverage` omitted when absent.

#### Scenario: MCP dimension record
- **WHEN** an MCP client queries a dataset whose `FREQ` dimension has code `A` available
- **THEN** `dimensions` contains a record with `id` `FREQ` whose `values` contain `{"id": "A", "name": "Annual"}`
