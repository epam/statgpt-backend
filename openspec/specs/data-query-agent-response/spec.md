# data-query-agent-response Specification

## Purpose

Defines what the Data Query tool gives the Supreme Agent and the chat user outside MCP: the tool response the model reads for each pipeline outcome, the result stage the user sees, the system message carrying the fetched data, the data attachments added at the end of the turn, the eval attachments, the data explorer link policy and the configurable messages. The discovery block appended to the response is specified in `discovery-search`; the MCP response is specified separately.

## Requirements

### Requirement: Pipeline outcome is recorded as a status
Every Data Query call SHALL record which branch produced its response as `status` on the tool state (visible in DIAL `custom_content.state`): `no_data`, `not_executed`, `dataset_selection_required`, `data_available`, `executed_no_data`, `failed`, `invalid_time_period` or `missing_dimensions`. `failed` SHALL be the default, so a response reporting it either hit an error while fetching the data or never reached a branch.

#### Scenario: Data returned
- **WHEN** the agent asks `GDP of France since 2020` and the executed query returns rows
- **THEN** the tool state carries `status` = `data_available`

#### Scenario: Pipeline never reached a branch
- **WHEN** no branch of the pipeline records an outcome
- **THEN** the tool state carries `status` = `failed`

### Requirement: Response when no query could be built
When the pipeline builds no dataset query, the status SHALL be `no_data` and the response SHALL be `messages.noData`, defaulting to `No relevant data found for the query. Try to change the query.`. The message SHALL be shown to the user in the result stage and returned to the model.

#### Scenario: Nothing found
- **WHEN** the agent asks for an indicator no dataset of the channel covers and no message is configured
- **THEN** the response is `No relevant data found for the query. Try to change the query.` and `status` = `no_data`

### Requirement: Response when a named country has no values
When the query names countries and no query can be built for them, the status SHALL be `no_data` and the response SHALL be `messages.noDataForCountry`, defaulting to `No data was found for {country_details}. Try to change the query.`. `{country_details}` SHALL be the channel's country named entity type followed by the comma-separated country names; a message without the placeholder SHALL be used as is. The message SHALL be shown in the result stage and returned to the model.

#### Scenario: Unknown country
- **WHEN** the agent asks `GDP of Atlantis` on a channel whose country named entity type is `country` and no country value matches `Atlantis`
- **THEN** the response is `No data was found for country Atlantis. Try to change the query.` and `status` = `no_data`

### Requirement: Response when several datasets match
When `clarifyIfMultipleDatasets` is true (the default) and more than one valid query is built, the queries SHALL NOT be executed and the status SHALL be `dataset_selection_required`. The result stage SHALL list the datasets with built queries, official ones first and marked with `OFFICIAL_DATASET_LABEL` (default `⭐`). The model SHALL get the same list, an instruction that no data was given and to pick a dataset or ask the user, then `messages.multipleDatasetsAgentOnly` when set.

#### Scenario: Two datasets match
- **WHEN** valid queries are built for two datasets and `clarifyIfMultipleDatasets` is true
- **THEN** no data is fetched, `status` = `dataset_selection_required`, and the model is told to choose or ask the user which dataset to use

#### Scenario: Clarification disabled
- **WHEN** valid queries are built for two datasets and `clarifyIfMultipleDatasets` is false
- **THEN** both queries are executed

### Requirement: Response when the requested period has no values
When no built query is valid and at least one was rejected because the requested time period lies outside the period the data covers, the status SHALL be `invalid_time_period`. The response SHALL be `messages.invalidTimePeriod` (by default a `## Result of query construction` section saying values exist only for a different period, then `## Constructed queries for datasets`) followed by each query with the reason it was rejected, shown in the result stage and returned to the model.

#### Scenario: Period out of range
- **WHEN** the agent asks for `GDP of France in 1850` and every query is rejected for its time period
- **THEN** `status` = `invalid_time_period` and the response lists each query with its invalid period reason

### Requirement: Response when required dimensions are missing
When no built query is valid and none was rejected for its period, the status SHALL be `missing_dimensions`. An LLM (`incompleteQueriesPrompt`) SHALL write the response from the query formatted with its missing dimensions and up to 10 example values each, streamed to the result stage and returned to the model.

#### Scenario: Missing dimension listed
- **WHEN** a query sets the country but no value for the `UNIT` dimension
- **THEN** `status` = `missing_dimensions`, the response asks about `UNIT`, and a table attachment `Unit (UNIT)` lists its available values

### Requirement: Value tables for missing dimensions
With status `missing_dimensions`, each categorical dimension that the first dataset's query has no filter for and that has available values SHALL get a table attachment titled `<dimension name> (<dimension id>)` listing the values' ids, names and descriptions.

#### Scenario: Value table attached
- **WHEN** the first dataset's query has no filter for the `UNIT` dimension
- **THEN** a table attachment `Unit (UNIT)` lists the available `UNIT` values

### Requirement: ALL selections are never missing
A dimension selected with the `ALL` operator SHALL NOT be listed as missing, even though its value list is empty.

#### Scenario: ALL operator is not missing
- **WHEN** a query is missing `UNIT` and selects every value of `SECTOR` with the `ALL` operator
- **THEN** only `UNIT` is listed under missing dimensions

### Requirement: Response when queries are executed
When at least one valid query is built and dataset clarification does not apply, the valid queries SHALL be executed and the response SHALL be `The following queries were executed:` then one block per dataset, official datasets first. Each block SHALL carry the dataset name, execution result, id, query summary, factual time period, citation, and query filters including default and `ALL` selections (shown as `*`).

#### Scenario: Two datasets executed
- **WHEN** valid queries for `IMF:CPI` (official) and `ECB:EXR` are executed with clarification disabled
- **THEN** the response lists the `IMF:CPI` block first, then `ECB:EXR`

### Requirement: Model-only tail of the executed response
The result stage SHALL show the same dataset blocks as the response. Only the model's copy SHALL then append `messages.dataQueryExecutedAgentOnly` when set, and a final line `[Data Query executed at <timestamp>]` in the configured timezone.

#### Scenario: Agent-only note and timestamp
- **WHEN** `dataQueryExecutedAgentOnly` is `Cite the exact values.` and the query is executed
- **THEN** the model's response ends with `Cite the exact values.` and `[Data Query executed at <timestamp>]`, and the result stage shows neither

### Requirement: Execution status
After execution the status SHALL be `data_available` when any dataset returned rows; otherwise `failed` when any request failed or any response failed to parse, fully or partially; otherwise `executed_no_data`. Rationale: a fetch or parse failure also yields an empty result, so the row count alone cannot tell it from a genuine empty answer.

#### Scenario: Partial fan-out success
- **WHEN** dataset `ECB:EXR` returns rows and dataset `IMF:CPI` fails
- **THEN** `status` = `data_available`

#### Scenario: Request failed
- **WHEN** the only executed query fails at the data source
- **THEN** `status` = `failed`, not `executed_no_data`

### Requirement: Execution result per dataset
Each executed dataset block SHALL state its own result: the series count when it has rows; the failure reason and advice to retry when the request failed; that parsing failed and the user can still see the data in the UI when parsing failed; otherwise that the response holds no data, with advice to try another period or other datasets or tools.

#### Scenario: Mixed results
- **WHEN** dataset `ECB:EXR` returns 3 series and dataset `IMF:CPI` fails at the data source
- **THEN** the `ECB:EXR` block reports 3 series and the `IMF:CPI` block reports the failure with advice to retry

### Requirement: Explorer link policy per surface
A dataset block SHALL carry a `View data in explorer` link after its execution result when the source provides one and the surface's policy allows it: `explorerLink.stage` (result stage) and `explorerLink.agent` (model's response), each `always` (default), `only_when_no_data` (only for a dataset with no rows, decided per dataset) or `never`. Rationale: on success the link sends users off the platform and the model copies it; on failure it is the most useful thing.

#### Scenario: Link kept for the user, hidden from the model
- **WHEN** `explorerLink.stage` is `always`, `explorerLink.agent` is `only_when_no_data` and the query returns rows
- **THEN** the result stage shows the link and the model's response does not

#### Scenario: Decided per dataset
- **WHEN** `explorerLink.agent` is `only_when_no_data`, dataset `ECB:EXR` returns rows and dataset `IMF:CPI` returns none
- **THEN** the model's response carries the link for `IMF:CPI` only

#### Scenario: Default configuration
- **WHEN** a channel sets no `explorerLink`
- **THEN** both the stage and the model's response carry the link for every dataset that has one

### Requirement: Message variants per audience
`messages.dataQueryExecutedAgentOnly` and `messages.multipleDatasetsAgentOnly` SHALL be sent to the model and never shown to the user; `noData`, `noDataForCountry` and `invalidTimePeriod` SHALL be shown to the user and sent to the model. Each message SHALL have a `*McpOnly` twin used only on the MCP path and falling back to the base message when unset; outside MCP the base message SHALL always be used.

#### Scenario: MCP twin ignored by the agent
- **WHEN** a channel sets `noData` and `noDataMcpOnly` and the agent's query finds no data
- **THEN** the agent's response is the `noData` text

### Requirement: Fetched data reaches the model in a system message
After each agent step that called Data Query, the fetched data of that step's calls SHALL be added to the agent history as one system message, responses for the same dataset merged into one block. Each block SHALL hold the data as TSV in `<DATA>` tags, in the values the dataset config transforms for the LLM, with columns prefixed `DIMENSION:` or `ATTRIBUTE:` and coded columns split into `_ID` and `_Name`, and SHALL tell the model the user sees the data in the UI and to cite exact values.

#### Scenario: One dataset fetched
- **WHEN** a Data Query call returns a table for `ECB:EXR`
- **THEN** the next system message carries that table as TSV between `<DATA>` and `</DATA>`

### Requirement: Datasets the system message cannot show as a table
A dataset whose table exceeds `toolResponseMaxCells` (default 300) cells SHALL be replaced in the system message by a note of its size, an empty one by a note that it is empty, and one that cannot be rendered by an error note; a dataset whose parsing failed SHALL be left out. A dataset whose parsing partially failed SHALL carry, inside its own block, a notice that some data is missing from the table.

#### Scenario: Too large
- **WHEN** the only dataset in the step has 450 cells and `toolResponseMaxCells` is 300
- **THEN** the system message says the data is too large instead of carrying the table

#### Scenario: Partially failed parsing
- **WHEN** `IMF:CPI` partially failed to parse and `ECB:EXR` parsed fully
- **THEN** only the `IMF:CPI` block carries the partial-parsing notice

### Requirement: Data-format disclaimer once per message
The DISCLAIMER explaining the column prefixes and coded columns SHALL appear once per system message, before the dataset blocks, and only when at least one block carries a table.

#### Scenario: Two datasets in one step
- **WHEN** two parallel Data Query calls return tables for `ECB:EXR` and `IMF:CPI`
- **THEN** one system message carries the DISCLAIMER once, followed by both tables

#### Scenario: No table, no disclaimer
- **WHEN** every dataset in the step is empty or too large
- **THEN** the system message carries no DISCLAIMER

### Requirement: Data attachments at the end of the turn
When the agent returns its final answer, each dataset with rows (merged across calls) SHALL get its enabled attachments in order: `customTable` (on, `Data: {dataset_source_id}`), `plotlyGrid` (off), `csvFile` (on), `jsonQuery` (off, `Query (JSON): {dataset_source_id}`), `pythonCode` (off), `plotlyGraphs` (on, `Graph: {figure_title}`). Zero-row datasets SHALL get none.

#### Scenario: Empty result
- **WHEN** the only executed query returns zero rows
- **THEN** no table, graph, CSV, JSON query or Python code attachment is added

#### Scenario: Default attachments
- **WHEN** a channel sets no `attachments` and `ECB:EXR` returns rows
- **THEN** the turn carries `Data: ECB:EXR`, the CSV and the graphs, and no JSON query or Python code

### Requirement: Merged Python code
With `merge_python_code` (chat configuration), one `mergedPythonCode` attachment (off by default, `Python Code`), added after all datasets, SHALL replace the per-dataset `pythonCode` attachments.

#### Scenario: Merged code
- **WHEN** `merge_python_code` is on, `mergedPythonCode` is enabled and two datasets return rows
- **THEN** the turn carries one `Python Code` attachment after both datasets' attachments and no per-dataset Python code

### Requirement: Attachment configuration
Each attachment's `enabledStr` MAY reference an environment variable, and an enabled attachment SHALL require a `name`.

#### Scenario: Enabled without a name
- **WHEN** a channel config sets `jsonQuery.enabledStr` to `True` and no `name`
- **THEN** the channel config is rejected as invalid

### Requirement: JSON queries carry a disabled flag
JSON query objects SHALL accept `disabled` (default false), meaning the query is excluded from processing. The JSON query attachment and every other JSON query the backend generates SHALL carry `"disabled": false` explicitly. The Python code generation endpoint (`POST /python-attachment`) SHALL skip disabled queries and SHALL reject a request whose queries are all disabled. JSON query attachments in incoming system messages SHALL be validated against the same model.

#### Scenario: One query disabled
- **WHEN** `POST /python-attachment` receives queries for `ECB:EXR` and `IMF:CPI` with `IMF:CPI` disabled
- **THEN** the generated code covers `ECB:EXR` only

#### Scenario: All queries disabled
- **WHEN** every query in the request has `disabled` = true
- **THEN** the request is rejected as invalid

### Requirement: Eval attachments
With `enable_debug_attachments` on (chat configuration, default `DIAL_SHOW_DEBUG_ATTACHMENTS`, false), every Data Query call that completes SHALL attach `data_query_eval_attachment_<tool_call_id>.json` with its retrieval results, plus the discovery eval attachment of `discovery-search` when the lookup ran. On the agent path they SHALL come with the data attachments at the end of the turn. A failure to emit them SHALL be logged and SHALL NOT fail the turn.

#### Scenario: Debug attachments off
- **WHEN** `enable_debug_attachments` is false
- **THEN** no eval attachment is added and the attachments storage is not contacted for them

#### Scenario: Storage unreachable
- **WHEN** on the direct tool calls path the attachments storage cannot be reached while eval attachments are uploaded
- **THEN** the answer is still returned without them

### Requirement: Eval attachments on direct tool calls
On the direct tool calls path (`ENABLE_DIRECT_TOOL_CALLS`, a request whose last message carries `tool_calls`), Data Query calls SHALL attach the same eval attachments under the same gate, and SHALL attach no data attachments: a direct caller renders the data from the tool message itself.

#### Scenario: Direct tool call
- **WHEN** debug attachments are enabled and a request directly calls Data Query with tool call id `call_1`
- **THEN** the response carries `data_query_eval_attachment_call_1.json` and no table, CSV, graph or Python code attachment

### Requirement: Pipeline errors become a failed tool message
When the Data Query pipeline raises, the agent SHALL receive a tool message with error status whose content names the tool type and the error, and the turn SHALL continue.

#### Scenario: Pipeline raises
- **WHEN** the data source client raises an unexpected error during a Data Query call
- **THEN** the agent receives `DATA_QUERY tool failed to execute. error: <error>` and can still answer
