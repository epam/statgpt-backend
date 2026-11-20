# Spec Delta

## MODIFIED Requirements

### Requirement: Data attachments at the end of the turn
When the agent's turn ends, each dataset with rows (merged across calls) SHALL get its enabled attachments in order: `customTable` (on, `Data: {dataset_source_id}`), `plotlyGrid` (off), `csvFile` (on), `jsonQuery` (off, `Query (JSON): {dataset_source_id}`), `pythonCode` (off), `plotlyGraphs` (on, `Graph: {figure_title}`). Zero-row datasets SHALL get none. The turn ends when the agent returns its final answer, reaches the iteration cap, or fails with an error after Data Query calls completed; the attachments SHALL be added in all three cases, because a later turn can answer from the saved tool responses without running the queries again (see `chat-error-handling`). A failure to add them after the turn failed SHALL be logged and SHALL NOT replace the turn's error. A turn the client cancelled SHALL get none.

#### Scenario: Empty result
- **WHEN** the only executed query returns zero rows
- **THEN** no table, graph, CSV, JSON query or Python code attachment is added

#### Scenario: Default attachments
- **WHEN** a channel sets no `attachments` and `ECB:EXR` returns rows
- **THEN** the turn carries `Data: ECB:EXR`, the CSV and the graphs, and no JSON query or Python code

#### Scenario: Rate limit after a query
- **WHEN** `ECB:EXR` returned rows and the Supreme Agent's model then hit a rate limit while writing the answer, on a streamed request
- **THEN** the failed message carries `Data: ECB:EXR`, the CSV and the graphs, then the rate-limit error

#### Scenario: Attachments fail after the turn failed
- **WHEN** the turn failed with a rate limit and adding the attachments then raises an exception
- **THEN** the exception is logged and the user still gets the rate-limit error
