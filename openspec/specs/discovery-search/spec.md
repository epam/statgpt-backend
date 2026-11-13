# discovery-search Specification

## Purpose

Shows a channel's discovery (Grade C, metadata-only) datasets that are relevant to a data query next to the Data Query result, by searching the channel's published discovery documents, letting an LLM judge keep the relevant ones, and rendering them with channel-configured templates.

## Requirements

### Requirement: Lookup runs beside every Data Query call
When the lookup is available for a channel, every Data Query call SHALL run it, for the agent and for MCP clients alike, searching with the tool's `query` argument and starting once the pipeline's chain is built, without waiting for any step such as named entity recognition. The lookup SHALL NOT be a tool the agent or an MCP client can call: `discoveryDatasets` is a channel config block outside the channel's tool list, because the Data Query tool runs the lookup, not the agent.

#### Scenario: Agent call runs the lookup
- **WHEN** the agent calls Data Query with `query` = `GDP of France since 2020` on a channel whose `discoveryDatasets.enabled` is true
- **THEN** the lookup searches the channel's discovery documents for `GDP of France since 2020` while the query pipeline runs

#### Scenario: MCP call runs the lookup
- **WHEN** an MCP client calls the Data Query tool on the same channel
- **THEN** the same lookup runs beside the pipeline

#### Scenario: No lookup tool is exposed
- **WHEN** the agent's tool list or the MCP tool list is built for a channel with a `discoveryDatasets` block
- **THEN** neither list contains a discovery datasets tool

### Requirement: Channel config gates the lookup
The lookup SHALL run only when the channel has a `discoveryDatasets` block and its `enabled` is true (the default). `enabled` SHALL gate only the chat-time lookup, never publishing or indexing, so a channel can be indexed before its records are shown to users. The block SHALL require `details.applicationId` (the Generic RAG application holding the channel's discovery documents, `$env:{VAR}` supported) and `details.templates` even when `enabled` is false.

#### Scenario: Disabled block indexes but shows nothing
- **WHEN** a channel has a `discoveryDatasets` block with `enabled: false`
- **THEN** a discovery indexing job can still publish its records, and Data Query calls run the pipeline alone with no lookup and no discovery eval attachment

#### Scenario: Missing block
- **WHEN** a channel has no `discoveryDatasets` block
- **THEN** Data Query calls run the pipeline alone

#### Scenario: Templates are required
- **WHEN** a channel config has a `discoveryDatasets` block with `enabled: false` and no `details.templates`
- **THEN** the channel config is rejected as invalid

### Requirement: Retrieval runs as the caller
Every read the lookup makes from the Generic RAG service, and every LLM call it makes, SHALL use the DIAL key of the user behind the chat turn or MCP call. Chat users SHALL therefore need access to the discovery RAG application (and the reference-area vocabulary application, when configured); a user without it gets no block and the failure is recorded.

#### Scenario: User without access to the RAG application
- **WHEN** a user whose DIAL key has no access to the discovery RAG application asks a data query
- **THEN** the search is refused by the service, no block is shown, and the eval attachment records the error

### Requirement: Search is scoped to the channel's own documents
Every discovery search SHALL carry a metadata matcher restricting it to documents whose `statgpt_channel` is the current channel, applied by the service before ranking so that `details.topN` (default 32) counts only this channel's documents. The search SHALL use `details.indexes` when set and otherwise every index the RAG application flags for hybrid search. The search SHALL NOT filter on `grade`: a channel's Grade B and Grade C records are both candidates.

#### Scenario: Shared RAG application
- **WHEN** channels `alpha` and `beta` publish to the same RAG application and a user queries channel `alpha` with `topN` = 32
- **THEN** the search matcher carries `statgpt_channel` = `alpha` in every filter entry and up to 32 of `alpha`'s documents are returned

#### Scenario: Both grades are shown
- **WHEN** channel `alpha` has published a Grade B and a Grade C record that both match the query
- **THEN** both can be candidates

### Requirement: Only discovery records become candidates
A search hit whose metadata does not form a valid discovery record SHALL be dropped before its body is downloaded. Candidates SHALL be numbered by `rank` from 1 in search order over the hits that survived, so ranks have no gaps. Each candidate's body SHALL be downloaded as its `description`; a failed download SHALL leave the description empty and the candidate SHALL still be judged.

#### Scenario: Non-record hit
- **WHEN** the search returns documents 11, 12 and 13 and document 12's metadata is not a discovery record
- **THEN** the candidates are 11 with rank 1 and 13 with rank 2, and document 12's body is never downloaded

#### Scenario: Failed download
- **WHEN** the body of document 13 cannot be downloaded
- **THEN** document 13 is judged with an empty description

### Requirement: Pre-filter configuration
The lookup SHALL try to narrow the search with a pre-filter under `details.preFilter`: `enabled` (default true), `axes` (default `reference_area`, `partner_reference_area`, `frequency`, `agency`), `referenceAreaTopN` (default 20), `llmModelConfig`, and an optional prompt per axis, given `{query}` and `{values}`, that falls back to a built-in default. Each axis SHALL be decided by its own LLM call offered only that axis's values; an axis with nothing to offer SHALL NOT be asked.

#### Scenario: Disabled pre-filter
- **WHEN** `preFilter.enabled` is false
- **THEN** no axis is asked and the search uses only the channel-scope matcher, with the fallback reason `pre-filtering is disabled for this channel`

#### Scenario: Only configured axes are asked
- **WHEN** `preFilter.axes` is `[frequency]`
- **THEN** only the frequency axis is asked and no reference-area vocabulary is searched

#### Scenario: Configured axis prompt
- **WHEN** `preFilter.prompts.agencyPrompt` is set and the other axis prompts are not
- **THEN** the agency axis uses the configured prompt with `{values}` listing the offered agencies, and the other axes use the built-in prompts

### Requirement: Values each axis is offered
Each area axis SHALL be offered up to `referenceAreaTopN` vocabulary labels best matching the query, restricted to the current channel and to the axis's role (subject or partner). `frequency` SHALL be offered the discovery template's fixed frequency list, and `agency` the agencies the discovery RAG application currently holds. The built-in axis prompts SHALL treat group labels as values in their own right, never expanding them into countries or collapsing countries into them.

#### Scenario: Partner axis offered partner labels only
- **WHEN** the query is `exports of Germany to China`
- **THEN** the partner axis is offered only labels the channel's records name as partners, up to 20

#### Scenario: Group is not expanded
- **WHEN** the query is `inflation in the euro area` and the vocabulary holds `Euro area`, `Germany` and `France`
- **THEN** the reference-area axis selects `Euro area` and not `Germany` or `France`

### Requirement: Selections are grounded in the channel's current values
A selected value SHALL be kept only if the discovery RAG application currently holds it on the axis's field (`parsed_reference_areas`, `parsed_partner_reference_areas`, `parsed_frequencies`, `agency`), compared ignoring case and padding, and sent under the application's spelling without duplicates. Rationale: the service rejects the whole search for any unknown value. Held values SHALL be cached per application for `DISCOVERY_DATASETS_PRE_FILTER_DIMENSIONS_TTL` seconds (default 300).

#### Scenario: Unknown value dropped
- **WHEN** the agency axis selects `ECB` and `XYZ` and the application holds only `ECB`
- **THEN** only `ECB` is sent

#### Scenario: Re-cased value recovered
- **WHEN** the reference-area axis selects ` united states` and the application holds `United States`
- **THEN** `United States` is sent

### Requirement: How axes combine into the matcher
Each axis SHALL become one filter field carrying all its grounded values, any of which matches, and all fields of an entry SHALL have to match. The two area axes SHALL be alternatives: each grounded area axis SHALL produce its own entry, which the service OR-s, and every entry SHALL also carry the frequency and agency fields and the `statgpt_channel` scope. A matcher SHALL thus hold at most two entries, and one when no area axis grounded a value.

#### Scenario: Subject or partner area
- **WHEN** the reference-area and partner axes both ground `United States` and the frequency axis grounds `Annual`
- **THEN** the matcher holds two entries, {`statgpt_channel`, `parsed_reference_areas` = [`United States`], `parsed_frequencies` = [`Annual`]} and {`statgpt_channel`, `parsed_partner_reference_areas` = [`United States`], `parsed_frequencies` = [`Annual`]}, so a record covering the United States as subject or as partner matches

#### Scenario: Several values on one axis
- **WHEN** the reference-area axis grounds `France`, `Germany` and `Italy` and no other axis grounds anything
- **THEN** the matcher holds one entry with `parsed_reference_areas` = [`France`, `Germany`, `Italy`]

### Requirement: Axes degrade independently
An axis that selects nothing, has no area vocabulary configured (`details.referenceAreaApplicationId` unset), or whose vocabulary search fails SHALL drop out of the matcher while the other axes stay. If the application's current values cannot be read, no axis SHALL be grounded and the pre-filter SHALL be skipped.

#### Scenario: No vocabulary configured
- **WHEN** `referenceAreaApplicationId` is unset and the query is `monthly CPI from Eurostat`
- **THEN** the area axes report `no reference-area vocabulary is configured for this channel` and the matcher still narrows on frequency and agency

#### Scenario: Partner vocabulary search fails
- **WHEN** the partner-role vocabulary search is rejected because no record names a partner
- **THEN** only the partner axis drops out, reporting `the reference-area vocabulary could not be read`

### Requirement: Fallback to the channel-scope search
The lookup SHALL search with only the channel-scope matcher whenever the pre-filter is disabled, grounds nothing, fails in any step, or its narrowed search is rejected. This fallback SHALL NOT be configurable, and its reason SHALL be recorded on the eval attachment. A rejected narrowed search SHALL also drop that application's cached values. Rationale: the cache can be minutes behind the application, and a stale cache must not cost users their results.

#### Scenario: Nothing grounded
- **WHEN** the query `latest figures` names no value on any axis
- **THEN** the search uses only the channel-scope matcher and the fallback reason says the query named no value this channel holds

#### Scenario: Narrowed search rejected
- **WHEN** the narrowed search is rejected with HTTP 422 because a cached value was removed from the application
- **THEN** the search is retried with only the channel-scope matcher, the pre-filter report has `applied` = false and the failure as `fallbackReason`, and the cached values are dropped

### Requirement: LLM judge decides relevance
Candidates SHALL be judged by an LLM (`details.llmModelConfig`, `details.prompts.relevancePrompt` or a built-in precision-first default) returning `relevant` and a one-sentence `reason` per `document_id`. Only relevant candidates SHALL be shown, in search rank order. Verdicts on ids not among the candidates SHALL be dropped. The judge SHALL NOT be called without candidates. Neither the judge nor the templates SHALL receive the `grade` or `statgpt_channel` metadata.

#### Scenario: Invented id
- **WHEN** the candidates are 11 and 13 and the judge marks 11 and 99 relevant
- **THEN** only document 11 is shown

#### Scenario: No candidates
- **WHEN** the search returns no discovery records
- **THEN** the judge is not called and no block is shown

### Requirement: Block rendered from channel templates
Relevant datasets SHALL be rendered with `details.templates.item` once per dataset and joined by newlines into the `{items}` placeholder of `details.templates.wrapper`. The item template SHALL have the document's metadata fields plus `description`, `display_name`, `document_id`, `rank` and `reason` as placeholders, and an unknown placeholder SHALL render as empty rather than fail. No block SHALL be produced when nothing is judged relevant or the rendered block is blank.

#### Scenario: Two relevant datasets
- **WHEN** documents ranked 1 and 3 are relevant and `item` is `- {name} ({agency}): {reason}`
- **THEN** the wrapper receives two lines in rank order 1, 3

#### Scenario: Unknown placeholder
- **WHEN** `item` contains `{publisher}` and no metadata field has that name
- **THEN** `{publisher}` renders as empty and the block is still shown

### Requirement: How the block reaches the agent and MCP clients
For the agent, a non-empty block SHALL be appended to the Data Query response string after a blank line, whatever the status of the data query. For an MCP client, it SHALL be added as a separate text content item, and the `message` in the tool result metadata and the structured content SHALL stay the pipeline's own response without the block.

#### Scenario: Agent response
- **WHEN** the pipeline responds `Here is your data.` and the block is `Related datasets: ...`
- **THEN** the agent receives `Here is your data.\n\nRelated datasets: ...`

#### Scenario: Block after a failed data query
- **WHEN** the pipeline finds no data for the query but the lookup renders a block
- **THEN** the block is still appended to the response

#### Scenario: MCP response
- **WHEN** an MCP client calls Data Query and the lookup renders a block
- **THEN** the tool result has an extra text content item holding the block and `message` is unchanged

### Requirement: Lookup failures never fail the turn
A failure anywhere in the lookup SHALL produce no block and SHALL be recorded on the eval attachment and the debug stage, and the Data Query call SHALL return the pipeline's result as if the lookup were not configured. A failure in the query pipeline SHALL cancel the lookup and SHALL raise the pipeline's own error unchanged, not wrapped in an exception group.

#### Scenario: Judge fails
- **WHEN** the relevance judge call fails
- **THEN** no block is shown, the eval attachment carries `error`, and the data query response is returned unchanged

#### Scenario: Pipeline fails
- **WHEN** the query pipeline raises an error while the lookup is running
- **THEN** the lookup is cancelled and the Data Query call raises that same error

### Requirement: Diagnostics go to an eval attachment and a debug stage
Each lookup SHALL record the query, pre-filter report, candidates, verdicts, selected ids, rendered block and any error. With debug attachments enabled, it SHALL be attached as `discovery_datasets_eval_attachment_<tool_call_id>.json`, even when it found nothing or failed; with debug stages shown, each step SHALL report to the stage `[DEBUG] <debugStageName>` (default `Discovery Datasets Lookup`). None of it SHALL go into tool state. Rationale: state is echoed back every turn and would grow.

#### Scenario: Eval attachment for a call
- **WHEN** debug attachments are enabled and the Data Query call `call_1` ran the lookup
- **THEN** the response carries `discovery_datasets_eval_attachment_call_1.json`

#### Scenario: Pre-filter report
- **WHEN** the reference-area axis was offered `France` and `Germany`, selected `France` and `Atlantis`, and the narrowed search succeeded
- **THEN** the report lists that axis with offered [`France`, `Germany`], selected [`France`, `Atlantis`], grounded [`France`], one filter entry and `applied` = true

#### Scenario: No debug stages
- **WHEN** debug stages are not shown
- **THEN** the lookup opens no stage
