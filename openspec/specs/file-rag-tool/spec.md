# file-rag-tool Specification

## Purpose

Answers questions from the channel's publications through a RAG DIAL application: it narrows the search with a publications pre-filter built from the query, calls the RAG application, and hands the answer, its attachments and its citations to the Supreme Agent and the user. How the RAG response's stages, annotations and attachments are relayed, and the `rewriteRules`, belong to `sub-deployment-relay`.

## Requirements

### Requirement: RAG backends
The tool's `version` SHALL select the backend: `GENERIC`, a Generic RAG DIAL application, or `DIAL`, the legacy DIAL RAG service. Both SHALL share the transport, pre-filter and response handling, differing only in the configuration payload. `deploymentId` SHALL name the RAG application and `metadataEndpoint` its metadata route (Generic RAG: `/v1/deployments/<app>/route/channel/metadata`).

#### Scenario: Generic RAG channel
- **WHEN** a channel's File RAG tool has `version: GENERIC` and `deploymentId: publications-rag`
- **THEN** the tool calls the `publications-rag` application with the Generic RAG configuration payload

### Requirement: Deprecated DIAL backend
The `DIAL` backend SHALL be deprecated in favor of `GENERIC`: the admin backend SHALL log a warning when a channel using it is created or updated, and `content init` SHALL print one.

#### Scenario: Deprecated backend
- **WHEN** `content init` creates a channel whose File RAG tool has `version: DIAL`
- **THEN** a warning says the `DIAL` backend is deprecated in favor of `GENERIC`

### Requirement: Non-default RAG endpoint
When `DIAL_RAG_PGVECTOR_URL` is set, the metadata request SHALL go to that endpoint with `DIAL_RAG_PGVECTOR_API_KEY`, and when both are set the RAG call SHALL go there too, instead of to DIAL Core with the caller's key.

#### Scenario: Remote RAG service
- **WHEN** both settings are set
- **THEN** the metadata request and the RAG call go to that endpoint with that key

### Requirement: Tool arguments
The tool SHALL take `query` and `search_all_publications` (default false). `query` SHALL be screened by the out-of-scope check when the tool is called over MCP. Two further arguments SHALL be injected by evaluations and never shown to the model: `target_prefilter_json`, a pre-filter used instead of building one from the query, and `target_current_date`, a date used as "today" when building the pre-filter and, on Generic RAG, in answer generation.

#### Scenario: Eval with a fixed date
- **WHEN** an evaluation calls the tool with `target_current_date` = 2025-03-31 and the query "latest monthly report"
- **THEN** "latest" is decoded relative to 2025-03-31 and Generic RAG generates the answer as of that date

### Requirement: Publications pre-filter
Unless `search_all_publications` is set or a target pre-filter is given, the tool SHALL build a publications pre-filter from the query. It SHALL load the RAG metadata, which SHALL have a `publication_type` dimension, as a list `[{name, values}]` or a map `{name: [values]}`. It SHALL run four LLM extractions in parallel: time period, "latest" intent, publication types and last N publications. Publication types the metadata lacks SHALL be dropped and hallucinated future dates corrected.

#### Scenario: Unknown type
- **WHEN** the model selects a publication type the metadata does not list
- **THEN** that type is dropped from the filter

### Requirement: Latest and last-N publications
When the query asks for the latest publications without a last N or a start date, each selected publication type, or every type when none was selected, SHALL be limited to the window its `decoderOfLatest` entry defines (default one year). A last N SHALL become a top-N selection of the N most recent documents by publication date.

#### Scenario: Latest without a type
- **WHEN** the query is "What do the latest publications say about growth?" and no publication type is named
- **THEN** the filter covers every publication type, each limited to its latest-publication window

### Requirement: Pre-filter failures do not block the search
When the metadata cannot be loaded or the pre-filter cannot be built, the tool SHALL search without a filter and show a user-friendly error in the tool-result stage. The stage SHALL carry the pre-filter's LLM output and the final filter as attachments, or a note that the filter was skipped at the agent's request.

#### Scenario: Metadata endpoint down
- **WHEN** the metadata request fails
- **THEN** the stage says the pre-filter could not be built and the RAG application is called without a filter

### Requirement: Agent sees the applied pre-filter
Every tool response SHALL end with a note stating the applied pre-filter as JSON, saying when it was decoded from a "latest" request, or saying that all publications were searched, so the agent can report it and broaden a search that found nothing by repeating it with `search_all_publications`.

#### Scenario: Nothing found with a filter
- **WHEN** a filtered search finds nothing
- **THEN** the tool response names the applied filter, and the agent may repeat the call with `search_all_publications: true`

### Requirement: Generic RAG configuration payload
For Generic RAG the pre-filter SHALL be sent as `custom_fields.configuration.retriever.document_selector` with `type: explicit`, its `filters` (publication type and publication date interval) and its `top_n`, and a date override as `custom_fields.configuration.generation.current_date`. The retriever `type` SHALL be left out, and nothing else sent, so the application's own defaults apply. Since the application ignores unknown keys, this nesting SHALL match its request configuration exactly.

#### Scenario: Last three reports
- **WHEN** the pre-filter selects publication type `Annual Report` and the last 3 publications
- **THEN** the request carries `retriever.document_selector` with `type: explicit`, a filter for `Annual Report`, and `top_n` of 3 sorted by `publication_date` descending

### Requirement: Empty and legacy payloads
With neither a filter nor a date override, no RAG configuration SHALL be sent. The legacy DIAL backend SHALL get its own filter format and SHALL ignore a date override with a warning.

#### Scenario: All publications
- **WHEN** the agent sets `search_all_publications` and no date override is given
- **THEN** the RAG request carries no `custom_fields.configuration`

### Requirement: Answer found or not found
A RAG answer SHALL count as found when it carries at least one attachment or annotation. A found answer SHALL be shown in the tool-result stage together with the query, and the agent SHALL get the answer, the titles and types of its attachments, and the pre-filter note. Otherwise the tool SHALL respond that it was unable to find relevant data for the query, with the pre-filter note. An error during the RAG stream SHALL be logged, and what arrived before it SHALL be handled the same way.

#### Scenario: Answer without sources
- **WHEN** the RAG application answers with text but no attachments or annotations
- **THEN** the tool reports that it was unable to find relevant data for the query

### Requirement: Where attachments go
`attachmentsTarget` SHALL decide where the RAG attachments are attached: `stage`, the tool-result stage (default), or `choice`, the assistant message. They SHALL be attached to one of the two, never both, after the RAG stream ends. The answer text SHALL always go to the stage. Annotations SHALL always go to the assistant message, and the `<cit>` markers in the stage text SHALL resolve against them.

#### Scenario: Attachments on the message
- **WHEN** `attachmentsTarget` is `choice`
- **THEN** the RAG attachments appear on the assistant message and not in the stage, while the answer text stays in the stage

### Requirement: Deprecated attachment URL override
`attachmentUrlOverride` SHALL be deprecated in favor of `rewriteRules`, which also cover annotations and stage attachments. When set, it SHALL replace the prefix of each response attachment's DIAL file `reference_url` with its value, after the rewrite rules. `content init` SHALL print a warning when a channel sets it.

#### Scenario: Both configured
- **WHEN** a rewrite rule changes an attachment's `reference_url` and `attachmentUrlOverride` is also set
- **THEN** the override is applied to the result of the rewrite rule

### Requirement: Pre-filter-only debug mode
When the `rag_prefilter_only` dev command is on for the conversation (or `CMD_RAG_PREFILTER_ONLY` is set), the tool SHALL build and show the pre-filter but skip the RAG call, responding with the query and the pre-filter note.

#### Scenario: Debugging the filter
- **WHEN** the command is on and the user asks about publications
- **THEN** the pre-filter is shown and the RAG application is not called
