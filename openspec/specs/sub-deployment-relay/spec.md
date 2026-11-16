# sub-deployment-relay Specification

## Purpose

Defines how the streamed response of a DIAL sub-deployment that a tool calls (Deep Research, File RAG, Web Search) is relayed into the StatGPT chat response: its stages, attachments, citation annotations and token usage, and the configurable rules that rewrite the links of relayed annotations and attachments. What each tool does with the relayed content belongs to the tool's own capability (`deep-research`, `file-rag-tool`).

## Requirements

### Requirement: Stages are relayed
The stages of a sub-deployment response SHALL be relayed when the tool's `stagesConfig.debugOnly` is false or the conversation shows debug stages. A stage SHALL be closed with the status the sub-deployment closed it with, so a step that failed downstream does not show as completed. Stages still open when the response ends SHALL be closed, children before parents.

#### Scenario: Failed downstream step
- **WHEN** the sub-deployment closes its stage "Searching sources" with status `failed`
- **THEN** the relayed stage is closed as failed, not completed

### Requirement: Stage nesting is kept
A relayed stage SHALL open under the relayed stage the sub-deployment nested it in. When that parent was never relayed or is already closed, the stage SHALL open at the root of the message and a warning SHALL be logged, because a stage out of place is better than a failed response.

#### Scenario: Unknown parent
- **WHEN** a stage names a parent stage index that was never relayed
- **THEN** the stage opens at the root of the message and a warning is logged

### Requirement: Annotations go to the message
Citation annotations of a sub-deployment response SHALL be relayed to the assistant message, never to a stage, even when the tool's text goes to a stage, because an annotation claims a `<cit>` marker tag in a message. Every field SHALL be relayed unchanged except `index` and the fields changed by rewrite rules. Where the response cannot stream (the MCP path), annotations SHALL be dropped with a warning logged.

#### Scenario: RAG answer in a stage
- **WHEN** File RAG puts its answer in the tool-result stage and the RAG response carries annotations
- **THEN** the annotations are sent on the assistant message

### Requirement: Annotation indexes are unique per response
Relayed annotation indexes SHALL be renumbered into one index space per chat response, shared by every sub-deployment relayed in it, because sub-deployments number their annotations from zero and run concurrently, and the client reads two annotations with the same index as one entry, which would hide sources of a pill that cites several. Within one sub-deployment response the same incoming index SHALL map to the same new index. An annotation without an index SHALL keep none.

#### Scenario: Two concurrent tools
- **WHEN** two File RAG calls in one turn each relay annotations with indexes 0 and 1
- **THEN** the message carries four annotations with distinct indexes

### Requirement: Attachments are relayed
Attachments of the response and of its stages SHALL be relayed with their type, title, data, url, reference url and reference type, after the rewrite rules are applied. An attachment with neither `data` nor `url` SHALL be relayed with empty `data`.

#### Scenario: Attachment without content
- **WHEN** a sub-deployment sends an attachment with a title only
- **THEN** it is relayed with empty data

### Requirement: Token usage is attributed to the deployment
Token usage reported by the sub-deployment per model SHALL be added to the turn's usage under the called deployment, so it appears in the token usage stage and cost.

#### Scenario: Nested models
- **WHEN** the Deep Research response reports usage for two models
- **THEN** both are added to the turn's usage, attributed to the Deep Research deployment

### Requirement: Rewrite rules apply to relayed items
Deep Research and File RAG SHALL accept `rewriteRules`, an ordered list of rules applied to every relayed annotation and attachment, including stage attachments. Links inside the message text SHALL NOT be rewritten. Only the first rule whose selector matches SHALL apply to an item, and an item no rule matches SHALL be relayed unchanged.

#### Scenario: Link in the text
- **WHEN** a rule rewrites `files/` links and the answer text contains a markdown link to `files/abc/report.pdf`
- **THEN** the link in the text is unchanged

### Requirement: Rewrite rule selectors
A rule's selector SHALL name `appliesTo` (`annotation`, `attachment` or `both`) and at least one regex condition among `type`, `url`, `referenceUrl`, `attachmentTitle` and `bodyTitle`. For an annotation, `type`, `url` and `attachmentTitle` SHALL be read from its `body.source.attachment`, and `bodyTitle` from `body.title`. An item SHALL match when every condition is found in its field (regex search); a missing field SHALL NOT match.

#### Scenario: First match wins
- **WHEN** rule 1 selects attachments whose url matches `^files/` and rule 2 selects all attachments with type `application/pdf`
- **THEN** a PDF attachment with url `files/abc/report.pdf` is rewritten by rule 1 only

#### Scenario: Missing field
- **WHEN** a rule selects attachments by `referenceUrl` and an attachment has no reference url
- **THEN** the rule does not apply to it

### Requirement: Rewrite rules modify links
A rule SHALL apply its `rewrites` in order. Each rewrite SHALL replace the parts of `url` or `reference_url` that match its `pattern` with its `replacement`, which supports regex backreferences and `$env:{VAR}`. A rewrite whose field the item lacks SHALL be skipped.

#### Scenario: Public link
- **WHEN** a rewrite has field `url`, pattern `^files/[^/]+/(.*)$` and replacement `https://docs.example.org/\1`
- **THEN** an annotation citing `files/abc/report.pdf` is relayed with url `https://docs.example.org/report.pdf`

### Requirement: Invalid rewrite rules are rejected on load
A channel config SHALL be rejected when a rewrite rule has an invalid regex, a selector with no condition, empty `rewrites`, or a selector condition or rewrite field that a selected kind does not have: `referenceUrl` exists only on attachments and `bodyTitle` only on annotations, so neither is allowed with `appliesTo: both`.

#### Scenario: Reference URL on annotations
- **WHEN** a rule with `appliesTo: annotation` rewrites `reference_url`
- **THEN** the channel config fails validation
