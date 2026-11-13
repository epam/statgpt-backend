# discovery-indicator-summarization Specification

## Purpose

Turns the indicator list a submitter authored for a discovery dataset (Grade C) into the shorter indicator text its search document is published with, keeps every LLM answer and what it cost, and lets an administrator review that work as an Excel report.

## Requirements

### Requirement: Short indicator lists are published as parsed
During an indexing run, every valid record SHALL get an `indicatorsSummary` derived from its `indicatorsCoverage`; invalid records SHALL NOT be touched. Entries SHALL be joined with `; `, an indicator repeated ignoring case listed once, and each entry SHALL read `name (units)`, or the bare name without units. When this compacted text is no longer than `summaryThresholdChars`, it SHALL be published as is, with no LLM call.

#### Scenario: Mixed units kept per entry
- **WHEN** a record lists `M1 (EUR billions)`, `M2 (USD millions)` and `m2 (USD millions)`, under the threshold
- **THEN** `indicatorsSummary` is `M1 (EUR billions); M2 (USD millions)` and `indicatorsSummarizedAt` is null

### Requirement: A shared unit is stated once
When every indicator has units and they are all equal, compared case-sensitively, the unit SHALL be stated once as a leading `Unit: <unit>` entry and the names listed without units, and the threshold SHALL be measured on this form. The prefix SHALL be added by the backend, never written by the LLM, so the shared unit a document states is exactly the one the submitter wrote.

#### Scenario: Shared unit
- **WHEN** a record lists `M1` and `M2`, both in `EUR billions`
- **THEN** `indicatorsSummary` is `Unit: EUR billions; M1; M2`

#### Scenario: One indicator without units
- **WHEN** a record lists `M1` in `EUR billions` and `Broad money (M2)` with no units
- **THEN** `indicatorsSummary` is `M1 (EUR billions); Broad money (M2)`

#### Scenario: Semicolon inside a name
- **WHEN** a record lists `Exports; goods` and `Imports`, both in `EUR billions`
- **THEN** `indicatorsSummary` is `Unit: EUR billions; Exports, goods; Imports`

### Requirement: Long indicator lists are collapsed into groups by an LLM
When the compacted list is longer than `summaryThresholdChars`, an LLM SHALL collapse it into indicator groups, published joined with `; ` after the `Unit:` prefix when there is one. A list longer than `chunkSizeChars` SHALL be split between indicators into parts, each summarized by its own call, and the parts' groups merged by one more call. An indicator SHALL never be split; one longer than a chunk forms a part by itself.

#### Scenario: One call when the list fits one chunk
- **WHEN** the compacted list is 9,000 characters with `summaryThresholdChars` 4000 and `chunkSizeChars` 50000
- **THEN** exactly one chunk call is made and its groups are published

#### Scenario: Long list summarized in parts and merged
- **WHEN** the list is 120,000 characters with `chunkSizeChars` 50000
- **THEN** three chunk calls each summarize whole indicators, and one merge call combines their groups into the published summary

### Requirement: Differing units are summarized by the LLM
When the indicators share no unit, the LLM answer SHALL also carry a `units` value. Units the model summarized as one value SHALL be published as the `Unit:` prefix; otherwise `units` is null and each group states its own units. A `;` inside a group or in `units` SHALL be replaced with `,`, since `;` separates the published entries.

#### Scenario: Differing units summarized as one value
- **WHEN** the indicators carry `USD millions` and `USD billions` and the model answers `units` = `USD`
- **THEN** the summary is published as `Unit: USD; <group>; <group>`

### Requirement: Summarization is configured per channel
Summarization SHALL be configured under `details.indexing` of the channel's `discoveryDatasets` tool, every field optional: `summaryThresholdChars` (default 4000), `chunkSizeChars` (default 50000), `maxSummaryChars` (default 4000), `llmModelConfig`, and `prompts.chunkSummaryPrompt` / `prompts.mergeSummaryPrompt`, each falling back to the built-in default when unset.

#### Scenario: Configured prompt replaces the default
- **WHEN** a channel sets `prompts.chunkSummaryPrompt` and leaves `mergeSummaryPrompt` unset
- **THEN** chunk calls use the channel's prompt and merge calls use the built-in default

#### Scenario: Prompt placeholders
- **WHEN** a configured prompt uses `{unit}`, `{units_instruction}`, `{max_summary_chars}` and, for the chunk prompt, `{indicators}` or, for the merge prompt, `{groups}`
- **THEN** they are filled with the shared unit (empty if none), the units instruction line, the limit, the indicators one per line, or each part's groups under a `## Part N` heading with the part's units

#### Scenario: Deployment-wide tuning
- **WHEN** `DISCOVERY_SUMMARIZE_CONCURRENCY`, `DISCOVERY_SUMMARIZE_TIMEOUT_SECONDS` and `DISCOVERY_SUMMARIZE_VALIDATION_RETRIES` are unset
- **THEN** 4 records are summarized at once, each call times out after 180 seconds, and an invalid answer is corrected up to 2 times

### Requirement: Answers must fit the published limit and are corrected with feedback
An answer SHALL be valid only when it matches the expected structure, has at least one non-empty group, and its published form, `Unit:` prefix and `; ` separators included, fits `maxSummaryChars`. An invalid answer SHALL be sent back in the same conversation with what is wrong with it, up to `DISCOVERY_SUMMARIZE_VALIDATION_RETRIES` times, and then the record SHALL fail. Continuing the conversation makes the model correct its answer rather than repeat it.

#### Scenario: Over-long answer corrected
- **WHEN** `maxSummaryChars` is 4000 and the first answer publishes as 4,250 characters
- **THEN** the model is told the summary is 250 characters over the limit of 4000, and its next, valid answer is published

#### Scenario: Retries exhausted
- **WHEN** retries are 2 and three consecutive answers are invalid
- **THEN** the record's summarization fails after three calls

### Requirement: A summary is recomputed only when its inputs change
A valid record SHALL be re-summarized only when its indicator list or the effective config changed since its summary was made, or when the run forces `summarization`; otherwise it is counted as unchanged. The effective config SHALL cover every `details.indexing` field, unset prompts as the defaults they fall back to, and the built-in instructions. The list SHALL be compared in its flat `name (units); ...` form, so summaries made before units were stored separately are not paid for again.

#### Scenario: Unchanged record skipped
- **WHEN** a record's list and the channel config are the same as at its last summarization and the run is not forced
- **THEN** no LLM call is made and the run counts it as unchanged

#### Scenario: Unset prompt matches its default
- **WHEN** a channel that left `chunkSummaryPrompt` unset is changed to set it to exactly the built-in default
- **THEN** no record is re-summarized

#### Scenario: Changed config or default prompt re-derives
- **WHEN** `summaryThresholdChars` changes from 4000 to 6000, or a release changes a built-in default prompt the channel relies on
- **THEN** every valid record is derived again on the next run

#### Scenario: Record never summarized
- **WHEN** a record has no stored summary fingerprint, such as one stored before summarization existed
- **THEN** it is derived on the next run

### Requirement: Forcing the summarization stage
The indexing-job trigger SHALL accept `force` as a repeatable query parameter naming stages, `summarization` or `publishing`, and SHALL refuse any other value, such as `true`, with 422. Forcing `summarization` SHALL re-derive every valid record's summary even when its inputs are unchanged.

#### Scenario: Forced re-summarization
- **WHEN** an indexing job is triggered with `force=summarization`
- **THEN** every valid record is summarized again and the job details start with `Forced: summarization.`

#### Scenario: Boolean force refused
- **WHEN** an indexing job is triggered with `force=true`
- **THEN** the response is 422 and no job is created

### Requirement: Applying a derived summary
A saved summary SHALL set `indicatorsSummarizedAt` to the current time when an LLM produced it and to null when it is a copy of the parsed list. If the text changed and the record was `INDEXED`, it SHALL become `OUTDATED` so the same run republishes it. `missingIndicators` is no longer derived and SHALL be emptied whenever a summary is saved, so values authored under the old template stop being published.

#### Scenario: Changed summary outdates the document
- **WHEN** an `INDEXED` record's list changes and its new summary differs from the stored one
- **THEN** the record becomes `OUTDATED`

#### Scenario: Identical summary keeps the document
- **WHEN** an `INDEXED` record is re-summarized by a forced run and the summary comes out the same
- **THEN** the record stays `INDEXED`

#### Scenario: Summary turns into a copy
- **WHEN** a record that was LLM-summarized is shortened below the threshold
- **THEN** its summary becomes the parsed list and `indicatorsSummarizedAt` becomes null

#### Scenario: Old template value cleared
- **WHEN** a record with `missingIndicators` = `Wages` is summarized
- **THEN** `missingIndicators` is empty

### Requirement: A failed summarization keeps the live document
When a record's summarization fails, by invalid answers or a request error, the record SHALL be marked `FAILED` with an `indexError` starting `Summarizing the indicator list failed:`, its previous summary, fingerprint and calls SHALL be kept so the next run tries again, and the run SHALL go on with the other records. Its live document SHALL be left exactly as it is, neither republished nor deleted as an orphan: it is still the best the channel has.

#### Scenario: One record fails, others continue
- **WHEN** two records are summarized and the first exhausts its retries
- **THEN** the first is `FAILED` with its document untouched, and the second is summarized and published

### Requirement: Progress is saved per record
Each record SHALL be saved as soon as its summary is derived or its failure recorded, so a run cut short keeps every summary it paid for and the next run skips them. A run cancelled before it finishes, for example by the background task timeout, SHALL mark the job `FAILED`. A completed job SHALL report `recordsSummarized`, count summarization failures in `recordsFailed`, and state `Indicator summaries: summarized N, copied N, unchanged N, failed N.` in its details.

#### Scenario: Run cut short resumes
- **WHEN** a run is cancelled after summarizing 30 of 50 records
- **THEN** the job is `FAILED` with a reason saying records summarized so far are kept, and the next run summarizes only the remaining 20

### Requirement: Every LLM answer is stored with its usage
A record SHALL keep in `indicatorsSummaryCalls` every LLM answer behind its current summary, rejected ones included, and in `indicatorsSummaryFailedCalls` the answers of its last failed summarization. Each SHALL be a run of `calls` plus `usage` summed per deployment and model. A copied summary SHALL store no calls, and a successful summarization SHALL clear `indicatorsSummaryFailedCalls`; a failure before any answer stores a failed run with no calls.

#### Scenario: Retry stored with the accepted answer
- **WHEN** a summary is accepted on the second answer of a one-part list
- **THEN** `indicatorsSummaryCalls.calls` holds two calls with `step` = `chunk` and `part` = 1, the first with the `error` sent back and its parsed `answer`, the second with `error` null

#### Scenario: Merge call
- **WHEN** a list was summarized in two parts and merged
- **THEN** the stored calls have `part` 1 and 2 for the `chunk` steps and `part` null for the `merge` step

#### Scenario: Answer that is not JSON
- **WHEN** the model's answer is cut short and is not a JSON object
- **THEN** its call is stored with `answer` null and the error

### Requirement: Each stored call records tokens, cost and duration
Each call's `usage` SHALL carry `deployment`, `model` (as the response names it, else the deployment), `promptTokens`, `completionTokens`, `cost` in USD priced with the deployment's DIAL pricing, and `durationSeconds`. When DIAL has no pricing, `cost` SHALL be null, and a summed run cost SHALL be null when any of its calls' cost is.

#### Scenario: Unpriced deployment
- **WHEN** DIAL has no pricing for the summarization deployment
- **THEN** each call's `cost` and the run's summed cost are null, and tokens and duration are still recorded

### Requirement: Records API returns summary calls only on request
The channel's record list and the single-record endpoint SHALL omit `indicatorsSummaryCalls` and `indicatorsSummaryFailedCalls` unless the request passes `include_summary_calls=true`, because the calls can outweigh the rest of a record many times over. `indicatorsSummary` and `indicatorsSummarizedAt` SHALL always be returned and are read-only.

#### Scenario: Calls left out by default
- **WHEN** a client lists a channel's discovery datasets without `include_summary_calls`
- **THEN** no record in the response carries `indicatorsSummaryCalls` or `indicatorsSummaryFailedCalls`

#### Scenario: Calls on request
- **WHEN** a client gets an LLM-summarized record with `include_summary_calls=true`
- **THEN** the response carries its `indicatorsSummaryCalls` with every call's usage

### Requirement: CLI writes a summary report
`statgpt discovery summary-report` SHALL take `-c/--channel`, `-o/--output` (default `discovery-summaries-<channel>-<YYYYMMDD-HHMM>.xlsx`) and `--all`. By default it SHALL include only records whose summary an LLM produced or whose last failed summarization holds at least one answer; `--all` includes every record. With nothing to report it SHALL write no file; otherwise it SHALL print `Wrote N dataset(s) to <path>`.

#### Scenario: Only summarized records by default
- **WHEN** a channel holds 10 records, of which 3 were LLM-summarized and 1 failed with answers
- **THEN** the report holds 4 records, and with `--all` it holds 10

#### Scenario: Nothing to report
- **WHEN** no record of channel `stats-ch` was LLM-summarized and `--all` is not passed
- **THEN** the CLI prints `stats-ch holds no summarized discovery datasets.` and writes no file

### Requirement: Summary report layout
The workbook SHALL have a `Summaries` sheet with one row per record, sorted by agency then dataset id ignoring case, and a totals row. Each record with stored calls SHALL get its own sheet, linked from its dataset id cell and back, listing the calls of its succeeded and failed runs. Text over Excel's 32,767-character cell limit SHALL be cut to fit, ending with `… [truncated]`.

#### Scenario: Record row
- **WHEN** a record was summarized from 300 indicators
- **THEN** its row shows agency, dataset id, name, validation and indexing status, summarized-at in UTC, indicator count, coverage and summary lengths and their ratio, parts, answers and rejected answers, model, token usage with cost, LLM seconds, whether the last run failed, the flat coverage and the summary

#### Scenario: Calls sheet
- **WHEN** a record's summary took a rejected and an accepted answer
- **THEN** its sheet lists both with run, step, part, whether accepted, model, usage, seconds, error, reasoning, units and groups

#### Scenario: Unknown cost in totals
- **WHEN** one of the reported calls has a null cost
- **THEN** the totals row's usage reads `<in> in + <out> out tokens, cost n/a`

#### Scenario: Over-long cell
- **WHEN** a record's flat coverage is 40,000 characters
- **THEN** its cell holds 32,767 characters ending with `… [truncated]`

### Requirement: Report sheet names are safe and unique
A record's sheet SHALL be named after its dataset id with characters Excel forbids replaced by `_`, cut to 31 characters, and unique ignoring case, numbered ` (2)`, ` (3)` on collision. It SHALL never take `Summaries` or `History`.

#### Scenario: Same dataset id under two agencies
- **WHEN** agencies `NSO` and `ECB` both have a summarized record with dataset id `gdp`
- **THEN** their call sheets are named `gdp` and `gdp (2)`

#### Scenario: Forbidden characters
- **WHEN** a record stored before ids had to be sheet-safe has dataset id `GDP/Q`
- **THEN** its sheet is named `GDP_Q`
