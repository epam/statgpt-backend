# discovery-indexing Specification

## Purpose

Lets an administrator run an indexing job over a channel's discovery datasets (Grade C), which re-validates every record and reconciles the channel's Generic RAG documents, and its reference-area vocabulary, with the records that pass.

## Requirements

### Requirement: Reindex is triggered as a background job
`POST /channels/{channel_id}/discovery-datasets/indexing-jobs` SHALL create a job in status `QUEUED`, run it in the background, and respond 202 with the job. While a job of the channel is `QUEUED` or `IN_PROGRESS`, a trigger SHALL respond 409 naming that job, and two simultaneous triggers SHALL NOT both create a job. A channel with no publish target SHALL be refused with 409 naming the field to configure, and no job SHALL be created.

#### Scenario: Trigger accepted
- **WHEN** an administrator triggers indexing of channel 7, which has no active job
- **THEN** the response is 202 with a job of channel 7 in status `QUEUED`

#### Scenario: Job already running
- **WHEN** job 41 of channel 7 is `IN_PROGRESS` and indexing of channel 7 is triggered again
- **THEN** the response is 409 saying indexing job 41 is already in progress for channel 7, and no job is created

#### Scenario: Channel with nowhere to publish
- **WHEN** channel 7 has no `discoveryDatasets` tool configured and indexing is triggered
- **THEN** the response is 409 naming `details.discoveryDatasets.details.applicationId`, and no job is recorded

### Requirement: Publish targets come from the discovery datasets tool
Records SHALL be published to the Generic RAG application named by `applicationId` of the channel's `discoveryDatasets` tool, and the reference-area vocabulary to the one named by `referenceAreaApplicationId`; both SHALL accept `$env:{VAR}`. The tool's `enabled` flag SHALL NOT affect indexing, because it gates only the chat-time lookup. An unset `referenceAreaApplicationId` SHALL skip the vocabulary without failing the job.

#### Scenario: Disabled tool still indexes
- **WHEN** a channel's `discoveryDatasets` tool has `enabled: false` and `applicationId: discovery-rag`
- **THEN** a triggered job publishes the channel's records to `discovery-rag`

#### Scenario: No vocabulary target
- **WHEN** `referenceAreaApplicationId` is unset
- **THEN** the job publishes the records, publishes no vocabulary, and still completes

### Requirement: Jobs can be polled and listed
`GET /discovery-datasets/indexing-jobs/{job_id}` SHALL return one job, or 404 for an unknown id. `GET /channels/{channel_id}/discovery-datasets/indexing-jobs` SHALL return the channel's jobs newest first, paginated by `limit` (default 100) and `offset` (default 0), with `count` and `total`.

#### Scenario: Unknown job
- **WHEN** job 999 does not exist and it is polled
- **THEN** the response is 404 saying discovery indexing job with id=999 was not found

#### Scenario: Listing a channel's jobs
- **WHEN** channel 7 has jobs 3, 5 and 9 and is listed with `limit=2`
- **THEN** the response holds jobs 9 and 5, `count` 2 and `total` 3

### Requirement: Job status reports the run's outcome
A job SHALL move from `QUEUED` to `IN_PROGRESS` and end `COMPLETED` or `FAILED`. A completed job SHALL carry its record and document counters and a one-line `details` summary. A failure of one record SHALL NOT fail the job; `records_failed` SHALL count records failed in summarization or publishing. A failure of the run itself SHALL mark the job `FAILED` with the error in `reason_for_failure`, keeping what earlier stages committed.

#### Scenario: Completed run
- **WHEN** a run over 10 records finds 8 valid, publishes 6, skips 2 already indexed and removes 1 orphan
- **THEN** the job is `COMPLETED` with `records_total` 10, `records_valid` 8, `records_invalid` 2, `documents_upserted` 6, `documents_deleted` 1, and `details` says it skipped 2 already indexed

#### Scenario: Run-level failure
- **WHEN** the RAG application cannot be reached while the run lists its documents
- **THEN** the job is `FAILED`, `reason_for_failure` describes the error, and the validation verdicts written earlier in the run are kept

### Requirement: Interrupted jobs do not block the channel
A run cancelled before it finishes, such as by the background task timeout (`BACKGROUND_TASKS_TASK_TIMEOUT`, default 3600 s), SHALL mark its job `FAILED` with a reason saying so. A job left `QUEUED` or `IN_PROGRESS` and untouched for over 12 hours SHALL be marked `FAILED` by the admin `fix_statuses` step (`FIX_STATUSES` and `INIT` modes). Until a job is final, triggers for its channel answer 409.

#### Scenario: Run cut off by the task timeout
- **WHEN** a run is still publishing when the background task timeout expires
- **THEN** the job is `FAILED` with a reason starting "Cancelled before it finished", and a new trigger for the channel is accepted

#### Scenario: Job abandoned by a crashed process
- **WHEN** a job has been `IN_PROGRESS` for 13 hours with no update and `fix_statuses` runs
- **THEN** the job is `FAILED` with a reason saying it was stuck in a non-final status

### Requirement: CLI reindex runs a job to completion
`discovery reindex` SHALL trigger a job for the channel given by `-c` (or chosen interactively), poll it until it ends, and print the job's counters and `details` on success or its `reason_for_failure` on failure. `--force` SHALL take comma-separated stage names, case-insensitive, and SHALL reject an unknown name before anything is sent.

#### Scenario: Forced reindex from the CLI
- **WHEN** an administrator runs `discovery reindex -c my-channel --force publishing`
- **THEN** a job forced to redo publishing is triggered, polled to `COMPLETED`, and its counters are printed

#### Scenario: Unknown stage
- **WHEN** `--force publish` is given
- **THEN** the CLI reports `publish` as an invalid stage, lists `publishing` and `summarization`, and triggers no job

### Requirement: Forced stages are named explicitly
The trigger's `force` query parameter SHALL take stage names, repeated once per stage: `summarization` re-derives every valid record's indicator summary, and `publishing` rebuilds every valid record's document and every vocabulary document, even where nothing changed. Any other value, such as `true`, SHALL be refused with 422. Validation SHALL NOT be a stage; it reruns on every record. Summarization SHALL run after validation and before publishing.

#### Scenario: Forcing both stages
- **WHEN** indexing is triggered with `?force=summarization&force=publishing`
- **THEN** the job's `details` starts with "Forced: summarization, publishing."

#### Scenario: Boolean force
- **WHEN** indexing is triggered with `?force=true`
- **THEN** the response is 422 and no job is created

### Requirement: Every record is re-validated in isolation
Each run SHALL evaluate the checks over every record, whatever its stored verdict. A record with any issue SHALL become `INVALID` with its issues (`field`, `message`) stored, otherwise `VALID` with issues cleared; both get `validated_at`. There SHALL be no severity: any issue makes a record unpublishable, so a check must only emit one when the record should not be indexed. A check that raises SHALL become an issue on its own field, and other checks and records SHALL still run.

#### Scenario: Previously valid record fails a new check
- **WHEN** a record stored as `VALID` now lacks a field the current checks require
- **THEN** the run marks it `INVALID` with an issue for that field

#### Scenario: A check crashes on one record
- **WHEN** the `url` check raises on record A
- **THEN** record A is `INVALID` with an issue on `url` saying the check could not be evaluated, and record B is validated normally

### Requirement: The check set
A record SHALL be `INVALID` when its `description` is empty; when `reference_area`, `frequency_coverage` or `indicators_coverage` names no value; when a frequency is outside the template vocabulary; when an indicator has no units; or when `url` is set and is not an `http`/`https` address with a host. Areas SHALL NOT be checked against ISO codes: group labels are valid. Empty areas, frequencies or indicators are refused because the pre-filter narrows on them.

#### Scenario: Group label accepted
- **WHEN** a record's `reference_area` is `Euro area` and every other field is valid
- **THEN** the record is `VALID`

#### Scenario: Unknown frequency
- **WHEN** `frequency_coverage` is `annual; Biennial`
- **THEN** the record is `INVALID` with one issue on `frequency_coverage` saying `'Biennial'` is not one of `Daily; Business daily; Weekly; Monthly; Quarterly; Semi-annual; Annual; Irregular`, and `annual` passes because matching ignores case

#### Scenario: Many indicators without units
- **WHEN** 8 indicators of a record have no units
- **THEN** one issue on `indicators_coverage` names the first 5, each cut to 60 characters, and ends with "and 3 more."

#### Scenario: Plain http and empty url
- **WHEN** one record's `url` is `http://stats.example.org/gdp` and another's is empty
- **THEN** neither gets a `url` issue, while `ftp://stats.example.org` would

### Requirement: One document per record, body is the description
Each published record SHALL be one plain-text document whose content is its `description` alone; every other field SHALL travel verbatim as metadata, with `grade` `C` and `statgpt_channel` set to the channel's deployment id. Only `indicators_coverage` SHALL be paraphrased: it carries the record's indicator summary, not the submitted list. `missing_indicators` SHALL be sent empty. No field is repeated between body and metadata, because the application indexes both and would index it twice.

#### Scenario: Document of a record
- **WHEN** a valid record of channel `ch-1` with description "Monthly money supply", agency `BI` and dataset `M2` is published
- **THEN** its content is "Monthly money supply" and its metadata holds `grade` `C`, `statgpt_channel` `ch-1`, `agency` `BI`, `dataset_id` `M2`, `name`, `url`, `reference_area`, `regional_coverage`, `time_coverage`, `frequency_coverage`, `indicators_coverage` and the `parsed_*` arrays, none of them holding the description

### Requirement: List cells are split into parsed arrays
At publish time `reference_area` SHALL be split into `parsed_reference_areas` and `parsed_partner_reference_areas`, and `frequency_coverage` into `parsed_frequencies`, beside the verbatim cells. Splitting SHALL use no LLM: split on `;`, normalize whitespace, drop empties and case-insensitive duplicates. A token starting with `partner countries:` SHALL mark itself and every later token as partners. Areas SHALL be kept as written; frequencies SHALL take the vocabulary's spelling.

#### Scenario: Subjects and partners in one cell
- **WHEN** `reference_area` is `Germany; France;; Partner countries: China; World`
- **THEN** `parsed_reference_areas` is `["Germany", "France"]` and `parsed_partner_reference_areas` is `["China", "World"]`, with `World` kept as a value rather than expanded

#### Scenario: Frequencies folded
- **WHEN** `frequency_coverage` is `annual; MONTHLY; Annual`
- **THEN** `parsed_frequencies` is `["Annual", "Monthly"]`

### Requirement: Array metadata fields are plain string lists
Every array metadata field of a discovery or vocabulary document SHALL be a non-nullable list of strings that defaults to empty. They SHALL NOT be made optional or enum-typed, because the RAG service derives its request model from the metadata schema and such a field makes that derivation fail for every search on the application.

#### Scenario: Rendered schema
- **WHEN** the metadata schema for `parsed_frequencies` or `roles` is rendered for the RAG application
- **THEN** it is an array of strings with no null option and no enum

### Requirement: Document names are unique per record
A record's document SHALL be named `{agency} - {dataset_id} [{digest}].txt`, with invalid file-name characters escaped, the readable part capped at 100 characters, and a 12-hex-character SHA-256 digest of the grade, the channel's deployment id and the record's key folded for case and whitespace. Uploads SHALL overwrite a file stored under that name. The digest exists because the RAG service derives storage paths from names alone, so equal names would collapse into one document.

#### Scenario: Same record in two channels
- **WHEN** record `BI` / `GDP` exists in channels `ch-1` and `ch-2`, both publishing to one application
- **THEN** the two documents have different digests and both stay in the index

#### Scenario: Re-spelled key
- **WHEN** a record's agency is re-spelled from `bi` to `BI`
- **THEN** the digest is unchanged and only the readable part of the name changes

### Requirement: Documents are scoped and the metadata schema is verified
A run SHALL only touch documents whose `grade` and `statgpt_channel` match it. Before writing anything, it SHALL read the metadata schema of the records application and, if set, the vocabulary application, and SHALL fail the job naming any filtered field not declared filterable: `grade`, `statgpt_channel`, `agency` and the three `parsed_*` arrays for records; `kind`, `statgpt_channel`, `roles` for the vocabulary.

#### Scenario: Shared application
- **WHEN** channels `ch-1` and `ch-2` publish to one application and `ch-1` is reindexed
- **THEN** none of `ch-2`'s documents is updated or deleted

#### Scenario: Field not filterable
- **WHEN** the records application's metadata schema does not declare `parsed_frequencies` filterable
- **THEN** the job is `FAILED` naming `parsed_frequencies`, and no document of either application is written

### Requirement: Valid records are published only when needed
Documents SHALL be matched to records by `agency` and `dataset_id` metadata, folded for case and whitespace. A `VALID` record that is `INDEXED` with a document not in error SHALL be skipped. Any other `VALID` record SHALL be published: in place if its document keeps the same name and is not in error, otherwise by deleting the document and uploading anew. A published record SHALL become `INDEXED` with `indexed_at` set and `index_error` cleared.

#### Scenario: Unchanged record
- **WHEN** an `INDEXED` record's document is `ready`
- **THEN** the run leaves it alone and counts it as skipped

#### Scenario: Edited record
- **WHEN** a record is `OUTDATED` after an edit that did not change its key
- **THEN** its document is replaced in place and the record becomes `INDEXED`

#### Scenario: Document in error
- **WHEN** an `INDEXED` record's document is in `error`
- **THEN** the document is deleted, a new one is uploaded, and `documents_deleted` counts the old one

### Requirement: Invalid records are withdrawn and orphans swept
For an `INVALID` record, a run SHALL delete its document, then set the record `NEW` with `indexed_at` and `index_error` cleared; deleting first means a crash never leaves a live document behind a record that reads as unpublished. A record whose summarization failed SHALL keep its document and status. A run SHALL then delete its documents no record claims, claiming no key, or duplicating a key (keeping the highest id); such a delete failure SHALL only be logged.

#### Scenario: Record becomes invalid
- **WHEN** an `INDEXED` record's description is emptied and the channel is reindexed
- **THEN** its document is deleted, `documents_deleted` counts it, and the record is `INVALID` and `NEW`

#### Scenario: Record deleted elsewhere
- **WHEN** a document of the channel claims a key no record holds any more
- **THEN** the run deletes it

### Requirement: Publish failures are recorded per record after settling
Records SHALL be published `DISCOVERY_PUBLISH_CONCURRENCY` at a time (default 8); an error marks that record `FAILED` with `index_error` and the run continues. The run SHALL then wait for uploads not yet `ready` or `error`, re-listing at 1 s doubling to 5 s, for up to `DISCOVERY_PUBLISH_SETTLE_TIMEOUT_SECONDS` (default 300, 0 disables). A document in `error` SHALL mark its record `FAILED` with `index_error`; one still processing at the deadline leaves its record `INDEXED`.

#### Scenario: Document fails to index in the background
- **WHEN** a record's upload is accepted with status `processing` and the document later lists as `error`
- **THEN** the record is `FAILED` with an `index_error` saying the next run publishes it again, and the job counts it in `records_failed`, not in `documents_upserted`

#### Scenario: Slow indexing
- **WHEN** a document is still `indexing` after `DISCOVERY_PUBLISH_SETTLE_TIMEOUT_SECONDS`
- **THEN** the job completes and the record stays `INDEXED`

### Requirement: Reference-area vocabulary is published from live records
After the records, a run SHALL publish one plain-text document per distinct reference area, whose content is the label and whose metadata is `kind` `reference_area`, `statgpt_channel`, `value` and sorted `roles` (`subject`, `partner`). Labels SHALL come from the parsed areas of `VALID` records with a live document, folded for case and whitespace, keeping the first spelling and the union of roles. It SHALL be named `{label} [{digest}].txt`, the digest covering kind, channel and folded label.

#### Scenario: One label, two roles
- **WHEN** record A names `China` as a subject and record B lists `partner countries: china`
- **THEN** the vocabulary holds one document `China` with `roles` `["partner", "subject"]`

#### Scenario: Label only an unpublished record names
- **WHEN** `Atlantis` is named only by a record whose publish failed
- **THEN** no vocabulary document for `Atlantis` is published

#### Scenario: Record held by summarization
- **WHEN** a record's summarization failed but an earlier run published it
- **THEN** its labels stay in the vocabulary

### Requirement: Vocabulary is reconciled and its failures fail the job
A run SHALL keep a vocabulary document of its channel whose label is still wanted with unchanged roles and not in error, upload the rest, and delete documents whose label is unwanted, missing or duplicated (keeping the highest id), logging delete failures. Other channels and other `kind`s SHALL be left alone. A vocabulary document returned in `error` SHALL fail the job, because a missing label silently narrows every query naming that area away from datasets covering it.

#### Scenario: Label no longer named
- **WHEN** the last record naming `Euro area` is deleted and the channel is reindexed
- **THEN** the `Euro area` vocabulary document is deleted

#### Scenario: Vocabulary document fails to index
- **WHEN** the vocabulary application returns `error` for `Germany` after the record documents were published
- **THEN** the record documents stay published and the job is `FAILED` with a reason naming `'Germany'`
