# discovery-dataset-records Specification

## Purpose

Lets an administrator store, inspect, edit, export and delete a channel's discovery dataset (Grade C) records through the admin API and the CLI, and carries those records through channel export and import. Loading records from files is covered by `discovery-dataset-upload`.

## Requirements

### Requirement: Records are keyed on channel, agency and dataset id
Every discovery dataset record SHALL belong to exactly one channel, and a channel SHALL hold at most one record per (agency, dataset id), compared ignoring case and leading, trailing and repeated whitespace. Every write SHALL normalize whitespace in each text field before storing it. Deleting a channel SHALL delete its records.

#### Scenario: Keys differing only in case and spacing collide
- **WHEN** a channel holds agency `BI`, dataset `GDP` and an admin adds agency ` bi `, dataset `gdp`
- **THEN** the response is 409 naming the colliding agency and dataset id, and nothing is saved

#### Scenario: Same key in another channel
- **WHEN** channel 1 holds agency `BI`, dataset `GDP` and an admin adds the same record to channel 2
- **THEN** the record is created in channel 2

### Requirement: Stored spelling is kept
A record's agency and dataset id SHALL keep the casing they were stored with, and that spelling is what the API, exports and published documents show. A file or archive resubmission differing from the stored key only in case SHALL leave the record and its statuses untouched, because rewriting it would mark an unchanged record outdated and republish it. A single-record or bulk edit SHALL rewrite the casing, since it is a deliberate correction.

#### Scenario: Case-only resubmission
- **WHEN** a channel holds agency `BI` with indexing status INDEXED and an upload carries the same record with agency `bi` and otherwise identical fields
- **THEN** the record is counted as unchanged, keeps agency `BI`, and stays INDEXED

#### Scenario: Edit corrects the casing
- **WHEN** an admin sends `POST /discovery-datasets/{id}` with `agency` = `BI` for a record stored as `bi`
- **THEN** the record is returned with agency `BI` and its verdicts are reset

### Requirement: Writes refuse structural problems only
Every write SHALL refuse a record whose agency or dataset id is empty after normalization, or whose dataset id cannot name a workbook sheet: over 31 characters, containing any of `: \ / ? * [ ]`, starting or ending with an apostrophe, or equal to `history` in any case; a JSON write answers 422. Content checks SHALL run only in the indexing job, never on a write, because a write cannot absorb LLM-based checks and a stored verdict would go stale whenever the check set changes.

#### Scenario: Dataset id too long
- **WHEN** an admin adds a record whose dataset id has 32 characters
- **THEN** the response is 422 saying the dataset id must be at most 31 characters

#### Scenario: Missing unit is accepted
- **WHEN** an admin adds a record with an indicator `GDP` and no units
- **THEN** the record is created with validation status NOT_VALIDATED

#### Scenario: Legacy dataset id stays readable
- **WHEN** a record stored before the sheet-name rules has a 40-character dataset id
- **THEN** it can still be read, listed and exported as CSV, and only resubmitting that id is refused

### Requirement: Validation and indexing are independent status axes
Each record SHALL carry a validation axis, `validation_status` (NOT_VALIDATED, VALID, INVALID) with `validation_issues` and `validated_at`, and an indexing axis, `indexing_status` (NEW, OUTDATED, INDEXED, FAILED) with `indexed_at` and `index_error`. A new record SHALL start NOT_VALIDATED and NEW, and only the indexing job SHALL set VALID, INVALID, INDEXED or FAILED. `index_error` SHALL be set exactly while the status is FAILED. The job's derived fields SHALL be read-only through the API.

#### Scenario: New record
- **WHEN** an admin adds a record to a channel
- **THEN** it is returned with `validationStatus` NOT_VALIDATED, `indexingStatus` NEW, and no `validatedAt`, `indexedAt` or `indexError`

#### Scenario: Derived field cannot be edited
- **WHEN** an admin edits a record sending `indicatorsSummary`
- **THEN** the stored `indicatorsSummary` is unchanged

### Requirement: Editing a record resets its verdicts
A write that changes any descriptive field of a stored record SHALL reset its validation axis to NOT_VALIDATED with no issues and no `validated_at`, SHALL turn INDEXED into OUTDATED, and SHALL turn FAILED into NEW while clearing `index_error`; NEW and OUTDATED stay. A write that changes no field SHALL leave the record and its statuses untouched.

#### Scenario: Editing a published record
- **WHEN** a record is VALID and INDEXED and an admin changes its `description`
- **THEN** it becomes NOT_VALIDATED and OUTDATED

#### Scenario: Editing a failed record
- **WHEN** a record is FAILED with `index_error` = `timeout` and an admin changes its `url`
- **THEN** it becomes NEW and `index_error` is null

#### Scenario: Unchanged edit
- **WHEN** an admin sends an edit whose values equal the stored ones
- **THEN** the record keeps its statuses

### Requirement: Single records are created, read, edited and deleted
The admin API SHALL expose `POST /channels/{channel_id}/discovery-datasets` to add a record, `GET`, `POST` (edit) and `DELETE` (204) on `/discovery-datasets/{id}`. An unknown id SHALL answer 404, and an add or edit landing on a stored key SHALL answer 409. An edit SHALL leave omitted or `null` fields unchanged and clear a field sent as `""`, except agency and dataset id, which cannot be cleared. Reads SHALL omit the summarization calls unless `include_summary_calls=true`.

#### Scenario: Clearing a field
- **WHEN** an admin edits a record with `{"name": ""}`
- **THEN** the record's name is empty and every other field is unchanged

#### Scenario: Unknown id
- **WHEN** an admin requests `GET /discovery-datasets/999` and no record has id 999
- **THEN** the response is 404 naming id 999

#### Scenario: Summary calls on request
- **WHEN** an admin reads a summarized record without and then with `include_summary_calls=true`
- **THEN** only the second response carries `indicatorsSummaryCalls`

### Requirement: Bulk create and bulk update are all or nothing
`POST /channels/{channel_id}/discovery-datasets/bulk` SHALL add a list of records and save nothing unless all can be saved: items sharing a key answer 400 with the per-record problem report naming their positions, and items colliding with stored records answer 409 naming every colliding key. `POST /discovery-datasets/bulk` SHALL edit records by `id`, answer 404 and save nothing when any id is unknown, and return the records in request order.

#### Scenario: Duplicate items in one payload
- **WHEN** a bulk create carries items 0 and 2 both with agency `BI`, dataset `GDP`
- **THEN** the response is 400 with a problem naming items 0 and 2, and nothing is saved

#### Scenario: Unknown id in a bulk edit
- **WHEN** a bulk edit carries ids 5 and 999, and 999 does not exist
- **THEN** the response is 404 and record 5 is unchanged

### Requirement: Channel records are listed with filters
`GET /channels/{channel_id}/discovery-datasets` SHALL return a page of the channel's records ordered by id, with `limit` (default 100), `offset` (default 0), `count` and the filtered `total`. It SHALL accept `validation_status`, `indexing_status` and `agency`, combined with AND, where `agency` matches ignoring case and whitespace, and `include_summary_calls` as the single-record read does.

#### Scenario: Agency filter folds case
- **WHEN** a channel holds 3 records of agency `BI` and an admin lists with `agency=bi&validation_status=INVALID`
- **THEN** the response holds the invalid records among those 3, and `total` counts only them

### Requirement: Stats summarize a channel's records
`GET /channels/{channel_id}/discovery-datasets/stats` SHALL return `total`, `by_validation_status` and `by_indexing_status`, each carrying every status including zeros, and `by_agency`. `by_agency` SHALL hold one entry per agency as the `agency` filter folds it, keyed by one of its stored spellings and ordered by count descending, then by name, so every key works as an `agency` filter value.

#### Scenario: Spellings folded into one entry
- **WHEN** a channel holds 2 records of agency `BI`, 1 of `bi` and 1 of `ECB`
- **THEN** `by_agency` has two entries: a stored spelling of `BI` with count 3, then `ECB` with count 1

#### Scenario: Empty channel
- **WHEN** a channel holds no records
- **THEN** `total` is 0 and every status in both status maps is 0

### Requirement: Deleting records withdraws their published documents
Every explicit delete, of one record, of records by id (`DELETE /discovery-datasets/bulk`, possibly spanning channels) or of a whole channel (`DELETE /channels/{channel_id}/discovery-datasets/bulk`), SHALL first remove the records' documents from each owning channel's discovery RAG application and only then delete the rows. If that removal fails the request SHALL answer 502 and leave every row in place. The bulk deletes SHALL return the deleted records.

#### Scenario: RAG application unreachable
- **WHEN** an admin deletes record 5, which is INDEXED, and the channel's discovery RAG application cannot be reached
- **THEN** the response is 502 and record 5 still exists

#### Scenario: Records of two channels
- **WHEN** a bulk delete carries ids of records from channels 1 and 2
- **THEN** each record's document is withdrawn from its own channel's application before the rows are deleted

#### Scenario: No publish target
- **WHEN** a channel has no discovery RAG application configured and an admin clears it
- **THEN** its records are deleted without contacting any application

#### Scenario: Replace does not withdraw
- **WHEN** a `replace` upload or `replace` channel import removes record C
- **THEN** C's document is not withdrawn by that request, and the channel's next indexing run removes it

### Requirement: Clearing a channel also clears its area vocabulary
Deleting all of a channel's records SHALL also clear the channel's whole reference-area vocabulary, when a reference-area application is configured, after the rows are deleted, and a failure to clear it SHALL NOT fail the request. Deleting records by id SHALL leave the vocabulary alone, because the next indexing run prunes labels no remaining record uses, while a cleared channel has no record left to prune from.

#### Scenario: Vocabulary clear fails
- **WHEN** an admin clears a channel holding 4 records and the reference-area application rejects the clear
- **THEN** the response is 200 with the 4 deleted records

### Requirement: Records export selects a channel slice or chosen ids
`GET /channels/{channel_id}/discovery-datasets/export` SHALL export the records matching the list endpoint's filters (the whole channel when none is given), or exactly the records named by `id`, repeatable up to 500 times. `id` combined with a filter SHALL answer 422. An `id` the channel does not hold SHALL answer 404 naming every such id, because a file missing a selected row would look complete. The export SHALL validate nothing.

#### Scenario: Foreign id
- **WHEN** an admin exports channel 1 with `id=5&id=77` and record 77 belongs to channel 2
- **THEN** the response is 404 naming id 77

#### Scenario: Id combined with a filter
- **WHEN** an admin exports with `id=5&agency=BI`
- **THEN** the response is 422

### Requirement: Records export produces an uploadable file
The export SHALL accept `format` = `csv` (default) or `xlsx`, returned as `discovery_datasets.csv` or `discovery_datasets.xlsx`. With the defaults the CSV SHALL be byte-identical to the channel archive's `discovery_datasets.csv`, and a selection matching nothing SHALL return the header alone. The workbook SHALL have the template's `Datasets` sheet plus one indicators sheet per record, named with its dataset id. Both formats SHALL upload back as the same records.

#### Scenario: Empty slice
- **WHEN** an admin exports with `validation_status=INVALID` and no record is invalid
- **THEN** the response is a CSV holding only its header row

#### Scenario: Dataset without indicators
- **WHEN** an `xlsx` export holds dataset `GDP` with no indicators
- **THEN** the workbook still has a sheet `GDP` holding only the indicator header row

### Requirement: Export can carry the stored validation verdict
With `include_validation=true` the export SHALL append to each row the stored validation status, `validated_at` in UTC, and the issues one per line as `<column label>: <message>`. A NOT_VALIDATED record SHALL say it has not been validated yet rather than show no issues. A workbook SHALL highlight the cells the issues name and every blank units cell. The upload SHALL ignore these columns, so the file stays uploadable.

#### Scenario: Validation report round trip
- **WHEN** an admin downloads `format=xlsx&include_validation=true` and uploads it to another channel
- **THEN** the second channel receives the same records, all NOT_VALIDATED

#### Scenario: Unchecked record
- **WHEN** a NOT_VALIDATED record is exported with `include_validation=true`
- **THEN** its issues cell says no indexing job has checked it since it was last written

### Requirement: Workbook export refuses records a workbook cannot hold
A `format=xlsx` export SHALL answer 400, naming up to 10 offending records and counting the rest, when a dataset id is shared by two agencies ignoring case and whitespace, names a template sheet (`Datasets`, `Instructions`) or cannot name a sheet, or when a value exceeds 32,767 characters. It SHALL NOT rename sheets or truncate values, because the file would upload back as different records. Every text cell SHALL be written as a string, so a value starting with `=` never becomes a formula.

#### Scenario: Dataset id shared by two agencies
- **WHEN** the selection holds agency `BI` dataset `GDP` and agency `ECB` dataset `gdp`
- **THEN** the response is 400 saying the dataset id is also used by another agency, and the same selection exports as CSV

#### Scenario: Formula-like value
- **WHEN** a record's description is `=HYPERLINK("http://x")`
- **THEN** the workbook cell holds that text as a string

### Requirement: Channel archives carry discovery records
A channel export with scope `full` or `configs` SHALL write the channel's records to `discovery_datasets.csv`, in the records export's default format and without statuses or derived fields, and no such file when the channel holds none. An import whose scope includes configs SHALL load the file, reconciling on the record key so re-importing is idempotent; a missing file SHALL NOT be an error. New imported records start NOT_VALIDATED and NEW, and a structural problem fails the import job.

#### Scenario: Channel without records
- **WHEN** an admin exports a channel holding no discovery records
- **THEN** the archive has no `discovery_datasets.csv` and imports without error

#### Scenario: Re-import
- **WHEN** the same archive is imported twice into one channel
- **THEN** the second import changes no record and keeps their statuses

### Requirement: Older archives with discovery records are refused
Channel archives SHALL be written at version 2. An import of a version 1 archive containing `discovery_datasets.csv` SHALL fail before anything is imported, because its indicators are in a format no longer read; a version 1 archive without that file SHALL import as before.

#### Scenario: Version 1 archive with discovery records
- **WHEN** an admin imports a version 1 archive containing `discovery_datasets.csv`
- **THEN** the import job fails asking to re-export the channel, and nothing is imported

### Requirement: Channel import reconciles records and glossary terms by separate modes
`POST /channels/import` SHALL accept `discovery_datasets_mode` and `glossary_terms_mode`, each `upsert` or `replace` and defaulting to `upsert`, which `statgpt channel import` exposes as `--discovery-datasets-mode` and `--glossary-terms-mode`. A mode SHALL apply only when merging into an existing channel without `clean_up`: `upsert` keeps records the archive does not mention, `replace` deletes them, clearing the collection when the archive has no file for it.

#### Scenario: Replace with no records file
- **WHEN** a channel holds 5 discovery records and an archive without `discovery_datasets.csv` is merged into it with `discovery_datasets_mode=replace`
- **THEN** the channel holds no discovery records and its glossary terms are reconciled by `glossary_terms_mode`

#### Scenario: Upsert keeps unmentioned records
- **WHEN** a channel holds records A and B and an archive with only A is merged with the default modes
- **THEN** the channel still holds A and B

### Requirement: CLI clears a channel's discovery records
`statgpt discovery clear -c <channel>` SHALL delete all of the channel's discovery records and their published documents. When the channel holds none it SHALL say so and send no delete. Otherwise it SHALL ask for confirmation naming the channel's deployment id and record count, defaulting to no, unless `-y`/`--yes` is given, and SHALL refuse in non-interactive mode without `-y`. It SHALL report how many records were deleted.

#### Scenario: Confirmation names the scope
- **WHEN** an admin runs `statgpt discovery clear -c macro` and `macro` holds 12 records
- **THEN** the CLI asks whether to delete all 12 discovery datasets of `macro` and their published documents

#### Scenario: Nothing to clear
- **WHEN** an admin runs `statgpt discovery clear -c macro -y` and `macro` holds no records
- **THEN** the CLI reports that `macro` holds no discovery datasets and sends no delete request

### Requirement: CLI channel status shows a Grade C block
`statgpt channel status` SHALL show a "Discovery Datasets (Grade C)" block when the channel holds records or has a discovery RAG application. It SHALL show the total, the counts per status, the first 10 agencies by count with a count of the rest, and a `discovery reindex` hint when any record is NOT_VALIDATED, NEW, OUTDATED or FAILED. If the stats cannot be fetched, as from an older admin API without the stats endpoint, the block SHALL be omitted and the rest still render.

#### Scenario: Older admin API
- **WHEN** an admin runs `statgpt channel status` against an admin API that answers 404 for the stats endpoint
- **THEN** the channel status renders without the Grade C block

#### Scenario: Pending work
- **WHEN** a channel holds 3 records, one of them OUTDATED
- **THEN** the block shows the counts and suggests `discovery reindex -c <channel>`

#### Scenario: Publish target but no records
- **WHEN** a channel with a discovery RAG application holds no records
- **THEN** the block shows only `Total: 0`
