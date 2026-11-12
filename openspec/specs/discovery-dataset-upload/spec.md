# discovery-dataset-upload Specification

## Purpose

Lets an administrator load filled discovery dataset (Grade C) files into a channel through the admin API and the CLI, reconciling them with the records the channel already holds and reporting every structural problem before anything is saved.

## Requirements

### Requirement: Upload accepts several files in one request
The upload endpoint SHALL accept one or more files in the multipart field `file`, repeated once per file, and SHALL accept `.xlsx` and `.csv` files mixed in one request. A request carrying a single `file` part SHALL behave as a one-file batch.

#### Scenario: Mixed formats in one request
- **WHEN** an upload carries `agencies.xlsx` and `energy.csv`, both valid
- **THEN** the records of both files are loaded and the response is 200

#### Scenario: Single-file request keeps working
- **WHEN** an existing caller sends exactly one `file` part
- **THEN** the upload succeeds with the same channel-level counters as before this change

### Requirement: Batch is reconciled as one payload
The rows of all files in a request SHALL be merged into one set of records and reconciled with the channel in a single transaction. Records SHALL be matched on (agency, dataset id), ignoring case and spacing. In `upsert` mode records no file mentions SHALL be kept; in `replace` mode they SHALL be deleted. A record whose fields are unchanged SHALL NOT be rewritten.

#### Scenario: Replace keeps the records of every file
- **WHEN** a channel holds records A, B and C, and a `replace` upload carries one file with A and another with B
- **THEN** A and B are kept or updated, C is deleted, and `deleted` is 1

#### Scenario: Upsert keeps unmentioned records
- **WHEN** a channel holds record C and an `upsert` upload carries two files that do not mention C
- **THEN** C is kept and `deleted` is 0

### Requirement: Batch is all or nothing
If any file in a request has a structural problem, the upload SHALL respond 400, save nothing, and list the problems found in every file of the request, capped by the configured problem limit and marked truncated beyond it.

#### Scenario: Problems in two files reported together
- **WHEN** `a.xlsx` has an empty agency in row 4 and `b.csv` has an invalid value in row 9
- **THEN** the response is 400, lists both problems, and the channel's records are unchanged

### Requirement: Problems name their file
Each problem in a refused upload SHALL carry a `file` field holding the name of the uploaded file it was found in, alongside its existing `row` and `cell` location.

#### Scenario: Problem traced to its file
- **WHEN** row 4 of `a.xlsx` has an empty agency
- **THEN** the problem carries `file` = `a.xlsx`, `row` = 4 and the agency cell reference

### Requirement: File-level failures do not hide other files
A file that cannot be read as a discovery workbook or CSV, exceeds the per-file byte cap, or exceeds the per-file row cap SHALL be reported as a problem of that file with no row or cell, and the remaining files SHALL still be checked and their problems reported in the same response.

#### Scenario: Unreadable file among valid ones
- **WHEN** an upload carries `report.pdf` renamed to `report.xlsx` and `b.csv` with a problem in row 3
- **THEN** the response is 400 with one problem for `report.xlsx` saying it cannot be read and one for `b.csv` row 3

#### Scenario: Oversized file
- **WHEN** one file exceeds the per-file byte cap
- **THEN** the response is 400, not 413, with a problem for that file naming the limit

### Requirement: Duplicate keys across files are refused
Two records in a request that land on the same (agency, dataset id), ignoring case and spacing, SHALL refuse the upload with a problem naming every file and row involved, whether the records are in one file or in different files.

#### Scenario: Same dataset in two files
- **WHEN** `a.xlsx` row 2 and `b.csv` row 7 both describe agency `BI`, dataset `GDP`
- **THEN** the response is 400 with a problem naming `a.xlsx` row 2 and `b.csv` row 7, and nothing is saved

### Requirement: Duplicate file names are refused
An upload carrying two files with the same name SHALL be refused with 400 before any file is parsed, naming the repeated file name.

#### Scenario: Same name twice
- **WHEN** an upload carries two parts both named `energy.xlsx`
- **THEN** the response is 400 naming `energy.xlsx` and nothing is saved

### Requirement: Number of files per request is capped
The upload SHALL refuse a request carrying more files than the configured maximum (`DISCOVERY_UPLOAD_MAX_FILES`, default 20) with 400 before any file is parsed. The byte cap and the row cap SHALL apply to each file separately.

#### Scenario: Too many files
- **WHEN** the maximum is 20 and an upload carries 21 files
- **THEN** the response is 400 naming the limit and nothing is saved

#### Scenario: Row cap is per file
- **WHEN** the row cap is 10000 and an upload carries two files of 8000 rows each with distinct keys
- **THEN** the upload succeeds and `rows_read` is 16000

### Requirement: Summary reports the batch and each file
A successful upload SHALL return the channel-level counters `created`, `updated`, `unchanged` and `deleted` for the whole batch, `rows_read` as the total over all files, and `files`: one entry per uploaded file, in request order, with its `file` name and `rows_read`.

#### Scenario: Per-file breakdown
- **WHEN** an upload carries `a.xlsx` with 3 data rows and `b.csv` with 5
- **THEN** `rows_read` is 8 and `files` is `[{file: a.xlsx, rows_read: 3}, {file: b.csv, rows_read: 5}]`

### Requirement: CLI uploads several paths in one request
`statgpt discovery upload` SHALL accept `--file` (`-f`) more than once and SHALL take every value following it up to the next option. A directory SHALL expand to the `.xlsx` and `.csv` files directly inside it, excluding lock and hidden files; a value containing glob characters SHALL be expanded by the CLI, unless it names an existing file, which SHALL be taken literally. The resulting paths SHALL be de-duplicated, sorted, and sent in one request.

#### Scenario: Shell-expanded glob
- **WHEN** an admin runs `statgpt discovery upload -c ch --file data/a.xlsx data/b.csv` (as a shell expands `data/*`)
- **THEN** both files are sent in one upload request

#### Scenario: Directory and repeated flag
- **WHEN** an admin runs `discovery upload -c ch --file data/ --file extra.csv` and `data/` holds `a.xlsx` and `~$a.xlsx`
- **THEN** one request carries `a.xlsx` and `extra.csv`

#### Scenario: Pattern matches nothing
- **WHEN** a `--file` value is a pattern or directory that yields no upload file
- **THEN** the CLI reports the value and sends no request

### Requirement: CLI reports per-file results
After a successful upload the CLI SHALL print the batch counters and, when more than one file was sent, each file's rows read. After a refused upload it SHALL print every problem with its file, row, cell and field.

#### Scenario: Refused batch
- **WHEN** the API refuses a two-file upload with problems in both files
- **THEN** the CLI prints a problem table whose rows name the file of each problem

### Requirement: Content init uploads a channel's discovery files together
`statgpt content init` SHALL upload every discovery file of a client's discovery folder to each target channel in a single upsert request, so the folder is loaded all or nothing per channel and duplicates across its files are refused.

#### Scenario: One bad file in the folder
- **WHEN** a client's discovery folder holds `a.xlsx` and `b.csv`, and `b.csv` has a structural problem
- **THEN** no records from either file are loaded into that channel, the problems are printed, and the batch report records one failed discovery item for the channel
