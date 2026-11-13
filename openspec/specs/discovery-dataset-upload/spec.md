# discovery-dataset-upload Specification

## Purpose

Defines the discovery dataset (Grade C) workbook and CSV formats, and lets an administrator load filled files into a channel through the admin API and the CLI, reconciling them with the records the channel already holds and reporting every structural problem before anything is saved.

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
Two or more records in a request that land on the same (agency, dataset id), ignoring case and spacing, SHALL refuse the upload with one problem per key, located at the first of them and naming every file and row involved, whether the records are in one file or in different files. When they are all in one file, the problem's `cell` SHALL list every dataset id cell involved.

#### Scenario: Same dataset in two files
- **WHEN** `a.xlsx` row 2 and `b.csv` row 7 both describe agency `BI`, dataset `GDP`
- **THEN** the response is 400 with a problem naming `a.xlsx` row 2 and `b.csv` row 7, and nothing is saved

#### Scenario: Same dataset three times in one workbook
- **WHEN** rows 2, 3 and 4 of `a.xlsx` all describe agency `BI`, dataset `GDP`
- **THEN** the response carries a single problem at row 2 naming rows 2, 3 and 4, with cell `D2, D3, D4`

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
After a successful upload the CLI SHALL print the batch counters and, when more than one file was sent, each file's rows read, followed by the `discovery reindex` command to run, since an upload publishes nothing. After a refused upload it SHALL print every problem with its file, row, cell and field, say that nothing was saved, and exit non-zero.

#### Scenario: Refused batch
- **WHEN** the API refuses a two-file upload with problems in both files
- **THEN** the CLI prints a problem table whose rows name the file of each problem, and exits with code 1

#### Scenario: Successful upload
- **WHEN** `statgpt discovery upload -c macro --file a.xlsx` succeeds
- **THEN** the CLI prints the counters and suggests `discovery reindex -c macro`

### Requirement: Content init uploads a channel's discovery files together
With the `discovery` component selected, `statgpt content init` SHALL upload every `.xlsx` and `.csv` file of a client config's `discovery_datasets/` folder to each channel the client's `channels.yaml` declares, in a single upsert request per channel, so the folder is loaded all or nothing per channel and duplicates across its files are refused. A declared channel that does not exist SHALL be reported as skipped, and a channel refusing its batch SHALL NOT stop the others.

#### Scenario: One bad file in the folder
- **WHEN** a client's discovery folder holds `a.xlsx` and `b.csv`, and `b.csv` has a structural problem
- **THEN** no records from either file are loaded into that channel, the problems are printed, and the batch report records one failed discovery item for the channel

#### Scenario: No discovery folder
- **WHEN** a client config has no `discovery_datasets/` folder
- **THEN** init says so and uploads no discovery records for that client

### Requirement: Content init does not publish discovery records
`statgpt content init` SHALL NOT trigger a discovery indexing run; after each successful channel upload it SHALL print the `discovery reindex` command to run. A reindex inside init would turn a config sync into a long wait per channel, and is worth running once after the uploads.

#### Scenario: Records uploaded by init
- **WHEN** init uploads discovery files to channel `macro`
- **THEN** the new records stay NEW and the output suggests `discovery reindex -c macro`

### Requirement: Workbook rows come from the Datasets sheet
A workbook SHALL be read from its `Datasets` sheet, found by name ignoring case and surrounding spaces, or else from its second sheet; a one-sheet workbook without it SHALL be refused. The template columns A-I are reference area, regional coverage, agency, dataset id, name, description, URL, time coverage and frequency coverage. Headers SHALL be matched by text ignoring case and order; columns with unknown headers SHALL be ignored, as are an export's validation columns.

#### Scenario: Columns reordered
- **WHEN** the `datasets` sheet has `Dataset ID` in column A and `Agency / organization` in column C
- **THEN** both are read, and a problem on row 4's dataset id names cell `A4`

#### Scenario: Unrecognized header row
- **WHEN** no header is recognized and the sheet is exactly 9 columns wide
- **THEN** the columns are read in template order and row 1 is still treated as the header

#### Scenario: Header missing a key column
- **WHEN** the headers are recognized but none names the dataset id
- **THEN** the file is refused naming the missing `Dataset ID` column

### Requirement: Indicators live on one sheet per dataset
A workbook SHALL take each dataset's indicators from the sheet named after its dataset id, matched ignoring case and spacing, with `Indicator` and `Units of measure` columns (or exactly two columns in that order). A dataset with no such sheet, or a sheet holding only its header, SHALL get an empty indicator list. Sheets no dataset id names, and the `Instructions` sheet, SHALL be ignored unread. Two sheets matching one dataset id SHALL be a problem on its row.

#### Scenario: Dataset without a sheet
- **WHEN** row 2 describes dataset `GDP` and the workbook has no `GDP` sheet
- **THEN** the record is stored with no indicators, and the indexing run reports it invalid

#### Scenario: Stray sheet
- **WHEN** the workbook also holds a sheet `Notes` that no dataset id names
- **THEN** the sheet is ignored and the upload succeeds

### Requirement: Dataset ids are unique within a workbook
Within one workbook a dataset id SHALL be unique across agencies, ignoring case and spacing, and SHALL NOT name a template sheet (`Datasets`, `Instructions`), because it names the dataset's indicators sheet. These checks apply to workbooks only; such datasets can be uploaded as separate files or as CSV. The generic dataset id rules are in `discovery-dataset-records`.

#### Scenario: Same id under two agencies
- **WHEN** row 2 has agency `BI`, dataset `GDP` and row 5 has agency `ECB`, dataset `gdp`
- **THEN** the upload is refused with a problem at row 5 saying the id is already used on row 2

#### Scenario: Id naming a template sheet
- **WHEN** a row has dataset id `Instructions`
- **THEN** the upload is refused with a problem on that row's dataset id cell

### Requirement: Blank rows are ignored
A row whose every read cell is blank after whitespace normalization SHALL be skipped without stopping the scan and SHALL NOT count toward `rows_read` or the row cap, on the `Datasets` sheet, on indicators sheets and in a CSV, because the template ships formatted empty rows.

#### Scenario: Gap between rows
- **WHEN** the `Datasets` sheet has data in rows 2 and 6 and rows 3 to 5 are blank
- **THEN** both datasets are loaded and `rows_read` is 2

### Requirement: Files following the retired template are refused
A workbook or CSV carrying a column the current template dropped, `Excluded regional values` or `Relevant indicators not present` (in either spelling), or a workbook whose `Datasets` sheet carries `Indicators coverage`, SHALL be refused in full with a message naming the column and listing the expected columns. The values were authored, so dropping the column silently would be data loss.

#### Scenario: Old workbook
- **WHEN** the `Datasets` sheet has `Excluded regional values` in column J
- **THEN** the file is refused naming that column, and nothing is saved

#### Scenario: Indicators column on the Datasets sheet
- **WHEN** the `Datasets` sheet has an `Indicators coverage` column
- **THEN** the file is refused, pointing to one indicators sheet per dataset

### Requirement: A CSV carries indicators as a JSON list
A CSV SHALL be flat: the nine template columns plus `indicators_coverage`, holding a JSON list of `{name, units}`; a blank cell is an empty list. Headers SHALL match the template text or the field name (snake_case, spaced or camelCase), ignoring case. The flat `name (units); ...` text SHALL be refused as a problem on its row. A CSV without the column SHALL leave stored indicators as they are.

#### Scenario: Older flat text
- **WHEN** row 3's `indicators_coverage` is `GDP (USD); CPI (index)`
- **THEN** the upload is refused with a problem on row 3 asking for a JSON list

#### Scenario: Export round trip
- **WHEN** a records export CSV is uploaded to another channel
- **THEN** its records and indicators are loaded unchanged

### Requirement: An indicator is a name and its units
Each indicator SHALL be `{name, units}` with a non-empty `name`. A blank unit SHALL be accepted on upload, because units are required only by the indexing check set (`discovery-indexing`). An indicators sheet row giving units without a name, or a CSV indicator with an empty or missing `name`, SHALL be a problem naming its cell or indicator position.

#### Scenario: Missing unit
- **WHEN** sheet `GDP` lists indicator `Real GDP` with a blank `Units of measure`
- **THEN** the record is stored with that indicator and empty units

#### Scenario: Units without a name
- **WHEN** row 4 of sheet `GDP` has units `percent` and no indicator
- **THEN** the upload is refused with a problem at cell `'GDP'!B4`

### Requirement: Indicators per dataset are capped
A dataset SHALL list at most `DISCOVERY_UPLOAD_MAX_INDICATORS_PER_DATASET` indicators (default 50000). An indicators sheet over the cap SHALL fail its file as unreadable, read no further, because a compressed sheet can expand far beyond the byte cap; a CSV row over the cap SHALL be a problem on that row.

#### Scenario: Sheet over the cap
- **WHEN** the cap is 50000 and sheet `TRADE` lists 50001 indicators
- **THEN** the upload is refused with a problem for that file naming the sheet and the limit
