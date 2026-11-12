# Proposal

## Why

Agencies deliver their Grade C discovery index as several workbooks (per agency, per domain, per person who filled the template), but `POST /{channel_id}/discovery-datasets/upload` and `statgpt discovery upload` take exactly one file. That is not only tedious: `replace` reconciles against a single file, so making a channel match a set of files is impossible without merging them by hand; duplicates across files go unnoticed; and the per-cell error report covers one file at a time. Tracked in issue #658.

## What Changes

- The upload endpoint accepts one or more files in the existing multipart field `file` (repeated), mixing `.xlsx` and `.csv`. A request with a single `file` part keeps working unchanged.
- All files are parsed, their rows merged into one candidate set, and that set is reconciled with the channel in one transaction. `mode` applies to the union: `upsert` keeps records no file mentions, `replace` deletes them.
- All or nothing across the batch: a 400 lists the problems of every file, and nothing is saved. Each `DiscoveryPayloadProblem` gains a `file` field naming the file it was found in.
- A file that cannot be read, exceeds the byte cap, or exceeds `max_rows` is reported as a problem of that file, without stopping the other files from being checked. **BREAKING** (minor): an oversized file now gets a 400 with an itemized problem instead of a bare 413.
- The same (agency, dataset id) key in two files refuses the batch, naming both files and rows. Two parts with the same file name are refused before parsing.
- A new setting `DISCOVERY_UPLOAD_MAX_FILES` (default 20) caps the number of files per request. `max_rows` and the byte cap stay per file.
- `DiscoveryUploadSummary` keeps its channel-level counters for the whole batch (`rows_read` becomes the total) and gains `files`: a per-file `{file, rows_read}` breakdown.
- CLI `statgpt discovery upload`: `--file` can be repeated and takes every path that follows it until the next option, so a glob expanded by the shell works; a directory expands to its `.xlsx` / `.csv` files and a quoted glob is expanded by the CLI. The result shows the aggregate summary and the per-file rows; problems show their file. Interactive selection stays single-file.
- CLI `content init`: uploads a client's discovery folder as one all-or-nothing request per channel instead of one request per file.
- CLI: arguments passed from the system shell keep their quoting, so paths containing spaces survive (`shlex.join` instead of `" ".join`).

## Capabilities

### New Capabilities

- `discovery-dataset-upload`: loading discovery dataset (Grade C) files into a channel through the admin API and the CLI - accepted files, reconciliation modes, the all-or-nothing error report, limits, and the upload summary.

### Modified Capabilities

None - no specs exist yet.

## Impact

- Admin API: `POST /{channel_id}/discovery-datasets/upload` (request accepts repeated `file`; response and 400 body gain fields; oversized file 413 -> 400).
- Code: `statgpt/admin/routers/discovery_datasets.py`, `statgpt/admin/services/discovery_dataset.py`, `statgpt/admin/services/exceptions.py`, `statgpt/admin/settings/discovery.py`, `statgpt/common/schemas/discovery_dataset.py`, `statgpt/cli/commands/base.py`, `statgpt/cli/commands/discovery.py`, `statgpt/cli/commands/content.py`, `statgpt/cli/shared/admin_client.py`, `statgpt/cli/shared/discovery.py`, `statgpt/cli/__init__.py`.
- Docs: `statgpt/admin/README.md` (new env var), `statgpt/cli/README.md` (`discovery upload`).
- Unaffected: the channel import path (archive with `discovery_datasets.csv`), which keeps its own single-file handling.
- Admin frontend (separate repo): can keep sending one file; multi-file selection, the `file` prefix on problems and the per-file table are follow-up work there.
