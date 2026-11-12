# Tasks

## 1. Schemas and settings

- [x] 1.1 Add `file: str | None = None` to `DiscoveryPayloadProblem`, add `DiscoveryUploadFileSummary { file, rows_read }`, add `files: list[DiscoveryUploadFileSummary]` to `DiscoveryUploadSummary`, and export the new model from `statgpt/common/schemas/__init__.py`; verify by importing `schemas.DiscoveryUploadFileSummary` in a unit test and asserting a default summary has `files == []`
- [x] 1.2 Add `max_files: int = Field(default=20, ge=1)` to `DiscoveryUploadSettings` and document `DISCOVERY_UPLOAD_MAX_FILES` in `statgpt/admin/README.md`, including the note that the ingress body limit must allow `max_files x max_file_size_bytes`; verify a settings test reads the env var
- [x] 1.3 Make `_render_problem` in `statgpt/admin/services/exceptions.py` prefix the file name when a problem has one; verify a unit test of `str(DiscoveryPayloadError(...))` shows `a.xlsx D14: ...` and that problems without a file render as before

## 2. Batch upload in the service

- [x] 2.1 Add `file` to `_RawRecord` and `_DiscoveryCandidate`, pass it through `_build_candidates_from_rows` and `_problems_from_validation_error`; verify `tests/unit/admin/services/test_discovery_upload.py` asserts schema problems carry the file name
- [x] 2.2 Make `_duplicate_problems` / `_locations` name files when candidates come from different files, and set `cell` only when all occurrences share one file; verify unit tests cover in-file (`Rows 2, 3`) and cross-file (`a.xlsx row 2, b.csv row 7`) duplicates
- [x] 2.3 Change `DiscoveryDatasetService.upload` to take a list of uploaded files: refuse more than `max_files` and duplicate file names with `DiscoveryPayloadError` before reading, then read and parse each file in order, turning `DiscoveryUploadTooLargeError` / `DiscoveryUploadFormatError` into a problem of that file, stamping `file` on parser problems, merging candidates, and running one `_upsert`; verify unit tests for: two valid files, replace deleting a record no file mentions, an unreadable file plus a file with a row problem (both reported), an oversized file reported as a problem, too many files, duplicate file names, and `max_rows` applied per file
- [x] 2.4 Fill `summary.files` in request order and `summary.rows_read` as their total; verify the per-file breakdown in a unit test
- [x] 2.5 Change the router to `file: list[UploadFile]` and update its docstring for batch semantics; verify a router test sending one `file` part still returns the old counters and a test sending two parts returns `files` with two entries; check that the OpenAPI schema shows `file` as an array of binaries

## 3. CLI argument parsing

- [x] 3.1 Add `multiple: bool = False` to `CommandArg` and make `Command.parse_args` collect greedy, repeatable values into a list (stop at option-like tokens, require at least one value); verify new cases in `tests/unit/cli/commands/test_base.py`: `--file a b`, `--file a --file b`, `--file a -c ch`, `--file` with no value, and single-value args unchanged
- [x] 3.2 Build the direct-execution command string with `shlex.join` in `statgpt/cli/__init__.py`; verify a test that a path containing a space reaches the handler intact

## 4. CLI discovery upload and content init

- [x] 4.1 Change `AdminClient.upload_discovery_datasets` to take `file_paths: Sequence[str]` and post one `file` part per path; verify `tests/unit/cli/shared/test_admin_client_discovery.py` asserts the multipart body carries every file under the `file` key
- [x] 4.2 Make `--file` on `discovery upload` multi-value and expand directories (via `is_upload_file`), glob patterns and plain paths in the handler, de-duplicated and sorted, failing before any request on missing paths or values that yield no file; keep interactive selection single-file and say "absent from the files" in the replace confirmation; verify handler unit tests for a directory with a lock file, a quoted glob, a repeated flag, and an empty pattern
- [x] 4.3 Add a `File` column to `render_payload_problems` (only when some problem has a file) and a per-file table to `render_upload_summary` (only when more than one file); verify rendering tests that capture the console output
- [x] 4.4 Switch `content init`'s `_upload_discovery_datasets` to one upsert request per channel with all folder files, recording one `BatchReport` item per channel; update `tests/unit/cli/commands/test_content_discovery_upload.py` to assert one call per channel and an all-or-nothing failure
- [x] 4.5 Update the `discovery upload` section of `statgpt/cli/README.md` (repeatable and greedy `--file`, directories, globs, per-file output) and the CLI table in `CLAUDE.md` if its description changes; verify the documented examples parse with the new `parse_args`

## 5. Integration check

- [x] 5.1 Run `make lint` and `make test_unit` and verify both pass
- [x] 5.2 Against a local admin API, upload two template files (`.xlsx` and `.csv`) with `statgpt discovery upload --mode replace --file <dir>` and verify the summary, then introduce a cross-file duplicate and verify the 400 table names both files and nothing changed in the channel
