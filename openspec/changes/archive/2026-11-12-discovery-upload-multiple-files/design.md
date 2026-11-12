# Design

## Context

See proposal.md - Why. Today's single-file flow in `DiscoveryDatasetService`:

```
read_upload(file)            byte cap      -> raises DiscoveryUploadTooLargeError (413)
parse_discovery_file()       unreadable    -> raises DiscoveryUploadFormatError   (400, no problems)
                             > max_rows    -> raises DiscoveryUploadFormatError
                             per-cell      -> ParsedFile.problems (collected)
_build_candidates_from_rows  schema        -> problems (collected)
_duplicate_problems          same key      -> problems (collected)
_raise_for_problems          any problem   -> DiscoveryPayloadError (400, capped)
_upsert(candidates, delete_absent)          one commit
```

`_upsert` already works on any sequence of candidates and needs no change. The work is in the steps before it: they assume one file and turn file-level failures into exceptions that end the request.

The CLI turns `sys.argv` into a string with `" ".join` and `Command.parse_args` splits it again with `shlex`. A repeated option overwrites the earlier value, and a token that is not an option is an error.

## Goals / Non-Goals

**Goals:**
- One request, one reconciliation, one error report for any number of files up to the cap.
- No change on the wire for callers that send one file, except an oversized file now getting 400.
- Memory bounded by one file's bytes at a time, plus the parsed rows of the batch.

**Non-Goals:**
- Archive (zip) uploads.
- Changing how the channel import reads `discovery_datasets.csv`.
- A multi-select interactive file picker in the CLI.
- Splitting the problem budget fairly between files.
- Admin frontend changes (separate repository).

## Decisions

### Keep the field name `file`, typed `list[UploadFile]`
Multipart allows one key to repeat, so FastAPI collects every `file` part into the list and a request with one part is a list of one. The admin UI (`FormData.append('file', ...)`), the CLI and `content init` keep working without a coordinated release.
- *Alternative: `files`.* Cleaner name, but it breaks every current caller and forces the backend and frontend to ship together.
- *Alternative: accept both `file` and `files`.* Two fields in the OpenAPI schema for one concept.

### Process files one at a time, collecting instead of raising
`upload` takes a list of `(filename, UploadFile)` pairs. It first runs the request-level checks: the number of files against `max_files`, and duplicate file names. Either failure raises `DiscoveryPayloadError` before anything is read. Then, for each file in request order:
1. `read_upload` reads it (the byte cap still raises `DiscoveryUploadTooLargeError` inside).
2. `parse_discovery_file` runs in `asyncio.to_thread` (`DiscoveryUploadFormatError` for an unreadable file or more than `max_rows`).
3. Either exception is caught and becomes a `DiscoveryPayloadProblem(file=name, message=str(e))`, and processing moves to the next file.
4. Parsed rows become `_RawRecord`s tagged with the file name. The file's bytes are dropped before the next file is read.

Both exception classes stay. `read_upload` and `parse_discovery_file` keep raising them, and the channel import path still raises `DiscoveryUploadFormatError` for a bad archive. Only the upload loop turns them into problems. The 413 handler stays for any other caller, but the upload endpoint no longer reaches it.
- *Alternative: parse files concurrently.* Peak memory would be the sum of all files instead of the largest one, and parsing is CPU-bound in threads under the GIL anyway.

### Tag candidates and problems with their file
`_RawRecord` and `_DiscoveryCandidate` gain `file: str | None`. `_problems_from_validation_error` takes `file` and sets it on each problem. Problems from `ParsedFile.problems` are copied with `file` set (`model_copy(update=...)`), so the parser module stays unaware of batches. JSON payloads (`/bulk`, channel import) leave `file` as `None`.

`DiscoveryPayloadProblem` gains `file: str | None = None`. `_render_problem` in `exceptions.py` prefixes the file (`a.xlsx D14: ...`), so log lines and `str(exc)` can also be traced.

### Duplicate detection over the merged candidates
`_duplicate_problems` runs once on all candidates, so duplicates within one file and across files are found the same way. `_locations` renders `Rows 2, 3` when every candidate is in the same file, and `a.xlsx row 2, b.csv row 7` otherwise. The problem's `file`, `row` and `cell` point at the first occurrence; `cell` lists every cell only when they are all in one file, because cell references from different files mean nothing side by side.

### Problem order and budget
Problems are reported grouped by file in request order, each file's problems in row order, followed by duplicates. `max_reported_problems` covers the whole response, so a badly broken first file can push later files past the cap. The response is then marked `truncated`, and the fix-and-retry loop still makes progress.

### Summary shape
New `DiscoveryUploadFileSummary { file: str; rows_read: int }`. `DiscoveryUploadSummary` gains `files: list[DiscoveryUploadFileSummary] = Field(default_factory=list)`, and its `rows_read` becomes the sum over all files. There is no `rows_skipped`: blank rows are not counted anywhere today, and an all-or-nothing upload has no other kind of skipped row.

### Limits
New `DiscoveryUploadSettings.max_files: int = 20` (`ge=1`, env `DISCOVERY_UPLOAD_MAX_FILES`). `max_file_size_bytes` and `max_rows` stay per file. The parser's "Split it and upload the parts" advice becomes accurate once parts can be uploaded together. The worst case per request is about `max_files x max_rows` rows in one transaction, 200k with the defaults. A total-rows setting can be added later if that turns out to be too many.

### CLI multi-value arguments
`CommandArg` gains `multiple: bool = False`. For such an argument `parse_args`:
- starts the value as an empty list,
- after the option takes every following token that does not start with `-`, and requires at least one,
- appends to the list when the option repeats.

Option-like tokens (`-x`, `--x`) end the run, so `--file a b -c ch` works. `_execute_direct` builds the command string with `shlex.join(args)`, so quoting from the system shell survives the re-split. That fixes paths containing spaces, which exist today but are more likely with many paths.

`discovery upload` expands each value in its handler, in this order: an existing directory lists its direct children that pass `is_upload_file`; a value containing `*`, `?` or `[` goes through `glob.glob`; anything else is used as a path. The paths are made absolute, de-duplicated and sorted. A value that yields nothing, or a path that does not exist, is an error before any request is sent. With no `--file`, the interactive picker still returns one path.

`AdminClient.upload_discovery_datasets(channel_id, file_paths: Sequence[str], mode)` opens every file inside an `ExitStack` and posts a list of `("file", (basename, fh, "application/octet-stream"))` tuples.

Rendering: `render_payload_problems` gains a `File` column, shown only when some problem has a file, so `/bulk`-style reports look as before. `render_upload_summary` keeps its counters table and adds a per-file table when `files` has more than one entry. The `replace` confirmation text says "absent from the files".

### Content init: one request per channel
`_upload_discovery_datasets` sends all the folder's files in one upsert call per target channel. On success it prints the summary line and each file's rows; on `DiscoveryPayloadError` it prints the problem table. `BatchReport` records one item per channel, labelled `<deployment_id>: <n> discovery file(s)`.

## Risks / Trade-offs

- [An oversized file changes from 413 to 400] -> The admin frontend does not branch on 413, and the CLI shows `detail.message` either way. Listed as BREAKING in the proposal.
- [Ingress body limits now have to fit several files] -> A request can be up to `max_files x max_file_size_bytes` (200 MB with defaults); an ingress `client_max_body_size` below that gives a 413 from the proxy, not from the app. Document this next to `DISCOVERY_UPLOAD_MAX_FILES` in the admin README.
- [`content init` loses per-file isolation] -> A single bad file now blocks that channel's discovery load. This was chosen on purpose: the folder is meant to be the channel's collection, and partial loads are what #658 removes.
- [Greedy `--file` swallows a path starting with `-`] -> Unlikely for workbooks; `./-name.xlsx` gets around it.
- [Two different files with the same base name from different directories] -> Refused by the duplicate file name check; the CLI could catch it before sending, but the server check is authoritative.

## Migration Plan

Ship the backend and CLI together, since both are in this repository. Existing admin UI builds keep working because they send one `file` part and ignore the new response fields. The frontend adds multi-file support afterwards. Rollback is a redeploy of the previous version; there is no data model or migration change.
