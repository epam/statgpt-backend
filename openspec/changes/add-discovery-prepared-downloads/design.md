## Context

See `proposal.md` for motivation and the two delta specs for the behavior contract.

- `statgpt/admin/routers/discovery_datasets.py` currently calls `AdminPortalDiscoveryDatasetService.export_discovery_datasets` and returns rendered bytes. Existing router tests cover parameter validation and file headers; rendering and errors already have their own implementation and handlers.
- `AttachmentsStorage` supports byte upload, deletion and paginated listing, but its existing listing helper accumulates all pages into one list. `open_file_stream` supplies an async stream and explicit close callback; the channel archive download router's background-callback pattern does not guarantee closure when streaming fails.
- Existing import/export `Job` records have no owner or expiration and represent a different lifecycle. `User.id` already comes from the verified token's configured identity claim.
- `fix_statuses.py` isolates recovery operations with independent sessions and exception logging. It runs in `FIX_STATUSES` and `INIT`; the deployed daily CronJob is supplied context, not something this change needs to modify.

## Goals / Non-Goals

**Goals:** Keep rendering shared between direct and prepared exports; separate HTTP handling, retrieval metadata and storage operations; make retrieval replica-independent and cleanup retryable.

**Non-Goals:** Generalizing every export into a job framework, adding a durable queue or cache service, changing authentication claims or admin channel authorization policy, or investigating the frontend. The proposal defines the feature scope.

## Decisions

### API and shared rendering

Add the following routes to the existing channel-scoped discovery router:

- `POST /channels/{channel_id}/discovery-datasets/exports`: same query parameters as direct export, no request body; return 201 with a typed response containing UUID `id` and UTC `expiresAt`.
- `GET /channels/{channel_id}/discovery-datasets/exports/{download_id}/download`: stream the prepared file after validation; declare 404, 410 and 502 responses in OpenAPI, alongside existing authentication/parameter errors.

Keep the runtime behavior of `GET .../export` unchanged, but mark its OpenAPI operation with `deprecated=True` and recommend the preparation/download flow in its operation description and `statgpt/admin/README.md`. Leave the preparation and prepared-download operations non-deprecated. Do not set a removal deadline or add deprecation/Sunset headers. Share construction of `DiscoveryExportFilters` and its current query-level validation with the POST so existing 400/404/422 details do not drift. Rendering continues through `export_discovery_datasets`; verify the channel exists before preparation and handle its concurrent removal during metadata persistence as described below. This preserves the existing GET runtime contract, including its edge cases, and avoids a second writer or a new async job lifecycle.

Use a dedicated Admin prepared-download service in `statgpt/admin/services/prepared_downloads.py` for persistence, authorized lookup and maintenance. The router orchestrates discovery rendering and this service. New inner-layer failures are domain exceptions registered in `exception_handlers.py`, not HTTP exceptions in the new service. Storage errors map to sanitized 502 JSON; preserve the current handling of unrelated database/internal failures rather than disguising them as storage errors.

### Metadata and channel deletion

Add `PreparedDownload` in `statgpt/common/models/models.py`, backed by `prepared_downloads`:

| Field | Mapping |
| --- | --- |
| `id` | UUID primary key generated with UUID4 |
| `owner_user_id` | Non-null string from authenticated `User.id` |
| `channel_id` | Channel foreign key, nullable only after channel deletion, `ON DELETE SET NULL` |
| `file_path` | Non-null DIAL relative file URL |
| `filename`, `media_type` | Non-null download headers derived from the selected format |
| `created_at`, `updated_at` | Existing timezone-aware `DateMixin` fields |
| `expires_at` | Non-null timezone-aware timestamp with an index |

Use `DateMixin` and the SQLAlchemy `Base`, not the integer-ID `DefaultBase`. Export the model through `common/models/__init__.py`; use a dedicated Pydantic response rather than the integer-ID `DbDefaultBase` or raw dictionary mutation.

Preparation always sets a real channel ID. `ON DELETE SET NULL` prevents this short-lived metadata from blocking channel deletion and preserves the file reference for subsequent cleanup; do not add an ORM cascade that deletes the row with the channel. Null channel scope cannot match a download request, so deleted channels cannot retrieve their snapshots. Deleting source records does not invalidate an already prepared snapshot while its channel still exists.

### Storage and preparation lifecycle

Store bytes using existing `AttachmentsStorage.put_file` and `SystemUserAuthContext` under the dedicated `prepared-downloads/<uuid>.csv` or `.xlsx` prefix in the system key's resolved DIAL bucket/appdata namespace. Do not use `exports`, which is swept independently by existing archive cleanup. Do not expose DIAL paths, credentials or public storage URLs in preparation responses; callers derive the API URL from the returned ID.

Render first, upload second, then capture one UTC preparation timestamp and commit the complete metadata with `expires_at = prepared_at + timedelta(hours=1)`. Return success only after commit. Keep the one-hour window and storage prefix as backend constants; no new environment variable is needed for the agreed fixed window. A database failure, upload failure or cancellation must not return an identifier; compensate a completed upload with best-effort deletion when metadata cannot be committed, preserving the original failure if compensation also fails. A hard crash can still leave an orphan, handled by maintenance below.

Handle a channel disappearing before metadata persistence as a domain not-found error mapped to 404, including a metadata insert rejected by the channel foreign key after concurrent deletion. Roll back a failed metadata transaction before resolving that failure; distinguish the missing-channel case from unrelated integrity/database errors, which retain their existing handling. Compensate a completed upload through the same failure path and issue no identifier. `ON DELETE SET NULL` protects existing metadata, not an insert referencing a channel already deleted. Use short metadata transactions and do not hold channel locks across rendering or storage calls.

The service takes the authenticated `User` for identity and an `AuthContext` for storage operations; do not use the shared system API key as the owner or persist bearer tokens. This reuses the current authentication path instead of adding token parsing or signed bearer download URLs. OIDC-disabled deployments retain the existing `Anonymous` identity and cannot provide distinct-user isolation.

### Lookup, expiration and streaming

Query by UUID, owner and channel together before any file I/O. A missing row or scope mismatch becomes the same 404; an authorized row with `expires_at <= now` becomes 410 with `Prepared download has expired. Prepare a new export.` A row removed by maintenance is simply unknown and returns 404. Invalid UUID syntax retains normal request validation behavior. Use timezone-aware UTC comparisons and do not update the deadline on reads.

After authorized lookup, the router opens `open_file_stream` using the validated internal path and uses stored headers. Close opened storage resources through its close callback in a response-level `finally`, with cancellation-safe awaited cleanup on completion, disconnect, cancellation, and send or iterator failure. A background callback alone is insufficient because streaming exceptions can bypass it. Test closure on completion, client disconnect/cancellation, response-send failure and mid-stream storage failure; do not buffer the full file during retrieval. Map storage-not-found to 404 and other opening failures to 502. Return `Cache-Control: no-store` on preparation and file responses. Expiration is checked when admitting a request, not by aborting a transfer already underway. Storage/network failures after headers are sent cannot become JSON errors.

### Daily maintenance and abandoned uploads

Add a separate cleanup wrapper in `fix_statuses.py`, using the existing session/MSI contexts and system storage auth, after existing recovery operations. Do not rename entry points, alter the CronJob schedule, call `clear_jobs`, or broaden stuck-job recovery.

Process expired rows in bounded, deterministically ordered batches. Delete a file first; treat storage 404 as success; then delete/commit its metadata. Keep rows when deletion fails, log failures with `_log.exception`, and advance the batch cursor even for failed rows so one failure cannot cause an infinite loop or starve later rows. A later run can complete cleanup after file deletion or database deletion failures. Operations must be idempotent if scheduled and initialization runs overlap; use short database transactions rather than holding row locks across storage calls.

After row-based cleanup, sweep only recognized UUID CSV/XLSX filenames in `prepared-downloads`. Add an incremental page or async-iterator interface to `AttachmentsStorage` and process files and metadata-reference checks with memory bounded by the current page/batch, rather than materializing the entire namespace. Preserve the existing list-returning helper for its current callers. Reclaim only files with a known storage timestamp older than 24 hours and no current metadata reference, checking references immediately before deletion. This conservative orphan grace is an implementation default, not a longer access window. Preserve recent files and timestamp-less files. Continue past individual orphan deletion errors, and isolate listing/cleanup failures from status recovery. Do not add another background job type or scheduler.

## Risks / Trade-offs

- [Daily cleanup lags expiration, or maintenance fails] -> The API always enforces the one-hour deadline; storage retention can be longer and deletion failures remain retryable.
- [DIAL and PostgreSQL are not one transaction] -> Commit before issuing success, compensate known failures and sweep aged abandoned files in the dedicated namespace.
- [Preparation adds storage latency and remains request-bound] -> Reuse the existing synchronous renderer and bytes upload; do not introduce polling solely for error handling.
- [Cleanup overlaps downloads] -> Preserve unexpired rows; admit streams based on the deadline at lookup. A stream admitted near expiry can encounter storage deletion or transport failure, like any other interrupted download.
- [Missing storage timestamps] -> Skip unknown-age orphans rather than risk deleting a live file; log for operational follow-up.
- [Backend-only delivery does not finish the UI acceptance criteria of #833] -> Document the API contract and explicitly leave frontend adoption and session-error presentation to a separate change.

## Migration Plan

1. During implementation, add the model, run Alembic `revision --autogenerate`, stop and report if it fails, review the generated migration and set `Versions.ALEMBIC_TARGET_VERSION` to its revision. Do not hand-create a migration.
2. Verify upgrade/downgrade and the channel `SET NULL` behavior against the integration test database. Existing channels require no backfill.
3. Apply the migration through the admin image before serving new endpoints or running updated maintenance. Existing direct-export consumers need no changes; the daily maintenance deployment only needs the updated image, not schedule/config changes.
4. If rollback is required, stop new preparation, allow active identifiers to expire and run cleanup successfully before downgrading the schema and deploying the prior code. Review any failed deletions/abandoned storage files before removing their metadata; do not silently discard cleanup references.
