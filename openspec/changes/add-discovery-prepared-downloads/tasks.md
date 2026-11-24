## 1. Prepared-download metadata

- [x] 1.1 Add and export the UUID-keyed `PreparedDownload` model with owner, channel `ON DELETE SET NULL`, file metadata, timezone-aware expiration and its cleanup index; do not cascade metadata deletion with a channel.
- [x] 1.2 Generate an Alembic revision with `--autogenerate`, stop/report if generation fails, review the generated upgrade/downgrade and update `Versions.ALEMBIC_TARGET_VERSION`.
- [x] 1.3 Add the typed preparation response with `id` and `expiresAt`, plus backend constants for the one-hour window and dedicated prepared-download storage prefix.

## 2. Preparation and download API

- [x] 2.1 Share the existing export query/filter validation between direct export and the new preparation route without changing the direct GET contract or existing rendering/error texts.
- [x] 2.2 Implement prepared-file persistence in `statgpt/admin/services/prepared_downloads.py`, using `User` and `AuthContext`, existing DIAL byte uploads, post-upload UTC expiration and commit-before-success; compensate uploaded files on failed commit/cancellation without masking the original failure. Handle a channel disappearing before metadata persistence as a domain not-found error, including its foreign-key insert failure after rollback, without remapping unrelated database errors or holding channel locks across rendering/storage calls.
- [x] 2.3 Add `POST /channels/{channel_id}/discovery-datasets/exports` with channel existence checking, existing discovery rendering and a 201 preparation response that prohibits caching.
- [x] 2.4 Implement owner/channel-scoped lookup and fixed deadline validation before file I/O, with domain exceptions for unknown/mismatched downloads and expired downloads.
- [x] 2.5 Add `GET /channels/{channel_id}/discovery-datasets/exports/{download_id}/download`, reusing streaming helpers and stored headers, prohibiting caching and closing opened storage resources in a response-level `finally` with cancellation-safe awaited cleanup on completion, disconnect, cancellation, and response-send or storage-iterator failure; do not rely solely on a background callback.
- [x] 2.6 Register HTTP mappings for preparation channel-disappearance 404, download 404/410 and sanitized storage 502 errors, and declare preparation/file responses and error schemas in OpenAPI without changing unrelated error handlers.

## 3. Daily maintenance

- [x] 3.1 Implement bounded expired-row cleanup: file first, metadata second, missing-file success, per-item failure isolation and cursor advancement past failed rows; make overlapping runs/retries idempotent.
- [x] 3.2 Add an incremental page or async-iterator interface to `AttachmentsStorage`, preserving the existing list-returning helper for its current callers. Use it for a memory-bounded orphan sweep limited to recognized generated files in the dedicated namespace, older than 24 hours with known timestamps and no current metadata references; preserve recent, referenced, unknown-age and unrelated files.
- [x] 3.3 Integrate a separate cleanup wrapper into `fix_statuses.py` using existing session/MSI contexts and system storage authentication; isolate/log cleanup failures and preserve both `FIX_STATUSES` and `INIT` recovery behavior without scheduler changes.

## 4. Behavioral and database tests

- [x] 4.1 Extend router tests for both formats, defaults, filters, repeatable/limited IDs and unchanged 400/404/422 preparation details; assert rendering failures perform no storage upload and successful preparation exposes only the typed descriptor.
- [x] 4.2 Add service tests for one-hour UTC deadlines, persisted file metadata, upload/commit failures, compensation and cancellation, and retrieval of the original bytes after source records change; cover channel disappearance before metadata persistence returning 404 with upload compensation and unchanged handling of unrelated database failures.
- [x] 4.3 Add download tests for owner/channel mismatch, missing identifiers/files, deleted channels, equality/after-expiry 410, refreshed identity, authentication rejection before lookup, repeat retrieval without deadline extension and storage-opening 502; verify resource closure on completion, disconnect, cancellation, response-send failure before the first body chunk and mid-stream storage-iterator failure, without converting an interrupted transmission into JSON.
- [x] 4.4 Add cleanup tests for deletion ordering, missing files, file/row deletion retry, batches containing failures, active-file preservation, orphan age/reference guards and overlapping idempotent runs; verify cleanup/storage failures do not alter recovery behavior. Cover a multi-page orphan backlog with processing/deletion before the full listing is fetched and memory bounded by the current page/batch, and retain compatibility of the existing list-returning helper.
- [x] 4.5 Add integration coverage using the existing test database and mocked DIAL operations for metadata persistence across sessions, UUID/UTC mappings, channel deletion retaining a cleanup reference and eventual cleanup; cover concurrent channel deletion after upload but before metadata persistence, including the channel foreign-key failure, 404 mapping and upload compensation.
- [x] 4.6 Verify the autogenerated migration's upgrade/downgrade on the integration test database and confirm the migration target matches; report unavailable database prerequisites rather than skipping silently.

## 5. Documentation and validation

- [x] 5.1 Document the two-step backend contract in `statgpt/admin/README.md`, including endpoint parameters, one-hour deadline, authentication, retry/expiry errors, storage cleanup and the OIDC-disabled identity limitation; explicitly leave frontend adoption out of scope.
- [x] 5.2 Run `make format`, `make lint` and relevant unit/integration tests; use the equivalent Poetry formatter/linter commands if the local platform cannot run the Makefile targets, and report any failures or unavailable prerequisites.
- [x] 5.3 Validate the OpenSpec change and check implementation against both delta specs, retaining direct-export and channel archive compatibility without unrelated changes.

## 6. Legacy direct-export deprecation

- [x] 6.1 Mark the direct-export GET operation as deprecated in OpenAPI and recommend the preparation/download flow in its operation description and `statgpt/admin/README.md`; preserve availability and the existing runtime contract, without a removal deadline or additional response headers.
- [x] 6.2 Extend router/OpenAPI tests to assert the legacy GET's deprecation flag and replacement guidance, non-deprecated preparation/download operations and unchanged direct-file behavior; run relevant tests, formatting/lint checks and OpenSpec validation.
