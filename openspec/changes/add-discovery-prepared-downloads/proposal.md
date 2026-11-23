## Why

Discovery dataset export currently returns either a file or an error from the same endpoint, so a browser download can show raw error JSON instead of a notification. Issue #833 needs a backend preparation request that reports export errors before a separate authenticated download request.

## What Changes

- Add synchronous preparation for discovery CSV/XLSX exports, with the existing selection parameters and error statuses/texts, returning an opaque identifier and expiration only after the complete file is stored.
- Persist prepared-download metadata in PostgreSQL and generated files in the existing DIAL storage; identifiers belong to the requesting user and channel and expire one hour after preparation.
- Add an authenticated file endpoint that checks ownership, channel and expiration and allows repeated downloads before expiry without regenerating the export.
- Preserve the existing direct-export GET endpoint and channel archive import/export behavior.
- Add expired-download cleanup to the existing `fix_statuses` maintenance entry point, also used during initialization, without coupling cleanup failures to status recovery. The deployed daily schedule remains unchanged.
- Document the backend API and its error contract. Frontend investigation, implementation and end-to-end UI verification are out of scope; this change delivers the backend portion of #833 only.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `discovery-dataset-records`: Add preparation and retrieval of user/channel-scoped discovery exports with a one-hour download window while retaining direct export.
- `admin-background-jobs`: Extend the existing maintenance entry point with safe, retryable cleanup of expired prepared downloads, independent of stuck-job recovery.

## Impact

- Admin discovery export routes, service orchestration, domain exceptions/HTTP handlers, response schemas and API documentation.
- A new prepared-download SQLAlchemy model and Alembic migration, plus the migration target version. Generate the migration with autogenerate during implementation; stop if generation fails.
- Existing DIAL upload/stream/delete helpers, authenticated `User.id` and maintenance script integration; no new storage service, background queue or frontend changes.
- Tests covering compatibility, preparation errors, stored snapshots, identity/channel isolation, expiration, stream closure, cleanup retries and database lifecycle.
