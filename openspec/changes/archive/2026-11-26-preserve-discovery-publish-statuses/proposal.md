# Proposal

## Why

Issue #814 reports that a reference-area vocabulary publishing failure rolls back discovery records' indexing results even though their documents have already been uploaded or withdrawn. Administrators then see stale indexing statuses, timestamps and errors until a later run succeeds.

## What Changes

- Make the completed record-publishing stage durable before starting reference-area vocabulary publishing.
- Preserve each record's resulting `indexing_status`, `indexed_at` and `index_error` when vocabulary publishing fails, while continuing to mark the job `FAILED` with its failure reason.
- Replace the existing test's reliance on a no-op mocked rollback with stage-order assertions and database-backed regression coverage.
- Keep metadata-schema verification ahead of publishing to either application.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `discovery-indexing`: Specify that completed record-publishing results survive a subsequent vocabulary failure, alongside the already required preservation of published documents.

## Impact

- Implementation: `statgpt/admin/services/discovery_indexing_job.py`.
- Unit coverage: `tests/unit/admin/services/test_discovery_indexing_job.py`.
- Database-backed coverage: a new `tests/integration/services/test_discovery_indexing_job.py`, using the existing test database infrastructure and mocked external services.
- No API, configuration, dependency or database-schema changes. Job-counter preservation on failure, per-record publishing commits and external-document compensation are outside this change.
