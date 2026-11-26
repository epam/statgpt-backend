# Design

## Context

See `proposal.md` for motivation. `DiscoveryPublisher.publish` mutates ORM records and leaves persistence to its caller. `_publish_records` currently invokes the record and vocabulary publishers consecutively; `_fail_job` rolls back their shared session. Validation and per-record summarization already have earlier commit boundaries.

The existing vocabulary-failure unit test uses `SimpleNamespace` records and an `AsyncMock` session whose rollback does not restore persisted state. The integration suite provides real PostgreSQL sessions and examples of checking persisted values through a separate session.

## Goals / Non-Goals

**Goals:**
- Keep transaction ownership in the indexing job service.
- Make the completed record stage independent of the later vocabulary stage without changing publisher interfaces.
- Verify both stage ordering and persistence after a real rollback.

**Non-Goals:**
- Commit individual publishing attempts or preserve a record stage interrupted before completion.
- Change failed-job counters, publisher reconciliation, summarization or metadata-schema requirements.
- Add database migrations, dependencies or compensation for external document changes.

## Decisions

### Commit at the completed record-publishing boundary

In `_publish_records`, await the service session's commit immediately after `publisher.publish` completes, before invoking vocabulary publishing. Apply the same boundary when no vocabulary target is configured. Keep both metadata-schema checks before either publisher writes documents.

The service already owns the session, so no persistence callback or session dependency is added to either publisher. A commit after `_publish_records` returns is too late because vocabulary publishing is inside that method. Removing `_fail_job`'s rollback would instead persist incomplete failed-stage work and break database-error recovery.

If this commit fails, do not start vocabulary publishing; let the existing failure path roll back and mark the job failed. If vocabulary publishing fails after the commit, retain `_fail_job` unchanged: its rollback cannot discard the completed record stage.

### Separate ordering checks from database durability checks

Update the unit spy to assert a record-results commit occurs before vocabulary publishing, rather than treating surviving in-memory mutations as proof of persistence. Cover a failing boundary commit, metadata-schema rejection, successful publishing and the no-vocabulary path.

Add `tests/integration/services/test_discovery_indexing_job.py`. Run the real indexing service and publishers with real ORM records and database transactions, mocking ingestion-client operations and summarization to avoid external services. Exercise both a vocabulary document returned in `error` and a vocabulary transport failure after record publishing. Reload records and the job in a fresh session; assert successful uploads, withdrawals and failed uploads retain all three indexing fields, while the job is `FAILED` with its vocabulary failure reason. Track mocked document operations independently of database rollback. Use isolated fixtures with a test-database guard and cleanup, following `test_prepared_downloads.py`.

## Risks / Trade-offs

- A database commit can fail after external documents changed -> leave vocabulary unstarted and report job failure; cross-system atomicity is outside this fix.
- A process can die during record publishing before the boundary -> partial record-stage persistence remains outside scope.
- Database-backed regressions require the test database -> run them with the documented integration environment; unit ordering tests remain available without containers.

## Migration Plan

Deploy as an ordinary backend update; no migration or configuration change is required. Existing stale records are reconciled by a subsequent indexing run. Reverting the code restores the previous rollback behavior without a data migration.
