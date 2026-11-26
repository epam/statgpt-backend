# Tasks

## 1. Record-publishing transaction boundary

- [x] 1.1 Update `tests/unit/admin/services/test_discovery_indexing_job.py` to assert record results are committed before vocabulary publishing begins, including the vocabulary-failure path; verify the new assertion fails against the current implementation.
- [x] 1.2 Commit the service session after record publishing completes in both branches of `_publish_records`, before vocabulary publishing or returning without a vocabulary target; update the method documentation and verify the unit suite covers successful publishing, vocabulary failure, no vocabulary target and metadata-schema rejection before document writes.
- [x] 1.3 Add a unit regression for a failing record-stage commit; verify vocabulary publishing is not invoked and the existing failure handler rolls back and marks the job `FAILED` with the commit failure reason.

## 2. Database-backed rollback regression

- [x] 2.1 Add `tests/integration/services/test_discovery_indexing_job.py` with isolated database fixtures, a test-database guard and cleanup, real service/publisher execution, and mocked ingestion operations and summarization; verify a vocabulary document returned in `error` leaves the job `FAILED` and fresh-session reads retain successful uploads as `INDEXED`, withdrawals as `NEW`, and failed uploads as `FAILED`, including each record's expected `indexed_at` and `index_error`.
- [x] 2.2 Extend the database-backed regression to a vocabulary transport failure and successful publishing from both `NEW` and `OUTDATED` starting states; verify persisted results and failure reasons through a fresh session and independently assert document uploads and withdrawals are not undone by database rollback.

## 3. Cross-cutting verification

- [x] 3.1 Run the discovery indexing job, record publisher and reference-area publisher unit suites plus the new integration suite using the documented test database environment; verify they pass and report any unavailable infrastructure rather than treating unrun tests as successful.
- [x] 3.2 Run `make format` and `make lint`, using the established Poetry equivalents if the local `python3` limitation persists; verify formatting and lint checks pass, and validate this change with `openspec validate preserve-discovery-publish-statuses --strict`.
