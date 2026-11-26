## ADDED Requirements

### Requirement: Maintenance removes expired prepared downloads independently of status recovery
`ADMIN_MODE=FIX_STATUSES` and `ADMIN_MODE=INIT` maintenance SHALL clean expired prepared downloads in bounded batches, deleting files before metadata. Missing files SHALL count as removed; failed deletions SHALL retain metadata for retry without blocking other items or status recovery. Cleanup failures SHALL be logged. Active downloads, unrelated import/export files, recovery rules and the daily schedule SHALL remain unchanged. Access expiry SHALL remain independent of cleanup.

#### Scenario: Daily cleanup of an expired download
- **WHEN** maintenance runs after a prepared download's expiration
- **THEN** its file is removed before its metadata is removed

#### Scenario: Storage deletion fails for one download
- **WHEN** maintenance processes two expired downloads and deleting the first file fails
- **THEN** the first download's metadata remains for retry, cleanup proceeds with the second and stuck-job recovery still runs

#### Scenario: File already missing
- **WHEN** an expired download's file was removed before maintenance runs
- **THEN** maintenance removes its remaining metadata without treating the missing file as an error

#### Scenario: Row deletion failed after file deletion
- **WHEN** a previous maintenance run removed an expired file but could not remove its metadata, and maintenance runs again
- **THEN** the missing file is accepted as already removed and metadata deletion is retried

#### Scenario: Unexpired download and unrelated archive
- **WHEN** maintenance runs while an unexpired prepared download and an existing channel export archive are stored
- **THEN** neither file is removed by prepared-download cleanup

#### Scenario: Initialization without expired downloads
- **WHEN** `ADMIN_MODE=INIT` runs with no expired prepared downloads
- **THEN** existing status recovery completes and active downloads remain unchanged

#### Scenario: Existing maintenance entry points and schedule
- **WHEN** prepared-download cleanup is deployed
- **THEN** it runs through existing `FIX_STATUSES` and `INIT` maintenance without introducing a new scheduler or changing the daily schedule or stuck-job recovery rules

### Requirement: Maintenance reclaims abandoned prepared-export files conservatively
Maintenance SHALL incrementally reclaim unreferenced generated UUID CSV/XLSX files only from the prepared-download namespace, using memory bounded by the current page/batch. Deletion SHALL require a known storage timestamp older than 24 hours and an immediate check that no metadata references the file. Recent, referenced, unknown-age and unrelated files SHALL be preserved. This orphan grace SHALL NOT extend the one-hour access window.

#### Scenario: Upload completed before preparation crashed
- **WHEN** maintenance finds a prepared-export file older than 24 hours with no retrieval metadata
- **THEN** it removes the abandoned file without touching unrelated exports

#### Scenario: Orphan backlog spans multiple listing pages
- **WHEN** the prepared-download namespace contains multiple pages of files with eligible orphans in the first page
- **THEN** maintenance begins processing and deleting those orphans before listing the entire namespace, without accumulating files or metadata-reference checks across all pages

#### Scenario: Recent or referenced file
- **WHEN** maintenance finds a recent unreferenced file or an old file still referenced by download metadata
- **THEN** the orphan sweep leaves the file unchanged
