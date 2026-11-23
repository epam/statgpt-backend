## ADDED Requirements

### Requirement: Maintenance removes expired prepared downloads independently of status recovery
`ADMIN_MODE=FIX_STATUSES` and the maintenance phase of `ADMIN_MODE=INIT` SHALL remove prepared-download files and metadata whose explicit expiration deadline has passed, without changing existing stuck-job recovery rules. Cleanup SHALL use bounded batches, preserve unexpired downloads and unrelated import/export files, and process each expired file before removing its metadata. A missing file SHALL count as already removed. Failed file deletions SHALL retain their metadata for retry on a later run; one failed deletion SHALL NOT prevent other eligible downloads from being cleaned. A cleanup failure SHALL be logged and SHALL NOT prevent status recovery. File access expiration SHALL remain independent of this cleanup, and this change SHALL NOT introduce a new scheduler or change the existing daily maintenance schedule.

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

### Requirement: Maintenance reclaims abandoned prepared-export files conservatively
Maintenance SHALL also reclaim generated prepared-export files left without retrieval metadata by interrupted preparation or failed compensation. It SHALL restrict this sweep to the dedicated prepared-download storage namespace and known generated filename pattern, require an available file timestamp older than 24 hours, and confirm the file has no metadata reference before deletion. Listing and processing SHALL be incremental, with memory bounded by the current page/batch rather than the total number of files in the namespace. Files referenced by any metadata, recent files, files without an available age and unrelated files SHALL NOT be removed by the orphan sweep. The 24-hour orphan grace period SHALL NOT extend the one-hour download window.

#### Scenario: Upload completed before preparation crashed
- **WHEN** maintenance finds a prepared-export file older than 24 hours with no retrieval metadata
- **THEN** it removes the abandoned file without touching unrelated exports

#### Scenario: Orphan backlog spans multiple listing pages
- **WHEN** the prepared-download namespace contains multiple pages of files with eligible orphans in the first page
- **THEN** maintenance begins processing and deleting those orphans before listing the entire namespace, without accumulating files or metadata-reference checks across all pages

#### Scenario: Recent or referenced file
- **WHEN** maintenance finds a recent unreferenced file or an old file still referenced by download metadata
- **THEN** the orphan sweep leaves the file unchanged
