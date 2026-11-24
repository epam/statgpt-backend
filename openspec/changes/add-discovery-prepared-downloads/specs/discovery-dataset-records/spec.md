## ADDED Requirements

### Requirement: Discovery export can be prepared before downloading
`POST /channels/{channel_id}/discovery-datasets/exports` SHALL synchronously prepare a discovery export using the direct-export endpoint's query parameters, defaults, selection limits and CSV/XLSX rendering rules. A successful preparation SHALL return 201 with JSON containing `id`, an opaque UUID download identifier, and `expiresAt`, a UTC expiration timestamp, only after the complete file and its retrieval metadata are stored. It SHALL NOT return file content, an internal storage URL or a polling job. An unknown channel, including one deleted during preparation before retrieval metadata is persisted, SHALL answer 404 without issuing an identifier. A completed upload whose metadata cannot be persisted SHALL receive best-effort deletion; compensation failure SHALL NOT mask the preparation error. Unrelated database failures SHALL retain their existing error handling. The preparation response SHALL prohibit caching.

#### Scenario: Prepare the default export
- **WHEN** an authenticated admin prepares an existing channel's export with no query parameters
- **THEN** the response is 201 JSON containing `id` and `expiresAt`, and the identified file is the same `discovery_datasets.csv` the direct export would produce

#### Scenario: Prepare a filtered validation workbook
- **WHEN** an admin prepares with `format=xlsx&include_validation=true&validation_status=INVALID`
- **THEN** the identified workbook contains exactly the invalid records and their stored validation verdicts, following the existing workbook rules

#### Scenario: Unknown channel
- **WHEN** an admin prepares an export for a channel that does not exist
- **THEN** the response is 404 and no download identifier is issued

#### Scenario: Channel deleted before metadata persistence
- **WHEN** the export file has been uploaded and another request deletes the channel before preparation persists its retrieval metadata
- **THEN** preparation answers 404, issues no identifier and attempts to delete the uploaded file, preserving the 404 if that compensation fails

### Requirement: Preparation reports export errors without issuing a download
Preparation SHALL preserve the direct-export endpoint's validation and rendering error statuses and texts, including 400 for records a workbook cannot hold, 404 for selected records outside the channel and 422 for invalid query parameters or combinations. Failed preparation SHALL NOT issue a usable identifier. Failure to store the complete file or its metadata SHALL NOT be reported as successful preparation; a storage failure SHALL answer 502 without exposing storage credentials or internal paths.

#### Scenario: Description exceeds the Excel cell limit
- **WHEN** an admin prepares an XLSX export of a record whose description exceeds 32,767 characters
- **THEN** the response is 400 JSON with the same error detail as the direct export and contains no download identifier

#### Scenario: Selected record is outside the channel
- **WHEN** an admin prepares channel 1 with `id=5&id=77` and record 77 is not in channel 1
- **THEN** the response is 404 with the existing missing-record text and no download identifier

#### Scenario: Invalid selection combination
- **WHEN** an admin prepares an export with `id=5&agency=NSO`
- **THEN** the response is 422 with the existing invalid-combination detail and no download identifier

#### Scenario: Storage unavailable
- **WHEN** the file cannot be stored after rendering succeeds
- **THEN** preparation answers 502 JSON and issues no download identifier

### Requirement: A prepared download belongs to its authenticated user and channel
`GET /channels/{channel_id}/discovery-datasets/exports/{download_id}/download` SHALL require the same admin authentication as preparation and SHALL permit retrieval only when the identifier belongs to the authenticated user's stable identity and the requested channel. Unknown identifiers, owner or channel mismatches, deleted channels and missing stored files SHALL answer 404. Authentication SHALL run before download lookup, and lookup SHALL disclose no file content, storage path or expiration information to another user or channel. Refreshing credentials for the same identity SHALL NOT invalidate an otherwise valid download. With OIDC disabled, the existing anonymous identity behavior SHALL remain; this mode SHALL NOT be described as providing per-user isolation.

#### Scenario: Another admin knows the identifier
- **WHEN** admin B requests the download prepared by admin A
- **THEN** the response is 404 and no file is read or returned

#### Scenario: Identifier used for another channel
- **WHEN** an admin requests a channel 1 identifier through channel 2's download endpoint
- **THEN** the response is 404 and no file is read or returned

#### Scenario: Expired authentication
- **WHEN** a caller requests a valid identifier with an expired authentication token while OIDC is enabled
- **THEN** the request is rejected with the existing authentication error before any download lookup or file retrieval

#### Scenario: Refreshed authentication
- **WHEN** the preparing admin retrieves an unexpired download using refreshed credentials with the same user identity
- **THEN** the file is returned without preparing it again

#### Scenario: Channel deleted after preparation
- **WHEN** a channel is deleted after an admin prepares its export and the admin requests that download
- **THEN** the response is 404 even if the generated file has not yet been physically removed

### Requirement: The download window is one hour and does not slide
A prepared download SHALL expire one hour after successful file preparation. `expiresAt` SHALL represent this fixed deadline. Retrieval SHALL be permitted only while the server's current UTC time is strictly before that deadline; equality SHALL count as expired. An expired identifier whose metadata still exists and matches the caller and channel SHALL answer 410 with JSON detail `Prepared download has expired. Prepare a new export.` without reading the file. An identifier whose metadata has been cleaned up SHALL answer 404. Repeated downloads SHALL neither consume the identifier nor extend its expiration. Expiration SHALL be enforced independently of maintenance frequency or physical file deletion.

#### Scenario: Repeated download
- **WHEN** the preparing admin downloads the same file twice before `expiresAt`
- **THEN** both requests return the file and the deadline remains unchanged

#### Scenario: Expiry boundary before cleanup
- **WHEN** the preparing admin requests a matching identifier exactly at `expiresAt` while its file and metadata still exist
- **THEN** the response is 410 with the expiration detail and the file is not read

#### Scenario: Identifier already cleaned up
- **WHEN** an admin requests an identifier after expired metadata has been removed
- **THEN** the response is 404

### Requirement: Retrieval returns the prepared snapshot
A successful download SHALL return 200 with the stored bytes, the corresponding CSV/XLSX media type and `Content-Disposition` naming `discovery_datasets.csv` or `discovery_datasets.xlsx`. It SHALL NOT reselect records, regenerate the export or revalidate workbook content. Prepared files and metadata SHALL remain retrievable across Admin Backend restarts and through different replicas sharing the configured database and file storage. File responses SHALL prohibit caching. A storage outage while opening a download SHALL answer 502 rather than returning success; failures after transmission starts cannot be converted into a new JSON response. Retrieval SHALL close opened storage resources on completion, client disconnect, cancellation, and response-send or storage-iterator failure, including failures before the first body chunk is sent.

#### Scenario: Records change after preparation
- **WHEN** a record is edited after its XLSX export is prepared and before it is downloaded
- **THEN** the downloaded bytes are those prepared before the edit

#### Scenario: Different replica serves the download
- **WHEN** preparation succeeds on one replica and the matching download request reaches another replica using the same database and storage
- **THEN** the prepared file is returned without relying on the first replica's memory or local files

#### Scenario: Client disconnect or request cancellation
- **WHEN** a client disconnects during retrieval or the response task is cancelled after storage resources have opened
- **THEN** those resources are closed despite the disconnect or cancellation

#### Scenario: Response send fails before the first body chunk
- **WHEN** storage resources have opened but sending the response fails before the first body chunk
- **THEN** those resources are closed even though body iteration did not complete

#### Scenario: Storage fails during transmission
- **WHEN** the storage iterator raises after part of the prepared file has been transmitted
- **THEN** opened storage resources are closed and the interrupted response is not replaced with a JSON error

### Requirement: Two-step export preserves direct-export compatibility
The existing `GET /channels/{channel_id}/discovery-datasets/export` SHALL remain available as a deprecated compatibility endpoint and retain its direct-file response, query parameters, defaults, filenames, media types, rendering behavior and existing error statuses/texts. Its OpenAPI operation SHALL declare `deprecated: true`; its operation description and API documentation SHALL recommend the preparation/download flow. The preparation and prepared-download operations SHALL NOT be deprecated. Deprecation SHALL NOT introduce a removal deadline or additional response headers. The new endpoints SHALL NOT change channel archive export/import behavior.

#### Scenario: Existing client downloads directly
- **WHEN** an existing client calls the direct-export endpoint without query parameters
- **THEN** it receives the same CSV download response rather than preparation JSON

#### Scenario: OpenAPI identifies the deprecated direct export
- **WHEN** a caller reads the generated OpenAPI schema
- **THEN** the direct-export GET operation declares `deprecated: true` and its description recommends the preparation/download flow
- **AND** the preparation and prepared-download operations are not marked deprecated
