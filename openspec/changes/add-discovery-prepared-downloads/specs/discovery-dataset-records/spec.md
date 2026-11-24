## ADDED Requirements

### Requirement: Discovery export can be prepared before downloading
`POST /channels/{channel_id}/discovery-datasets/exports` SHALL synchronously prepare CSV/XLSX using the direct-export parameters, defaults, selection limits and rendering rules. Success SHALL return 201 only after the complete file and retrieval metadata are stored, with JSON containing only opaque UUID `id` and UTC timestamp `expiresAt`. The response SHALL prohibit caching and SHALL NOT return a polling job.

#### Scenario: Prepare the default export
- **WHEN** an authenticated admin prepares an existing channel's export with no query parameters
- **THEN** the response is 201 JSON containing `id` and `expiresAt`, and the identified file is the same `discovery_datasets.csv` the direct export would produce
- **AND** the response prohibits caching and contains no file content, internal storage URL or polling job

#### Scenario: Prepare a filtered validation workbook
- **WHEN** an admin prepares with `format=xlsx&include_validation=true&validation_status=INVALID`
- **THEN** the identified workbook contains exactly the invalid records and their stored validation verdicts, following the existing workbook rules

#### Scenario: Unknown channel
- **WHEN** an admin prepares an export for a channel that does not exist
- **THEN** the response is 404 and no download identifier is issued

#### Scenario: Channel deleted before metadata persistence
- **WHEN** the export file has been uploaded and another request deletes the channel before preparation persists its retrieval metadata
- **THEN** preparation answers 404, issues no identifier and attempts to delete the uploaded file, preserving the 404 if that compensation fails

#### Scenario: Metadata cannot be committed after upload
- **WHEN** a completed upload's retrieval metadata cannot be committed, including because preparation is cancelled
- **THEN** preparation issues no identifier and attempts to delete the uploaded file
- **AND** compensation failure does not mask the original failure, and unrelated database failures retain their existing error handling

### Requirement: Preparation reports export errors without issuing a download
Preparation SHALL preserve direct-export validation and rendering error statuses and texts. Failed preparation SHALL NOT issue a usable identifier or report success. Failure to store the complete file or metadata SHALL fail preparation; storage failures SHALL return 502 JSON without exposing credentials or internal paths.

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
- **AND** the error detail does not expose storage credentials or internal paths

### Requirement: A prepared download belongs to its authenticated user and channel
`GET /channels/{channel_id}/discovery-datasets/exports/{download_id}/download` SHALL use preparation's admin authentication before lookup and allow only the preparing stable identity and channel. Unknown IDs, scope mismatches, deleted channels or missing files SHALL return 404 without disclosing content, paths or expiry. Credential refresh SHALL preserve access for the same identity. With OIDC disabled, the shared anonymous identity SHALL remain, without per-user isolation.

#### Scenario: Another admin knows the identifier
- **WHEN** admin B requests the download prepared by admin A
- **THEN** the response is 404 and no file is read or returned
- **AND** lookup discloses no storage path or expiration information

#### Scenario: Identifier used for another channel
- **WHEN** an admin requests a channel 1 identifier through channel 2's download endpoint
- **THEN** the response is 404 and no file is read or returned

#### Scenario: Expired authentication
- **WHEN** a caller requests a valid identifier with an expired authentication token while OIDC is enabled
- **THEN** the request is rejected with the existing authentication error before any download lookup or file retrieval

#### Scenario: Refreshed authentication
- **WHEN** the preparing admin retrieves an unexpired download using refreshed credentials with the same user identity
- **THEN** the file is returned without preparing it again

#### Scenario: OIDC-disabled identity remains shared
- **WHEN** OIDC is disabled and callers use the existing anonymous authentication path
- **THEN** they share the existing anonymous identity and this mode is not described as providing per-user isolation

#### Scenario: Channel deleted after preparation
- **WHEN** a channel is deleted after an admin prepares its export and the admin requests that download
- **THEN** the response is 404 even if the generated file has not yet been physically removed

#### Scenario: Stored file is missing
- **WHEN** an authenticated owner requests an unexpired matching identifier but its stored file is missing
- **THEN** retrieval answers 404

### Requirement: The download window is one hour and does not slide
Prepared downloads SHALL expire one hour after successful file preparation at the fixed UTC `expiresAt` deadline. Retrieval SHALL require server UTC time strictly before that deadline. Reads SHALL neither consume IDs nor extend expiry. Matching expired metadata SHALL return 410 without file I/O; cleaned-up IDs SHALL return 404. Expiration SHALL be independent of maintenance and physical deletion.

#### Scenario: Repeated download
- **WHEN** the preparing admin downloads the same file twice before `expiresAt`
- **THEN** both requests return the file and the deadline remains unchanged

#### Scenario: Expiry boundary before cleanup
- **WHEN** the preparing admin requests a matching identifier exactly at `expiresAt` while its file and metadata still exist
- **THEN** the response is 410 with JSON detail `Prepared download has expired. Prepare a new export.` and the file is not read

#### Scenario: Identifier already cleaned up
- **WHEN** an admin requests an identifier after expired metadata has been removed
- **THEN** the response is 404

### Requirement: Retrieval returns the prepared snapshot
Successful retrieval SHALL return 200 with stored bytes, the CSV/XLSX media type and `Content-Disposition` naming `discovery_datasets.csv` or `discovery_datasets.xlsx`, without reselection, regeneration or workbook revalidation. Snapshots SHALL survive backend restarts and be available to replicas sharing the database and storage. File responses SHALL prohibit caching.

#### Scenario: Records change after preparation
- **WHEN** a record is edited after its XLSX export is prepared and before it is downloaded
- **THEN** the downloaded bytes are those prepared before the edit

#### Scenario: Different replica serves the download
- **WHEN** preparation succeeds on one replica and the matching download request reaches another replica using the same database and storage
- **THEN** the prepared file is returned without relying on the first replica's memory or local files

#### Scenario: Prepared file response
- **WHEN** an authenticated owner retrieves an unexpired matching CSV or XLSX snapshot
- **THEN** the response is 200 with the stored bytes, the format's media type and `Content-Disposition` naming `discovery_datasets.csv` or `discovery_datasets.xlsx`
- **AND** the file response prohibits caching

### Requirement: Streaming failures close prepared-download resources
A storage outage while opening a download SHALL answer 502 rather than success. Retrieval SHALL close opened resources on completion, disconnect, cancellation, and response-send or storage-iterator failure, including before the first body chunk is sent. Failures after transmission starts SHALL NOT be replaced by a JSON response.

#### Scenario: Storage unavailable while opening
- **WHEN** storage cannot open an authenticated owner's unexpired matching download
- **THEN** retrieval answers 502 rather than returning a successful file response

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
The existing `GET /channels/{channel_id}/discovery-datasets/export` SHALL remain available with unchanged direct-file responses, query parameters, defaults, filenames, media types, rendering and error statuses/texts. Channel archive export/import behavior SHALL remain unchanged.

#### Scenario: Existing client downloads directly
- **WHEN** an existing client calls the direct-export endpoint without query parameters
- **THEN** it receives the same CSV download response rather than preparation JSON

### Requirement: Direct-export deprecation preserves runtime compatibility
The direct-export OpenAPI operation SHALL declare `deprecated: true`, with its operation description and API documentation recommending the preparation/download flow. The preparation and prepared-download operations SHALL remain non-deprecated. Deprecation SHALL NOT add a removal deadline or response headers.

#### Scenario: OpenAPI identifies the deprecated direct export
- **WHEN** a caller reads the generated OpenAPI schema
- **THEN** the direct-export GET operation declares `deprecated: true` and its description recommends the preparation/download flow
- **AND** the preparation and prepared-download operations are not marked deprecated

#### Scenario: Deprecation adds no response headers or removal deadline
- **WHEN** an existing client uses the deprecated direct-export endpoint
- **THEN** deprecation adds no response headers and no removal deadline is introduced
