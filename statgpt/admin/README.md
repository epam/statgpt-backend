# Admin

## MCP Server (Beta)

Admin includes an optional MCP (Model Context Protocol) server for coding agent integration (Cursor, Claude Code).
This feature is disabled by default and requires installing additional dependencies and enabling via environment variable.

For setup instructions, see [MCP README](mcp/README.md).

## Environment variables

Below are the environment variables specific to the Admin Application. Other required variables can be found in
the [common README file](../common/README.md).

| Variable                                    | Required                                                                | Description                                                                                                                                                                                                                            | Available Values | Default values          |
|---------------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|-------------------------|
| OIDC_AUTH_ENABLED                           | No                                                                      | If this setting is enabled, all admin endpoints require OIDC authentication. Otherwise, all endpoints can be called without any authentication, which can be useful for local development. MUST BE ENABLED IN PRODUCTION ENVIRONMENTS! | `true`, `false`  | `true`                  |
| OIDC_CONFIGURATION_ENDPOINT                 | Yes, if `$OIDC_AUTH_ENABLED`                                            | OIDC Configuration Endpoint                                                                                                                                                                                                            |                  |                         |
| OIDC_CLIENT_ID                              | Yes, if `$OIDC_AUTH_ENABLED`                                            | OIDC Client ID                                                                                                                                                                                                                         |                  |                         |
| OIDC_ISSUER                                 | Yes, if `$OIDC_AUTH_ENABLED`                                            | OIDC Issuer                                                                                                                                                                                                                            |                  |                         |
| OIDC_USERNAME_CLAIM                         | Yes, if `$OIDC_AUTH_ENABLED`                                            | OIDC Username Claim                                                                                                                                                                                                                    |                  |                         |
| OIDC_AUDIT_USER_ID_CLAIM                    | No                                                                      | JWT claim(s) used to populate audit log `performed_by` (single value, comma-separated string, or JSON array)                                                                                                                           |                  | `oid,sub`               |
| OIDC_AUDIT_PERFORMED_BY_NAME_CLAIM          | No                                                                      | JWT claim(s) used to populate audit log `performed_by_name` (single value, comma-separated string, or JSON array)                                                                                                                      |                  | `unique_name,email`     |
| ADMIN_ROLES_CLAIM                           | Yes, if `$OIDC_AUTH_ENABLED`                                            | OIDC Admin Roles Claim                                                                                                                                                                                                                 |                  |                         |
| ADMIN_ROLES_VALUES                          | Yes, if `$OIDC_AUTH_ENABLED`                                            | OIDC Admin Roles Values                                                                                                                                                                                                                |                  |                         |
| ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED        | No                                                                      | If specified, the admin portal will check for scopes in the OIDC token, otherwise this check will be skipped.                                                                                                                          | `true`, `false`  | `true`                  |
| ADMIN_SCOPE_CLAIM                           | Yes, if `$OIDC_AUTH_ENABLED` and `ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED` | The name of the custom access token field that contains scope information.                                                                                                                                                             |                  |                         |
| ADMIN_SCOPE_VALUE                           | Yes, if `$OIDC_AUTH_ENABLED` and `ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED` | Required scope claim value to get access to admin portal if scope claim validation is enabled.                                                                                                                                         |                  |                         |
| BACKGROUND_TASKS_MAX_CONCURRENT             | No                                                                      | Maximum number of background tasks that can be run concurrently                                                                                                                                                                        |                  | `5`                     |
| BACKGROUND_TASKS_TASK_TIMEOUT               | No                                                                      | Timeout in seconds for a single background task. Set to empty to disable.                                                                                                                                                              |                  | `3600` (60 minutes)     |
| EXIM_VECTOR_STORE_CONCURRENCY_LIMIT         | No                                                                      | Maximum number of concurrent export operations related to vector store                                                                                                                                                                 |                  | `10`                    |
| EXIM_ELASTIC_CONCURRENCY_LIMIT              | No                                                                      | Maximum number of concurrent export operations related to elasticsearch                                                                                                                                                                |                  | `10`                    |
| DISCOVERY_UPLOAD_MAX_FILE_SIZE_BYTES        | No                                                                      | Reject a discovery dataset file larger than this, before parsing it; applies to each file of an upload                                                                                                                                 |                  | `10485760` (10 MB)      |
| DISCOVERY_UPLOAD_MAX_FILES                  | No                                                                      | Reject a discovery dataset upload carrying more files than this. A request can be up to `MAX_FILES x MAX_FILE_SIZE_BYTES`, so the ingress body limit must allow that                                                                   |                  | `20`                    |
| DISCOVERY_UPLOAD_MAX_ROWS                   | No                                                                      | Reject a discovery dataset file with more data rows than this; applies to each file of an upload                                                                                                                                       |                  | `10000`                 |
| DISCOVERY_UPLOAD_MAX_INDICATORS_PER_DATASET | No                                                                      | Reject a discovery dataset whose indicators sheet, or CSV indicators column, lists more indicators than this                                                                                                                           |                  | `50000`                 |
| DISCOVERY_UPLOAD_MAX_REPORTED_PROBLEMS      | No                                                                      | Cap on the problems listed in a rejected upload's response; beyond it the response is marked as truncated                                                                                                                              |                  | `200`                   |
| DISCOVERY_PUBLISH_CONCURRENCY               | No                                                                      | How many discovery records an indexing run publishes at once                                                                                                                                                                           |                  | `8`                     |
| DISCOVERY_PUBLISH_SETTLE_TIMEOUT_SECONDS    | No                                                                      | How long an indexing run waits for the documents it published to finish indexing before leaving the rest to the next run; 0 disables the wait                                                                                          |                  | `300` (5 minutes)       |
| DISCOVERY_SUMMARIZE_CONCURRENCY             | No                                                                      | How many discovery records an indexing run summarizes at once                                                                                                                                                                          |                  | `4`                     |
| DISCOVERY_SUMMARIZE_TIMEOUT_SECONDS         | No                                                                      | Timeout of one indicator-summarization LLM call                                                                                                                                                                                        |                  | `180`                   |
| DISCOVERY_SUMMARIZE_VALIDATION_RETRIES      | No                                                                      | How many times a summarization call is continued with what is wrong with an invalid answer, before the record is marked FAILED                                                                                                         |                  | `2`                     |
| OTEL_APP_SERVICE_NAME                       | No                                                                      | OpenTelemetry service name for the admin application                                                                                                                                                                                   |                  | `statgpt-admin-backend` |
| BETA_MCP_ENABLED                            | No                                                                      | Enables MCP support for StatGPT(dataset config generation/test creation). Requires `mcp` dependencies to be installed.                                                                                                                 | `true`, `false`  | `false`                 |

## Discovery exports

Discovery dataset exports support two request flows. The existing `GET /channels/{channel_id}/discovery-datasets/export` continues to return a CSV/XLSX file directly. The two-step flow separates preparation errors from the file request:

1. Call `POST /channels/{channel_id}/discovery-datasets/exports` with query parameters and no request body. Preparation renders and stores the complete file synchronously; there is no background job or polling endpoint.
2. On a `201 Created` response, use its `id` to call `GET /channels/{channel_id}/discovery-datasets/exports/{id}/download`. This returns the prepared snapshot, not a regenerated export.

Both export flows accept the same parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `format` | `csv` | `csv` or `xlsx`; the download is named `discovery_datasets.csv` or `discovery_datasets.xlsx`. |
| `include_validation` | `false` | Include the stored validation verdict and, for XLSX, workbook validation highlighting. No new validation runs during export. |
| `validation_status` | Unset | Filter by `NOT_VALIDATED`, `VALID`, or `INVALID`. |
| `indexing_status` | Unset | Filter by `NEW`, `OUTDATED`, `INDEXED`, or `FAILED`. |
| `agency` | Unset | Match agency ignoring case and whitespace. Combine these filters with AND. |
| `id` | Unset | Repeat to select specific record IDs, up to 500 occurrences. Cannot be combined with the other filters. |

For example, `POST /channels/7/discovery-datasets/exports?format=xlsx&include_validation=true&validation_status=INVALID` returns a descriptor shaped as follows:

```json
{
  "id": "10f90c65-3e73-47d5-b1ac-ff51f0a90523",
  "expiresAt": "2026-11-24T13:00:00Z"
}
```

The descriptor contains only an opaque UUID and a UTC deadline, never file content, credentials, or a DIAL storage URL. Success is returned only after storage upload and metadata commit. Preparation and successful file responses use `Cache-Control: no-store`.

### Authentication and download lifetime

Preparation and retrieval use the existing admin authentication. With OIDC enabled, send `Authorization: Bearer <token>` on both requests. A download belongs to the preparing user's verified stable identity and the requested channel. Refreshed credentials for that same identity remain valid; knowing another user's UUID does not grant access. With `OIDC_AUTH_ENABLED=false`, all requests use the existing `Anonymous` identity, so this development mode does **not** provide per-user isolation.

The deadline is fixed at one hour after the complete file is uploaded. Repeated downloads do not consume the identifier or extend that deadline. Retrieval checks the server's UTC time before opening storage, and equality with `expiresAt` counts as expired. A transfer already admitted is not aborted merely because its deadline passes, but storage cleanup or a transport failure can still interrupt it.

Source record edits or deletions do not change an existing snapshot. Deleting its channel prevents further retrieval. Metadata is stored in PostgreSQL and bytes in the system user's resolved DIAL bucket/appdata namespace, so another backend replica or a restarted backend can serve the same snapshot.

### Errors and retries

| Status | Meaning |
|--------|---------|
| `400` | XLSX cannot represent the selected records, such as an invalid/conflicting sheet name or a cell exceeding 32,767 characters. Preparation preserves the direct-export error detail. Correct the records/selection or use CSV. |
| `401` / `403` | Existing admin authentication or authorization rejection. Refresh credentials or resolve the access restriction. |
| `404` | Unknown channel, selected IDs outside the channel, unknown/cleaned-up download, owner/channel mismatch, deleted channel, or missing stored file. For retrieval, prepare a new export if appropriate. |
| `410` | A matching download still has metadata but has expired. The JSON detail is `Prepared download has expired. Prepare a new export.` |
| `422` | Invalid query parameters or an invalid combination, such as `id` with `agency`. Existing validation details are preserved. |
| `502` | Preparation storage or opening the stored download is unavailable. The JSON detail is sanitized; retry after storage recovers. Failed preparation does not issue an identifier. |

Unrelated database/internal failures retain their existing error handling. Failures after file response headers have been sent interrupt the stream and cannot be replaced by a JSON error response. Opened storage resources are closed on completion, disconnect, cancellation, and send/iterator failure.

### Storage cleanup and scope

Files use the dedicated `prepared-downloads` namespace, separate from channel archive exports/imports. Existing `ADMIN_MODE=FIX_STATUSES` maintenance and the maintenance phase of `ADMIN_MODE=INIT` remove expired files before their metadata. Missing files count as already removed; failed deletions retain metadata for retry without blocking status recovery or later cleanup items. The existing daily maintenance schedule is unchanged, and the API enforces expiration even when physical deletion is delayed.

Maintenance also incrementally reclaims recognized generated UUID CSV/XLSX files that have no metadata reference and a known storage timestamp older than 24 hours. Recent, referenced, unknown-age, and unrelated files are preserved. This orphan grace period does not extend the one-hour download window. Failed metadata persistence triggers best-effort upload compensation; aged abandoned uploads remain recoverable by maintenance. Apply the admin database migration before serving the new endpoints or running updated maintenance.

This is the backend portion of the two-step flow only. Frontend adoption, notification/session-error presentation, and end-to-end UI verification are separate work; channel archive export/import and existing direct-export clients remain unchanged.

## Discovery indexing (Grade C)

A discovery indexing job publishes a channel's discovery dataset records into a Generic RAG
channel, so the channel configuration has to say which one:

```jsonc
{
  "discoveryDatasets": {
    "type": "DISCOVERY_DATASETS",
    "name": "discovery_datasets",
    "description": "Discovery datasets surfaced alongside data query results.",
    "details": {
      "applicationId": "statgpt-generic-rag-grade-b-and-c",
      "referenceAreaApplicationId": "statgpt-generic-rag-reference-areas",
      "templates": {
        "wrapper": "### Other datasets that may be relevant\n\n{items}",
        "item": "- **{name}** ({agency}) - {url}"
      }
    }
  }
}
```

`applicationId` is the DIAL application fronting the RAG channel; `$env:{VAR}` is supported.
Triggering a job on a channel without this block returns 409. Requests are authenticated with
`DIAL_API_KEY`, since a background job has no user token.

`referenceAreaApplicationId` is optional: a second RAG channel that the same job fills with the
reference-area labels the channel's records use, for the chat-time pre-filter. Leave it unset and
no vocabulary is published.

Each application's own configuration owns its document metadata schema. Both are generated
from the models that publish them, so neither has to be written by hand:

```
python scripts/print_discovery_metadata_schema.py [--patch dial/core/config/config.json]
python scripts/print_discovery_metadata_schema.py --schema reference-areas [--patch ...]
```

Changing these fields means republishing every document (`discovery reindex --force
publishing`): a document holding a string where an array is declared breaks the channel's search
and its metadata endpoint alike.

Indicator summarization is configured under `details.indexing` of the same block; every setting
is optional:

```jsonc
{
  "indexing": {
    "summaryThresholdChars": 4000,
    "chunkSizeChars": 50000,
    "maxSummaryChars": 4000,
    "llmModelConfig": { "deployment": "gpt-4.1-2025-04-14" },
    "prompts": { "chunkSummaryPrompt": null, "mergeSummaryPrompt": null }
  }
}
```

How records are validated, summarized, published and searched is specified in
[`openspec/specs`](../../openspec/specs): `discovery-dataset-records`, `discovery-indexing`,
`discovery-indicator-summarization` and `discovery-search`.
