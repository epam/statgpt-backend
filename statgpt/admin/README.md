# Admin

## MCP Server (Beta)

Admin includes an optional MCP (Model Context Protocol) server for coding agent integration (Cursor, Claude Code).
This feature is disabled by default and requires installing additional dependencies and enabling via environment variable.

For setup instructions, see [MCP README](mcp/README.md).

## Environment variables

Below are the environment variables specific to the Admin Application. Other required variables can be found in
the [common README file](../common/README.md).

| Variable                                 | Required                                                                | Description                                                                                                                                                                                                                            | Available Values | Default values          |
|------------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|-------------------------|
| OIDC_AUTH_ENABLED                        | No                                                                      | If this setting is enabled, all admin endpoints require OIDC authentication. Otherwise, all endpoints can be called without any authentication, which can be useful for local development. MUST BE ENABLED IN PRODUCTION ENVIRONMENTS! | `true`, `false`  | `true`                  |
| OIDC_CONFIGURATION_ENDPOINT              | Yes, if `$OIDC_AUTH_ENABLED`                                            | OIDC Configuration Endpoint                                                                                                                                                                                                            |                  |                         |
| OIDC_CLIENT_ID                           | Yes, if `$OIDC_AUTH_ENABLED`                                            | OIDC Client ID                                                                                                                                                                                                                         |                  |                         |
| OIDC_ISSUER                              | Yes, if `$OIDC_AUTH_ENABLED`                                            | OIDC Issuer                                                                                                                                                                                                                            |                  |                         |
| OIDC_USERNAME_CLAIM                      | Yes, if `$OIDC_AUTH_ENABLED`                                            | OIDC Username Claim                                                                                                                                                                                                                    |                  |                         |
| OIDC_AUDIT_USER_ID_CLAIM                 | No                                                                      | JWT claim(s) used to populate audit log `performed_by` (single value, comma-separated string, or JSON array)                                                                                                                           |                  | `oid,sub`               |
| OIDC_AUDIT_PERFORMED_BY_NAME_CLAIM       | No                                                                      | JWT claim(s) used to populate audit log `performed_by_name` (single value, comma-separated string, or JSON array)                                                                                                                      |                  | `unique_name,email`     |
| ADMIN_ROLES_CLAIM                        | Yes, if `$OIDC_AUTH_ENABLED`                                            | OIDC Admin Roles Claim                                                                                                                                                                                                                 |                  |                         |
| ADMIN_ROLES_VALUES                       | Yes, if `$OIDC_AUTH_ENABLED`                                            | OIDC Admin Roles Values                                                                                                                                                                                                                |                  |                         |
| ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED     | No                                                                      | If specified, the admin portal will check for scopes in the OIDC token, otherwise this check will be skipped.                                                                                                                          | `true`, `false`  | `true`                  |
| ADMIN_SCOPE_CLAIM                        | Yes, if `$OIDC_AUTH_ENABLED` and `ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED` | The name of the custom access token field that contains scope information.                                                                                                                                                             |                  |                         |
| ADMIN_SCOPE_VALUE                        | Yes, if `$OIDC_AUTH_ENABLED` and `ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED` | Required scope claim value to get access to admin portal if scope claim validation is enabled.                                                                                                                                         |                  |                         |
| BACKGROUND_TASKS_MAX_CONCURRENT          | No                                                                      | Maximum number of background tasks that can be run concurrently                                                                                                                                                                        |                  | `5`                     |
| BACKGROUND_TASKS_TASK_TIMEOUT            | No                                                                      | Timeout in seconds for a single background task. Set to empty to disable.                                                                                                                                                              |                  | `3600` (60 minutes)     |
| EXIM_VECTOR_STORE_CONCURRENCY_LIMIT      | No                                                                      | Maximum number of concurrent export operations related to vector store                                                                                                                                                                 |                  | `10`                    |
| EXIM_ELASTIC_CONCURRENCY_LIMIT           | No                                                                      | Maximum number of concurrent export operations related to elasticsearch                                                                                                                                                                |                  | `10`                    |
| DISCOVERY_UPLOAD_MAX_FILE_SIZE_BYTES     | No                                                                      | Reject a discovery dataset upload larger than this, before parsing it                                                                                                                                                                  |                  | `10485760` (10 MB)      |
| DISCOVERY_UPLOAD_MAX_ROWS                | No                                                                      | Reject a discovery dataset file with more data rows than this                                                                                                                                                                          |                  | `10000`                 |
| DISCOVERY_UPLOAD_MAX_REPORTED_PROBLEMS   | No                                                                      | Cap on the problems listed in a rejected upload's response; beyond it the response is marked as truncated                                                                                                                              |                  | `200`                   |
| DISCOVERY_PUBLISH_CONCURRENCY            | No                                                                      | How many discovery records an indexing run publishes at once                                                                                                                                                                           |                  | `8`                     |
| DISCOVERY_PUBLISH_SETTLE_TIMEOUT_SECONDS | No                                                                      | How long an indexing run waits for the documents it published to finish indexing before leaving the rest to the next run; 0 disables the wait                                                                                          |                  | `300` (5 minutes)       |
| OTEL_APP_SERVICE_NAME                    | No                                                                      | OpenTelemetry service name for the admin application                                                                                                                                                                                   |                  | `statgpt-admin-backend` |
| BETA_MCP_ENABLED                         | No                                                                      | Enables MCP support for StatGPT(dataset config generation/test creation). Requires `mcp` dependencies to be installed.                                                                                                                 | `true`, `false`  | `false`                 |

## Discovery indexing (Grade C)

A discovery indexing job publishes a channel's discovery dataset records into a Generic RAG
channel, so the channel configuration has to say which one:

```jsonc
{
  "discoveryRag": { "applicationId": "statgpt-generic-rag-grade-b-and-c" }
}
```

`applicationId` is the DIAL application fronting the RAG channel; `$env:{VAR}` is supported.
Triggering a job on a channel without this block returns 409. Requests are authenticated with
`DIAL_API_KEY`, since a background job has no user token.

The application's own configuration owns the document metadata schema. It is generated from
`DiscoveryDocumentMetadata`, so it never has to be written by hand:

```
python scripts/print_discovery_metadata_schema.py [--patch dial/core/config/config.json]
```

A run reads the application's schema back and refuses to publish if any field the model
declares filterable is missing - discovery search pre-filters by `agency` and
`reference_area`, and `grade` / `statgpt_channel` are what let several channels and both
discovery grades share one application without overwriting each other's documents.
