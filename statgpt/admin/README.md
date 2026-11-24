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
