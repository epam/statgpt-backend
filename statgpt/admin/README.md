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

`referenceAreaApplicationId` is a second RAG channel, holding one document per distinct
reference-area label the channel's records use. Each document names the roles its records use
the label in - `subject`, `partner`, or both - so the two chat-time area axes can search the one
channel separately, each offered only the labels its own field holds. The same job publishes it,
right after the records, and the chat-time pre-filter searches it to resolve the areas a query
names onto the values the discovery channel actually holds. Leave it unset and no vocabulary is
published: the pre-filter then narrows on the remaining axes. A failure to publish it fails the job, because a
vocabulary that does not match the records narrows queries away from datasets that do answer
them.

The same block also configures the chat-time lookup over what was published. `enabled: false`
switches that lookup off while leaving indexing available, so discovery data can be indexed
before it is surfaced to users.

Note that `details.templates` is required even for a channel that only wants indexing: one block
owns both halves of the feature, so a channel with `enabled: false` still has to carry templates
nothing will render.

One RAG channel can serve several StatGPT channels and both discovery grades. Documents are
tagged with the publishing channel's deployment id, and both the indexing job and the chat-time
lookup filter on it, so a channel never publishes over, withdraws, or surfaces another
channel's records.

Each application's own configuration owns its document metadata schema. Both are generated
from the models that publish them, so neither has to be written by hand:

```
python scripts/print_discovery_metadata_schema.py [--patch dial/core/config/config.json]
python scripts/print_discovery_metadata_schema.py --schema reference-areas [--patch ...]
```

A run reads the application's schema back and refuses to publish if any field the model
declares filterable is missing. Discovery search pre-filters by `agency` and by the
`parsed_reference_areas`, `parsed_partner_reference_areas` and `parsed_frequencies` arrays -
derived from the ';'-separated cells, because a filter matches a whole value - while `grade` /
`statgpt_channel` are what let several channels and both discovery grades share one application
without overwriting each other's documents.

The vocabulary channel's `roles` is an array on the same terms, and it is what the two area
axes filter on.

The arrays have to be declared as plain, non-nullable string arrays. The RAG service turns the
schema into its own request model, and an optional or enum-typed array makes that derivation
fail on *every* search against the channel, not only one carrying the field. For the same
reason, changing these fields means republishing every document: a document holding a string
where an array is declared breaks the channel's search and its metadata endpoint alike.

### Validation

Records are validated at the start of every run, and an invalid record is withdrawn from the
index rather than published. Besides a description and a well-formed URL, a record must name at
least one reference area and at least one frequency: the chat-time pre-filter narrows by both,
so a record naming neither would be indexed and then never surfaced. Group labels such as
`Euro area` or `World` count as reference areas in their own right, and are never expanded into
the countries inside them.
