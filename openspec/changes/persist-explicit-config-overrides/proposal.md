# Proposal

## Why

Channel, data-source, and dataset JSON configs currently materialize Pydantic defaults on writes and some reads/exports, turning omitted settings into fixed overrides. Store and expose only explicit overrides so omitted settings inherit current application defaults, and provide a controlled way to refresh existing configs from their source files.

## What Changes

- Persist validated channel, data-source, and dataset `details` using recursive `exclude_unset=True`; retain explicit values even when they equal a default, including accepted explicit nulls.
- Keep submitted configs as whole-object replacements, not nested patches; an omitted top-level `details` continues to leave the stored config unchanged.
- **BREAKING**: Config reads, write responses, and config exports expose explicit stored overrides rather than filling absent config fields with defaults. Runtime config models still resolve defaults before use.
- Preserve database-owned/generated identities, required scalar values, historical resolved configs, discovery records, and non-config payload serialization.
- Preserve the existing external ownership, enrichment, and write rules for proxy `proxyConfig`.
- Add `statgpt content init --update-unchanged` to update selected existing channels, data sources, and datasets even when ordinary change detection would skip them. Replacing their configs removes stored overrides absent from the assembled config files.
- Keep ordinary initialization idempotent with sparse configs; the new flag does not bypass validation, skip prompts, delete absent records, or force reindexing.
- Do not bulk rewrite existing database configs; administrators can explicitly refresh selected configured items with the new flag.

## Capabilities

### New Capabilities

- `admin-config-overrides`: Explicit JSON configuration persistence and read/export contracts for channels, data sources, and datasets, distinct from effective runtime models and historical snapshots.
- `content-initialization`: Config-driven initialization, sparse-config change detection, and opt-in updating of unchanged configured items through the CLI.

### Modified Capabilities

None. Existing `sdmx-proxy-config` ownership and external-config behavior, SDMX config-template generation, and admin background-job contracts remain unchanged.

## Impact

- Backend: admin channel/data-source/dataset write and import/export services, shared channel response serialization, data-source storage serialization, and config comparison paths.
- CLI: `content init` argument registration and handler/helper plumbing, change detection, help/documentation, and batch tests.
- API consumers: the Admin Frontend and config/evaluation tooling must tolerate omitted config keys without writing inferred defaults back during unchanged round trips.
- Related-repository branches were checked and fast-forward pulls completed; source inspection is blocked by this session's allowed-directory settings. Compatibility checks remain an explicit implementation prerequisite; no related-repository edits are authorized here.
- Tests: service persistence/read/export/import coverage, nested-field serialization, runtime and resolved-config regressions, proxy ownership, and CLI selection/idempotence/failure isolation.
- No database schema migration, new environment variable, new dependency, backend force parameter, or automatic reindex operation is required.
