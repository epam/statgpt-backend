# Design

## Context

See `proposal.md` for motivation and the two delta specs for the behavior contract. This is cross-cutting across persistence, response serialization, exports, imports, and CLI initialization.

Observed implementation boundaries:

- `statgpt/admin/services/channel.py` fully dumps `ChannelConfig` on create/replacement and uses an outer `exclude_unset=True` for partial scalar updates.
- `statgpt/admin/services/data_source.py` validates source-specific config and calls `DataSourceConfig.dump_for_storage()`. The proxy override excludes externally owned `proxy_config`.
- `statgpt/admin/services/dataset.py` fully dumps validated dataset config on create/replacement; dataset UUID and other row fields are assigned separately.
- `Channel.details` is typed, so ordinary channel serialization materializes defaults. Data-source and dataset response `details` are stored dictionaries.
- Config exports reuse these entity models. Import paths revalidate config and may skip unchanged items using effective-value comparisons.
- `statgpt/cli/commands/content.py` assembles tool/onboarding config before channel writes. It skips unchanged items and currently normalizes incoming dataset config differently from stored config.
- `statgpt/common/data/sdmx/v21/datasource.py` produces full resolved version configs. `statgpt/app/services/channel_datasets_metadata.py` deliberately exposes those snapshots when available.
- Related frontend/config repository branches are up to date, but source reads were blocked by session directory permissions. Their compatibility is not yet verified.

## Goals / Non-Goals

**Goals:**

- Preserve explicitness across live config persistence and editable/exported representations without sacrificing typed runtime access.
- Keep sparse persistence, effective-value comparison, and resolved snapshots as distinct serialization uses.
- Let administrators explicitly refresh legacy configs using `--update-unchanged`.

**Non-Goals:**

- Global changes to `BaseModel.model_dump`, all FastAPI response exclusions, or entire-row inserts.
- Changes to validation policy, explicit-null handling, current accepted empty-config behavior, authentication, or unknown-field normalization.
- Sparse historical snapshots, query results, discovery records, generated SDMX templates, or metadata deliberately presenting resolved configs.
- Changing row identity, deleting unlisted records, automatic reindexing, database schema changes, or related-repository implementation.

## Decisions

### 1. Sparse persistence at live config boundaries

Add recursive `exclude_unset=True` to the validated config dumps used for channel create/replacement, dataset create/replacement, and data-source `dump_for_storage()`. Retain existing aliases and JSON mode. Keep the proxy exclusion of `proxy_config` in its storage override.

Preserve required union discriminators as structural metadata, not inherited settings. In `statgpt/common/schemas/onboarding.py`, use model-local normalization to record `response_type` as supplied when constructing `PredefinedTextResponse` or `PredefinedDataQueryResponse` with its defaulted tag. Sparse serialization must retain the tag without materializing unrelated defaults. Keep existing JSON validation requirements and typed constructor convenience; do not infer missing union tags from untyped input.

Continue existing top-level partial-update handling. A supplied accepted config replaces its JSON column; omitted `details` is not an instruction to clear it. Persist generated `DataSetBase.id_`, scalar request defaults, and other row assignments as before.

Use Pydantic serialization rather than a recursive raw-dictionary default stripper. `exclude_defaults` would lose explicit default-valued overrides; `exclude_none` would lose accepted explicit nulls. Validation may normalize explicitly supplied legacy fields into their supported replacement fields; preserve existing normalization instead of treating serialization as a byte-for-byte copy of the input.

### 2. Sparse exchange without changing runtime model behavior

Keep `Channel.details` typed as `ChannelConfig`. Implement sparse serialization specifically for that field on the channel entity envelope, using a JSON-mode field serializer that honors existing alias and field-selection conventions. This makes REST responses, nested channel responses, CLI response reserialization, and channel config export agree without replacing typed config access with raw dictionaries.

Keep data-source and dataset live `details` as stored dictionaries, with existing proxy enrichment. Do not apply `exclude_unset` to complete response envelopes or a shared config base class. Such global changes would affect unrelated defaults and resolved snapshot dumps.

Do not fully dump and revalidate a sparse config before storage or exchange: that marks inferred defaults as supplied. Preserve nested field-set information through typed validation/copying. Existing persisted keys count as explicit, even when equal to a default; reads do not clean legacy configs.

Increment `JobsConfig.CURRENT_EXPORT_VERSION` from 2 to 3 and extend `SUPPORTED_EXPORT_VERSIONS` to `{1, 2, 3}`. Version 3 identifies sparse live-config archive semantics; older importers reject it through their existing version validation instead of silently materializing omitted defaults. Continue importing supported older archives with their present keys treated as explicit. Keep the existing version-1 discovery restrictions and `DISCOVERY_INDICATORS_JSON_VERSION = 2` unchanged.

### 3. Distinct effective comparisons and complete snapshots

Normal init/import change detection continues its effective-value comparison where that is the current contract. Normalize both sides through the same source-specific model; in particular, fix the incoming-expanded/stored-raw dataset comparison to avoid update loops with sparse storage. Retain source-managed title exclusions and current alias handling.

Do not let sparse `dump_for_storage()` redefine effective equality inadvertently. Keep `matches_stored` and the proxy override's internal comparison based on full normalized effective values, with external `proxy_config` compared only when incoming config supplies it. Storage and comparison methods must no longer implicitly share the same dump policy.

Keep indexing hashes, existing historical JSON, and snapshot metadata/export behavior unchanged. New resolved snapshots must remain complete, including inferred defaults. In `_apply_config_internal`, when `last_completed.resolved_config` is `None`, reuse the complete `resolved_config` already produced by `_classify_config_update` rather than copying sparse `current_config`. Leave the normal merge branch unchanged; do not rewrite existing legacy versions or perform an additional resolution request.

Hashing must continue to operate on resolved typed values rather than the presence/absence of default keys. Refreshing a live config still invokes existing dataset propagation and its normal version/status reporting; the flag is not a promise of no version or timestamp changes.

### 4. Explicit refresh flag in CLI only

Register a boolean `update-unchanged` argument and thread `update_unchanged: bool = False` through `init_handler`, `_process_client`, and the channel/data-source/dataset processing helpers. For an existing matching item, validate initialization inputs before deciding whether to compare or update; update when `update_unchanged` is true or normal change detection finds a change. Bypass only comparison when the flag is true.

Move the existing missing-`details` guard out of `_data_source_changed` into the existing-item branch of `_process_data_sources`, before the flag/comparison decision and inside the per-item `try` block. A missing `details` key must fail with or without the flag and retain batch failure isolation. Preserve creation behavior, accepted empty/null handling, and the backend's valid metadata-only update contract.

Send the original assembled config through existing AdminClient update methods. Backend sparse serialization performs cleanup; no additional backend force parameter is needed. Keep new-item creation and existing identity matching unchanged.

Apply the flag only to selected channel/data-source/dataset processing, including existing data-source dependencies when datasets are selected. Do not reinterpret it for glossaries, files, or discovery. Keep `--yes`, confirmations, validation, failure isolation, `--verify`, and reindex reporting independent.

Without the flag, effective-equivalent legacy configs can remain unchanged. This is intentional: the refresh flag is the explicit mechanism for removing previously materialized defaults that ordinary equality cannot distinguish from omitted settings.

## Risks / Trade-offs

- Missing keys in editable config responses can break consumers or cause them to save inferred defaults back -> validate frontend/config-tooling sparse round trips before rollout; coordinate any required consumer changes separately.
- Typed-model full dump/revalidation can recreate overrides -> regression-test nested field-set preservation, REST responses, CLI serialization, and export/import.
- An application default change affects live sparse configs -> this is intended; explicit default values remain pinning overrides, and historical snapshots remain unchanged.
- Refresh can update timestamps or create normal propagation versions -> document this; retain existing validation, processing restrictions, and needs-reindex reporting.
- Updating a proxy source with explicitly supplied external config still writes the config server -> preserve existing ownership semantics; omitted external config does not produce a write.
- Older configs remain mixed with new sparse configs -> do not infer original intent; refresh only selected configured items explicitly.

## Migration Plan

1. Implement and test backend sparse persistence/exchange with ordinary comparison and snapshot regressions, alongside CLI support and documentation.
2. Verify consumers accept omitted config keys and preserve sparse overrides in unchanged edit/save and export/import workflows. Related source access requires adding the read-only directories to the session; related writes remain outside this change's scope.
3. Deploy compatible backend/CLI and any separately coordinated consumer changes. No schema migration or startup cleanup is introduced.
4. Refresh selected existing configs, for example `statgpt content init --client-id sample --only channels,datasources,datasets --update-unchanged -y`. Use config files as the source of explicit overrides; include a value explicitly when it must stay pinned.
5. Check refreshed JSON, runtime loading, propagation results, and the batch summary before deciding whether a reported reindex is needed.

Rollback restores previous application/CLI serialization behavior without a database migration. Older backends reject version-3 archives; use an importer supporting version 3 rather than relabeling them as version 2. Sparse JSON remains valid input for typed runtime models, but rolling back after defaults have changed can change effective values. Restore explicitly pinned source configs and deliberately reapply them when that stability is required; do not fabricate previously omitted values from current defaults.
