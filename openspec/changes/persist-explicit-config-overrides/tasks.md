# Tasks

## 1. Consumer compatibility

- [x] 1.1 Once read-only directory access is available, check the Admin Frontend config editor's handling of absent keys and unchanged edit/save round trips; record verified compatibility or concrete separately scoped consumer changes in the design before rollout.
- [x] 1.2 Check config/evaluation tooling's channel/data-source/dataset serialization and initialization inputs read-only; verify omitted keys and explicit defaults survive its round trips, without changing or executing the related repository's tooling.

## 2. Sparse live-config persistence

- [x] 2.1 Update channel create/replacement dumps in `statgpt/admin/services/channel.py` to preserve only supplied nested config fields; verify service tests cover omitted settings, explicit defaults/nulls, nested objects, and metadata-only updates.
- [x] 2.2 Update data-source `dump_for_storage()` and its proxy override to use sparse config serialization while retaining external-field exclusions; verify `tests/unit/common/data/statgpt_sdmx_proxy/test_config.py` and `tests/unit/admin/services/test_data_source_proxy_config.py` cover sparse defaults and unchanged external ownership.
- [x] 2.3 Keep ordinary data-source effective equality separate from sparse storage dumps, including the proxy `matches_stored` override; verify omitted-versus-explicit-default equality and incoming-only external-config comparisons.
- [x] 2.4 Update dataset create/replacement config dumps in `statgpt/admin/services/dataset.py`; verify nested URN defaults remain absent and generated dataset UUIDs, required scalar values, current null handling, and validation behavior remain unchanged.
- [x] 2.5 Extend relevant integration service tests in `tests/integration/services/test_channels.py`, `test_data_sources.py`, and `test_sdmx_datasets.py`; verify saved JSON keys, whole-config replacement, stable identities, and reads after commit against the test database.
- [x] 2.6 Document the sparse live-config and replacement contract in the admin documentation; verify examples distinguish an omitted key, an explicit default, an omitted top-level `details`, and a required structural discriminator.
- [x] 2.7 Add model-local normalization in `statgpt/common/schemas/onboarding.py` so typed `PredefinedTextResponse` and `PredefinedDataQueryResponse` construction records their defaulted `response_type` tags as supplied; verify typed channel create/replacement, sparse response/export serialization, and revalidation retain the tags without adding unrelated defaults or relaxing JSON discriminator validation.

## 3. Config reads, exports, and import round trips

- [x] 3.1 Add JSON-mode sparse serialization for typed channel `details` on the entity envelope without changing runtime attribute access or non-config fields; verify single/list/nested HTTP responses and CLI model reserialization omit only absent config keys and retain current aliases.
- [x] 3.2 Verify data-source/dataset live config responses remain stored dictionaries with sparse keys and proxy enrichment; add response tests proving unchanged GET-to-save round trips do not materialize defaults.
- [x] 3.3 Update any channel/data-source/dataset config export paths that still materialize defaults and set `JobsConfig.CURRENT_EXPORT_VERSION = 3` with `SUPPORTED_EXPORT_VERSIONS = {1, 2, 3}` in `statgpt/admin/settings/exim.py`; verify version-3 metadata, absent keys, explicit defaults/nulls, structural discriminators, existing external `proxyConfig` export behavior, and rejection before import by version validation supporting only versions 1 and 2. Keep version-1 discovery restrictions and `DISCOVERY_INDICATORS_JSON_VERSION = 2` unchanged.
- [x] 3.4 Preserve field-set information through existing config imports and maintain consistent effective comparisons; verify version-3 export/import into new items remains sparse, supported older archives preserve their present keys as explicit, version-1 discovery rejection remains unchanged, and ordinary imports do not introduce repeated updates or external-config writes when omitted.
- [x] 3.5 Update `_apply_config_internal` in `statgpt/admin/services/dataset.py` to reuse the complete `resolved_config` when `last_completed.resolved_config` is `None`, leaving the normal merge branch unchanged; add regressions proving sparse legacy refresh creates a complete new snapshot with defaulted settings and resolved URN, without an extra resolution request or rewriting existing historical records. Retain resolved config generation, snapshot export, indexing-hash, and `build_channel_datasets_metadata` regressions.
- [x] 3.6 Document the breaking config-response/export behavior, archive-version compatibility boundary, and unchanged snapshot exceptions in admin documentation; verify the documented examples match the response and round-trip tests.

## 4. Initialization comparison and refresh flag

- [x] 4.1 Normalize both incoming and stored dataset configs consistently in `_dataset_changed`, retaining source-managed title exclusions and ordinary effective comparison semantics; extend `test_content_change_detection.py` to verify sparse configs do not cause repeated unforced updates.
- [x] 4.2 Register `--update-unchanged` and thread its default-false value through `init_handler`, `_process_client`, and channel/data-source/dataset processors; verify CLI parsing/help and tests show existing matches are updated with the flag while new items use normal creation.
- [x] 4.3 Move the existing missing-`details` guard from `_data_source_changed` into the existing-item branch of `_process_data_sources`, before the flag/comparison decision and inside per-item failure handling. Bypass only comparison when the flag is true and send original assembled configs through existing AdminClient methods; verify missing `details` fails without an update request both with and without the flag, while creation behavior, accepted empty/null handling, and backend metadata-only updates remain unchanged. Verify legacy default keys are removed while explicitly pinned values and separately assembled tool/onboarding config remain present.
- [x] 4.4 Extend component-selection and batch-isolation tests to cover the flag; verify dataset filters, existing data-source dependencies, unselected/unlisted items, unaffected content components, confirmations, validation failures, summary results, and failure exits.
- [x] 4.5 Verify refresh keeps normal dataset propagation and external proxy handling; cover needs-reindex reporting without starting reindexing and no external configuration write when `proxyConfig` is absent.
- [x] 4.6 Update `statgpt/cli/README.md` and command help with scoped cleanup examples and side effects; verify `--update-unchanged` is distinguished from `--yes`, does not imply deletion/reindexing, and warns that ordinary updates can affect timestamps or propagation versions.

## 5. Integrated validation

- [x] 5.1 Run relevant unit suites covering sparse serializers, source/proxy config, propagation, imports/exports, metadata, and CLI initialization; verify all pass and existing discovery/non-config serialization regressions remain green.
- [x] 5.2 Run relevant service integration suites using the documented test database setup; verify create/read/update/export/import and selected legacy-config cleanup persist the expected sparse JSON without changing identities or historical snapshots.
- [x] 5.3 Run `make format`, `make lint`, and strict OpenSpec validation; verify success or report environment blockers explicitly, using the repository's documented equivalent tooling when necessary.
