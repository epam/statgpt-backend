# Spec Delta

## Purpose

Defines how the CLI initializes configured content and offers an explicit, selection-scoped refresh of existing JSON configs without turning initialization into deletion or forced reindexing.

## ADDED Requirements

### Requirement: Ordinary initialization remains idempotent
Without `--update-unchanged`, `content init` SHALL retain its existing change-detection and skip-unchanged behavior. Comparing an incoming config with a sparse stored config SHALL use consistent validation, default resolution, and alias normalization so absent default keys do not cause repeated updates.

#### Scenario: Sparse config is already effective-equivalent
- **WHEN** initialization runs without `--update-unchanged` against a sparse stored config with the same effective configuration as the file
- **THEN** the existing item is reported unchanged and no update is sent

#### Scenario: Legacy explicit default matches an omitted file setting
- **WHEN** a legacy stored config explicitly contains a current default, the file omits it, and ordinary change detection considers their effective configs equal
- **THEN** initialization without `--update-unchanged` keeps its existing skip behavior rather than silently normalizing the stored config

### Requirement: Updating unchanged items is opt-in
`content init` SHALL accept a boolean `--update-unchanged` flag, disabled by default. When enabled, each selected existing configured channel, data source, and dataset SHALL receive the normal update request even when change detection considers it unchanged. Newly configured items SHALL retain normal creation behavior.

#### Scenario: Refresh an unchanged channel
- **WHEN** a selected channel is unchanged by ordinary comparison and initialization runs with `--update-unchanged`
- **THEN** its normal update request is sent and the item is reported updated

#### Scenario: Refresh an unchanged data source or dataset
- **WHEN** selected existing data sources and datasets are unchanged by ordinary comparison and initialization runs with `--update-unchanged`
- **THEN** their normal update requests are sent

#### Scenario: Create a new item with the flag
- **WHEN** a selected config has no matching existing item and initialization runs with `--update-unchanged`
- **THEN** the item is created through the normal create operation

### Requirement: Refresh removes overrides absent from source configs
With `--update-unchanged`, the normal config replacement SHALL remove stored override keys absent from the submitted, assembled config files, recursively. Explicit values present in those files SHALL remain overrides even when they equal defaults. Generated identities and existing relationships SHALL be preserved.

#### Scenario: Remove a legacy materialized default
- **WHEN** the database config contains `locale` with its current default, the assembled file config omits `locale`, and initialization runs with `--update-unchanged`
- **THEN** the replacement removes `locale` from stored JSON despite the unchanged effective value

#### Scenario: Preserve a deliberately pinned default
- **WHEN** the file explicitly supplies a default-valued setting and initialization runs with `--update-unchanged`
- **THEN** that setting remains explicitly stored

#### Scenario: Tools are assembled before refreshing the channel
- **WHEN** initialization assembles a channel from its channel, tool, and onboarding config files and runs with `--update-unchanged`
- **THEN** the replacement includes their configured fields without treating separately assembled tool or onboarding settings as absent

### Requirement: Refresh respects existing selection
`--update-unchanged` SHALL respect component, config-directory, and dataset selections and existing component dependencies. It SHALL NOT update items outside the selected configured set or delete items absent from config files. Other content components SHALL retain their existing behavior.

#### Scenario: Restrict refresh to channels
- **WHEN** initialization runs with `--only channels --update-unchanged`
- **THEN** only selected configured channels receive the flag's unconditional updates

#### Scenario: Restrict selected datasets
- **WHEN** initialization runs with an explicit dataset selection and `--update-unchanged`
- **THEN** only the selected configured datasets are refreshed, while data-source processing retains its existing dependency selection

#### Scenario: Database contains an unlisted item
- **WHEN** initialization runs with `--update-unchanged` and an existing database item has no selected config file entry
- **THEN** that item is neither updated nor deleted by the flag

### Requirement: Existing data-source initialization validates before comparison
For existing data sources, initialization SHALL validate required inputs before deciding whether to compare or update. A missing `details` key SHALL fail with or without `--update-unchanged`. Creation behavior, accepted empty/null handling, and valid backend metadata-only updates SHALL remain unchanged.

#### Scenario: Existing data-source initialization entry omits details
- **WHEN** a selected existing data source's initialization entry has no `details` key, whether or not `--update-unchanged` is enabled
- **THEN** initialization reports that item failed without sending an update request
- **AND** independent valid items continue to be processed through normal batch failure isolation

### Requirement: Refresh does not bypass normal safeguards
`--update-unchanged` SHALL NOT imply `--yes`, bypass config validation or authentication, override processing restrictions, or force reindexing. Dataset updates SHALL retain normal config-propagation results and reporting. External proxy config SHALL retain the `sdmx-proxy-config` ownership and write rules.

#### Scenario: Confirmation is still required
- **WHEN** initialization uses `--update-unchanged` without `--yes` and the selected operation normally requires confirmation
- **THEN** the normal confirmation behavior remains in effect

#### Scenario: Replacement is invalid
- **WHEN** a selected unchanged item receives an invalid replacement during initialization with `--update-unchanged`
- **THEN** normal validation rejects the update and the item is reported failed

#### Scenario: Dataset refresh changes effective indexing settings
- **WHEN** a refresh changes a dataset's effective indexing config
- **THEN** normal propagation reports whether reindexing is needed without the flag itself starting a reindex

#### Scenario: External proxy config is not supplied
- **WHEN** a selected proxy source is refreshed with `--update-unchanged` and its assembled config omits `proxyConfig`
- **THEN** the external config server receives no configuration write

### Requirement: Refresh retains batch failure isolation
Initialization with `--update-unchanged` SHALL keep existing per-item failure isolation, batch summaries, and failure exit behavior. A failed refresh SHALL NOT stop independent selected items from being processed.

#### Scenario: One of several refreshes fails
- **WHEN** a selected item fails to update while other selected items are valid
- **THEN** independent items continue, the failed item appears in the summary, and the command retains its existing failure exit behavior
