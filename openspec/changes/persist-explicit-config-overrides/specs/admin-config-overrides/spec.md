# Spec Delta

## Purpose

Defines explicit override storage and exchange for channel, data-source, and dataset JSON configs, while keeping effective runtime configuration and historical resolved snapshots separate.

## ADDED Requirements

### Requirement: Live configs persist only explicit overrides
Creating or replacing a channel, data-source, or dataset JSON `details` config SHALL persist only explicitly supplied fields after validation and normalization, except that required union discriminators SHALL remain as structural metadata. This rule SHALL apply recursively to nested config objects, including objects inside lists and mappings.

#### Scenario: Optional channel setting is omitted
- **WHEN** a valid channel config is saved without `locale`
- **THEN** its stored `details` contains no `locale` key

#### Scenario: Nested dataset setting is omitted
- **WHEN** a valid dataset config supplies a URN object but omits its defaulted `version`
- **THEN** the stored URN object contains no `version` key

#### Scenario: Defaulted field inside a config collection is omitted
- **WHEN** an accepted config contains a list or mapping of config objects and a defaulted setting other than a required discriminator is absent from one object
- **THEN** the persisted object contains no key for that setting

### Requirement: Required union discriminators survive sparse serialization
Required discriminator tags SHALL survive sparse persistence and config exchange even when inferred from a valid typed variant. Model-local normalization SHALL record the tag as supplied without materializing unrelated defaults or relaxing existing discriminator requirements for untyped JSON input.

#### Scenario: Typed onboarding response uses its defaulted tag
- **WHEN** a valid channel config contains a typed predefined-text or predefined-data-query onboarding response constructed without explicitly passing `response_type`
- **THEN** the persisted response and config response retain the discriminator identifying that variant
- **AND** the serialized config can be validated again without adding unrelated omitted defaults

### Requirement: Explicit default values remain overrides
The system SHALL retain explicitly supplied config values even when they equal the current default. Accepted explicit nulls, false values, zero values, empty strings, empty lists, and empty objects SHALL NOT be removed merely because of their value.

#### Scenario: Explicit default is saved
- **WHEN** a valid config explicitly supplies `isOfficial: false`, matching its current default
- **THEN** `isOfficial: false` remains in the stored config and config response

#### Scenario: Nullable field explicitly clears an inherited value
- **WHEN** a valid config explicitly supplies null for a nullable setting
- **THEN** that setting remains present as null in storage and config exchange

#### Scenario: Empty nested override object is supplied
- **WHEN** a valid config explicitly supplies an empty object for a non-discriminated nested block whose fields all have defaults
- **THEN** the nested block remains present as an empty object without materialized child defaults

### Requirement: Submitted configs replace existing overrides
A supplied, accepted JSON `details` config SHALL replace the existing stored config rather than merge into it. Keys absent after validation and structural-discriminator normalization SHALL be removed recursively. Omitting the top-level `details` field SHALL leave the existing stored config unchanged; existing explicit-null handling SHALL remain unchanged.

#### Scenario: Override is removed from a replacement
- **WHEN** an existing config contains an explicit `locale` and a valid replacement omits it
- **THEN** the saved config contains no `locale`, and subsequent live config validation uses the current default

#### Scenario: Metadata-only update
- **WHEN** an update changes the item's title without supplying `details`
- **THEN** the item's JSON config remains unchanged

### Requirement: Validation and runtime defaults remain available
Existing config validation and normalization SHALL remain in effect. Consumers validating live stored configs SHALL resolve omitted fields using current application defaults without persisting or exposing those inferred defaults as explicit overrides.

#### Scenario: Sparse config is used at runtime
- **WHEN** a valid stored config omits a defaulted setting and is validated for live processing
- **THEN** processing receives the current default while the stored JSON remains unchanged

#### Scenario: Application default changes
- **WHEN** the application changes a default and validates a live config that has no explicit override for it
- **THEN** the effective live config uses the new default without a database rewrite

#### Scenario: Invalid supplied configuration
- **WHEN** a replacement violates existing config validation
- **THEN** the existing validation failure behavior applies and the invalid replacement is not persisted

### Requirement: Config responses expose stored overrides
Read and write responses exposing persisted channel, data-source, or dataset configs SHALL return the stored explicit overrides rather than materialize missing default keys. This rule SHALL apply to single-item, list, and nested entity responses without changing non-config response fields.

#### Scenario: List and single-item responses agree
- **WHEN** a stored channel config omits a defaulted block and the channel is read singly or in a list
- **THEN** neither response adds that block to `details`

#### Scenario: Dataset nested in a channel response
- **WHEN** a channel dataset response includes a live persisted dataset config
- **THEN** that config exposes its stored override keys without injecting defaults

#### Scenario: Unchanged response is saved again
- **WHEN** an explicit config response is resubmitted unchanged as a replacement
- **THEN** previously absent default keys remain absent

### Requirement: Config exports preserve explicitness
Channel config archives SHALL export explicit overrides and required structural discriminators for included live channel, data-source, and dataset configs without materializing unrelated defaults. Importing the exported config into a new item SHALL preserve absent default keys, accepted explicit defaults, and accepted explicit nulls.

#### Scenario: Export and import a sparse config
- **WHEN** a valid dataset config omits `isOfficial`, is exported, and is imported as a new dataset
- **THEN** the exported and newly stored configs both omit `isOfficial`

#### Scenario: Export an explicit default
- **WHEN** a valid dataset config explicitly stores `isOfficial: false` and is exported and imported
- **THEN** the explicit false value remains present through the round trip

### Requirement: Archives version sparse-config semantics
New archives SHALL declare export version 3. The updated backend SHALL support importing versions 1, 2, and 3 while retaining the existing restriction that version-1 archives containing discovery datasets are rejected. Unsupported archive versions SHALL be rejected before any import changes are applied. Existing keys in supported older archives SHALL be treated as explicit rather than inferred to be unset.

#### Scenario: Sparse archive declares its format version
- **WHEN** the updated backend exports a channel archive
- **THEN** its metadata declares export version 3

#### Scenario: Older importer receives a sparse archive
- **WHEN** an importer supporting only archive versions 1 and 2 receives a version-3 archive
- **THEN** version validation rejects it before importing any records

#### Scenario: Supported older archive is imported
- **WHEN** the updated backend imports a version-2 archive or a version-1 archive without discovery datasets
- **THEN** the archive remains accepted and its present config keys remain explicit, including default-valued keys

#### Scenario: Version-1 discovery restriction remains in effect
- **WHEN** the updated backend receives a version-1 archive containing discovery datasets
- **THEN** the existing format validation rejects it before importing any records

### Requirement: Legacy stored keys are not inferred to be unset
Existing stored config keys SHALL be treated as explicit until the config is replaced. Reads and exports SHALL NOT remove a stored key merely because its value equals a current default. Deploying sparse persistence SHALL NOT trigger a bulk rewrite of existing configs.

#### Scenario: Legacy materialized default is read
- **WHEN** an older stored config includes a default-valued key that was materialized before this change
- **THEN** reads and exports continue to include that key until an explicit replacement removes it

### Requirement: Sparse policy does not apply to entire database rows
Sparse serialization SHALL be limited to live configuration JSON. Generated identities, required scalar database fields, relationships, discovery record fields, audit envelope fields, and other non-config payloads SHALL retain their existing persistence and serialization contracts.

#### Scenario: Dataset identity omitted during creation
- **WHEN** a new dataset request omits its UUID
- **THEN** one generated UUID is persisted and remains stable on subsequent reads

#### Scenario: Scalar description omitted during creation
- **WHEN** a channel or data source is created without its optional request description
- **THEN** creation retains the existing scalar description behavior rather than omitting a required database column

### Requirement: Historical resolved configs remain snapshots
Historical resolved-config JSON and its existing exports SHALL retain their snapshot contract rather than apply the live-config omission policy. Newly created resolved snapshots SHALL record the complete effective config, including defaulted settings and the resolved URN. Existing historical records SHALL NOT be rewritten. APIs deliberately exposing resolved version configs or generated config templates SHALL retain their existing contracts.

#### Scenario: Export includes a resolved version
- **WHEN** an archive contains both a sparse live dataset config and a historical resolved version config
- **THEN** the live config stays sparse and the resolved snapshot retains its recorded fields

#### Scenario: Legacy version receives a sparse live-config refresh
- **WHEN** a dataset's last completed version has no resolved snapshot and propagation of a sparse live config determines that reindexing is not needed
- **THEN** the new completed pointer version records a complete resolved config, including omitted settings' effective defaults and the resolved URN
- **AND** the existing legacy version and sparse live config remain unchanged by snapshot creation

#### Scenario: Application default changes after snapshot creation
- **WHEN** an application default changes after a resolved snapshot has recorded that setting's effective value
- **THEN** interpreting that historical snapshot continues to use its recorded value rather than the new default

#### Scenario: Metadata selects a resolved version config
- **WHEN** a metadata endpoint deliberately exposes the last completed version's resolved config
- **THEN** the metadata continues to expose that snapshot rather than substitute the live override config

### Requirement: External proxy configuration retains its ownership
The `sdmx-proxy-config` ownership and enrichment contracts SHALL remain unchanged. The live external `proxyConfig` SHALL stay outside database config storage, retain its existing read/export availability semantics, and be written only when an accepted incoming config supplies it.

#### Scenario: Sparse replacement omits external configuration
- **WHEN** a proxy data source receives a valid replacement config without `proxyConfig`
- **THEN** the config server is not written, the stored JSON contains no `proxyConfig`, and responses retain existing external enrichment behavior
