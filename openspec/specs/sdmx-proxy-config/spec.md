# sdmx-proxy-config Specification

## Purpose

Lets an administrator see and edit the StatGPT SDMX proxy's registry and agency routing configuration through the proxy data sources of the admin API, while the proxy's config server remains the only place the configuration is stored. The configuration decides which upstream registry serves each agency, so it is what an operator changes when onboarding or repointing an agency.

## Requirements

### Requirement: The config server owns the configuration
`details.proxyConfig` of a `PROXY_SDMX30` data source SHALL be read live from the config server whenever the data source is read: listed, fetched by id, written, or exported. It SHALL never be stored in the database. The proxy, a forced reseed of the config server, and direct storage edits all change the configuration out of band, so a stored copy would drift and show the admin a stale configuration. Data sources that share a config server URL SHALL read it once per request.

#### Scenario: Configuration changed outside StatGPT
- **WHEN** the configuration is changed directly in the config server's storage and the admin then opens the data source
- **THEN** `details.proxyConfig` shows the changed configuration

#### Scenario: Nothing persisted
- **WHEN** an admin saves a proxy data source with a `proxyConfig`
- **THEN** the stored `details` of the data source have no `proxyConfig`

### Requirement: Config server address
The config server URL SHALL come from the data source's `configUrl`, which defaults to `$env:{SDMX_PROXY_CONFIG_SERVER_HOST}/statgpt/sdmx-proxy-config-server/api/v0/config`. When the URL still holds an unresolved `$env:{…}` placeholder, no request SHALL be sent and the configuration SHALL be treated as unavailable. The data source itself keeps working.

#### Scenario: Variable not set
- **WHEN** `SDMX_PROXY_CONFIG_SERVER_HOST` is not set and the data source keeps the default `configUrl`
- **THEN** no request is sent, and the data source is returned without `proxyConfig`

### Requirement: Nothing stored yet reads as null
When the config server answers 404, `proxyConfig` SHALL be `null`, meaning nothing is stored yet and the operator can create a configuration.

#### Scenario: Cold config server
- **WHEN** the config server answers 404 to the read
- **THEN** the data source is returned with `proxyConfig` = `null`

### Requirement: An unavailable config server leaves the field out
When the config server cannot be reached, does not answer within 3 seconds, or answers with an error other than 404, `proxyConfig` SHALL be absent, a warning logged, and the data source still returned. An absent field tells the Admin Portal not to offer it for editing. The timeout is short because data sources are read on every Admin Portal page load, and the page must render without the configuration.

#### Scenario: Config server down
- **WHEN** the config server answers 503 to the read
- **THEN** the data source is returned without `proxyConfig`, and the data sources list still loads

### Requirement: Writing goes through the config server
Creating or updating a proxy data source with `proxyConfig` SHALL send it to the config server, which replaces the whole configuration, before the data source is saved. The response SHALL carry the configuration the config server answered it stored, which may be normalized from the one sent. A write that omits `proxyConfig` SHALL leave the config server untouched, so a data source stays editable while its config server is down.

#### Scenario: Edit without the configuration
- **WHEN** the config server is down and an admin changes only the title of a proxy data source
- **THEN** the update succeeds and the config server receives no request

### Requirement: A failed configuration write saves nothing
A rejection by the config server (400 or 422) SHALL be reported as 422 carrying the server's message. Any other failure to write the configuration SHALL be reported as 502. In both cases the data source SHALL NOT be saved.

#### Scenario: Invalid configuration
- **WHEN** an admin saves a configuration in which an agency names a registry the configuration does not define
- **THEN** the response is 422 with the config server's message, and neither the configuration nor the data source changes

### Requirement: One configuration per config server
All proxy data sources that point at the same config server URL SHALL share one configuration. A write through any of them replaces it for all, and the last write wins.

#### Scenario: Two data sources, one server
- **WHEN** two proxy data sources use the default `configUrl` and an admin saves a configuration through the second one
- **THEN** the first one also shows the saved configuration

### Requirement: Configuration travels with exports and imports
A channel export SHALL include each proxy data source's `proxyConfig` as read at export time, and leave it out when the config server is unavailable. When an import or `statgpt content init` checks whether an existing data source changed, it SHALL compare `proxyConfig` only when the incoming details carry it. A file that omits it is therefore not reported as a change on every run.

#### Scenario: Import without the configuration
- **WHEN** an archive's proxy data source has the same details as the existing one but no `proxyConfig`
- **THEN** the data source is reported as up to date and the config server receives no write
