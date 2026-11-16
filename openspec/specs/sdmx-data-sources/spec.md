# sdmx-data-sources Specification

## Purpose

Defines what StatGPT asks of SDMX data sources: SDMX 2.1 REST APIs, QuantHub, and the StatGPT SDMX proxy, which serves an SDMX 3.0 API in front of upstream registries. This covers the data query parameters, the headers the proxy needs to tell artefacts apart, how empty proxy results are treated, and how the admin lists a source's providers and datasets when onboarding.

## Requirements

### Requirement: SDMX 2.1 data query parameters
A data query to an SDMX 2.1 source, QuantHub included, SHALL send `detail=full` and the requested period as `startPeriod`/`endPeriod`.

#### Scenario: SDMX 2.1 query with a period
- **WHEN** a data query for 2020 to 2023 is sent to an SDMX 2.1 source
- **THEN** the request carries `detail=full`, `startPeriod=2020` and `endPeriod=2023`

### Requirement: SDMX proxy data query parameters
A data query to the SDMX proxy SHALL send `attributes=all` and the requested period as `c[<time dimension id>]=ge:<start>+le:<end>`. It SHALL NOT send `detail`, `startPeriod` or `endPeriod`. SDMX 3.0 replaced `detail` with `attributes` and `measures`, and the period parameters with component filters. Without `attributes=all` the proxy returns no attribute values.

#### Scenario: Proxy query with a period
- **WHEN** a data query for 2020 to 2023 is sent to the SDMX proxy for a dataset whose time dimension is `TIME_PERIOD`
- **THEN** the request carries `attributes=all` and `c[TIME_PERIOD]=ge:2020+le:2023`, and no `detail`

### Requirement: Proxy structure requests name their data structure
Every concept scheme and codelist request to the SDMX proxy SHALL carry the header `X-Source-Artefact-Urn` with the URN of the dataset's data structure definition, e.g. `urn:sdmx:org.sdmx.infomodel.datastructure.DataStructureDefinition=AG:DSD_X(1.0)`. The proxy needs it to tell apart artefacts from different upstream registries that share an identity, and answers 400 without it.

#### Scenario: Concept scheme in two registries
- **WHEN** a proxy dataset's concept scheme `AG:CS_NA(1.14)` exists in two upstream registries
- **THEN** the concept scheme request carries `X-Source-Artefact-Urn` with the dataset's data structure URN, and the proxy serves the right one

### Requirement: Every structure-loading path sends the header
The `X-Source-Artefact-Urn` header SHALL be sent on every path that loads a proxy dataset's structures: loading the dataset, validating its configuration, resolving its URN during auto-update or reindex, and computing its structure hash.

#### Scenario: Auto-update of a proxy dataset
- **WHEN** an auto-update job loads the structures of a proxy dataset
- **THEN** its concept scheme and codelist requests carry the header, and the job does not fail with 400

### Requirement: Dynamic dataflow annotations are requested
The dataflow request that loads a proxy dataset's structures, and QuantHub's dynamic-annotation request, SHALL send `forceDataflowDynamicAnnotations=true`. Without it these servers omit dynamic annotations such as `dataUpdatedAt`, which a dataset's last-updated date can be read from.

#### Scenario: Last-updated annotation
- **WHEN** a proxy dataset's last-updated date is configured to come from the `dataUpdatedAt` annotation
- **THEN** the dataflow request carries `forceDataflowDynamicAnnotations=true`, so the annotation is present

### Requirement: Empty proxy answers mean no data
A proxy data request answered with 400, 404, or 200 with an empty body SHALL give an empty result, not a failed request, because the proxy answers this way when the registry has no series for the query. An availability request answered with 400 or 404 SHALL give an empty availability, for the proxy and for SDMX 2.1 sources alike.

#### Scenario: No series for the query
- **WHEN** the proxy answers a data query with 200 and an empty body
- **THEN** the dataset's result has no rows and its request is not reported as failed

### Requirement: Providers of a data source
`GET /data-sources/{id}/providers` SHALL list the providers of an SDMX data source, each once and sorted by id. A provider is a maintainer agency, the first segment of a dataset URN (`IMF.RES` in `IMF.RES:WEO(9.0.0)`).

#### Scenario: Agency listed twice
- **WHEN** two agency schemes of a data source both list agency `IMF.RES`
- **THEN** the providers include `IMF.RES` once

### Requirement: Provider discovery mode
How providers are found SHALL follow the source's `providerDiscovery`. With `agencyscheme` (the default), the agency schemes of all agencies are read, and each provider gets its localized name. With `dataflows`, all dataflows are listed and each provider is named by its id. `dataflows` is meant for SDMX 2.1 registries that expose no agency schemes.

#### Scenario: Registry without agency schemes
- **WHEN** an SDMX 2.1 source with `providerDiscovery: dataflows` exposes dataflows of agencies `ECB` and `IMF`
- **THEN** the providers are `ECB` and `IMF`, each named by its id

### Requirement: The proxy discovers providers through agency schemes
A proxy data source configured with `providerDiscovery: dataflows` SHALL be refused, because the proxy cannot list dataflows across all agencies.

#### Scenario: Proxy with dataflows discovery
- **WHEN** an admin saves a proxy data source with `providerDiscovery: dataflows`
- **THEN** the configuration is refused

### Requirement: Available datasets of a data source
`GET /data-sources/{id}/available-datasets` SHALL list the dataflows of a data source, including those already added to channels. The optional `provider` query parameter SHALL restrict the list to one maintainer agency; without it, every agency is requested.

#### Scenario: Datasets of one provider
- **WHEN** an admin requests the available datasets of a proxy source with `provider=IMF.RES`
- **THEN** only the dataflows maintained by `IMF.RES` are listed

### Requirement: Available datasets carry a configuration template
Each available dataset SHALL carry a configuration template for the admin to review, guessed from its dimensions:
- a time dimension becomes the time period;
- `FREQ` or `FREQUENCY` becomes the frequency;
- `COUNTRY`, `REGION` or `REF_AREA` becomes the region;
- `INDICATOR` or `SERIES` becomes a required indicator;
- any other dimension has no type.

When no template can be built, the entry SHALL carry its URN alone.

#### Scenario: Guessed dimension types
- **WHEN** a dataflow has the dimensions `FREQ`, `REF_AREA`, `INDICATOR` and `TIME_PERIOD`
- **THEN** its template has a frequency, a region, a required indicator and a time period dimension
