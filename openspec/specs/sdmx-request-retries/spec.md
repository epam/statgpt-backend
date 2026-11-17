# sdmx-request-retries Specification

## Purpose

Defines how requests to SDMX data sources (SDMX 2.1 REST APIs, QuantHub and the StatGPT SDMX proxy) survive transient failures, and how a request that keeps timing out is reported to administrators and chat users.

## Requirements

### Requirement: Transient failures are retried
Every request sent through the SDMX client SHALL be tried up to 3 times, 3 seconds apart, when it times out (connect, read, write or pool), gets a 5xx response, or fails with a network error (connect, read, write or close) or a remote protocol error. Gateways in front of data sources cause such errors when they recycle or scale. Retrying POST availability queries SHALL be allowed, because SDMX requests only read data.

#### Scenario: Connection dropped once
- **WHEN** the first attempt of an availability query fails because the gateway aborted the TLS handshake and the second attempt succeeds
- **THEN** the query returns the second attempt's result

#### Scenario: Server error on every attempt
- **WHEN** the data source answers 503 to all 3 attempts
- **THEN** the request fails with the 503 error after the third attempt

### Requirement: QuantHub side requests are not retried
QuantHub's attribute and dynamic-annotation requests are sent outside the SDMX client and SHALL be tried once.

#### Scenario: Annotation endpoint down
- **WHEN** QuantHub's dynamic-annotation endpoint answers 503
- **THEN** that request fails after one attempt

### Requirement: Permanent failures are not retried
A request SHALL fail at the first attempt when it gets a 4xx response or fails with an unsupported protocol, a proxy error or a local protocol error, because repeating it cannot succeed.

#### Scenario: Missing dataflow
- **WHEN** the data source answers 404 to a dataflow request
- **THEN** the request fails after one attempt

### Requirement: Exhausted timeouts raise a timeout error
When the last attempt times out, the request SHALL fail with a dedicated timeout error instead of the raw HTTP client exception. Its message SHALL name the method and URL, the number of attempts and the total time waited, and SHALL be safe to show to administrators and users. Any other error on the last attempt SHALL be raised unchanged.

#### Scenario: Slow proxy
- **WHEN** all 3 attempts to load a dataflow from the SDMX proxy time out
- **THEN** the request fails with a timeout error such as "SDMX request to GET https://proxy/…/dataflow/AG/DF/latest timed out after 3 attempt(s), waited 278.4s total"

### Requirement: Admin status names a timeout
When loading a dataset's structures fails with the timeout error and the caller accepts an offline dataset, the dataset SHALL be offline with the details "The SDMX server timed out while loading the dataflow structures. urn=…". Any other loading failure keeps the generic "Failed to load the dataflow or its associated structures. urn=…". This SHALL hold for SDMX 2.1, QuantHub and SDMX proxy data sources.

#### Scenario: Dataset status in the Admin Portal
- **WHEN** the admin loads a channel's datasets and the structure requests of dataset `AG:DF(1.0)` time out
- **THEN** the dataset is listed as offline with details saying the SDMX server timed out while loading the dataflow structures

### Requirement: Chat reports a timed-out query as a failed request
A data query whose request fails with the timeout error SHALL produce a failed request that carries the timeout message as its reason. The response then reports the failure with advice to retry (see `data-query-agent-response`) and never says there is no data for the period.

#### Scenario: Data query times out
- **WHEN** the data request for dataset `AG:DF(1.0)` times out on all attempts
- **THEN** the dataset's result is a failed request with the timeout message as its reason, not an empty result

### Requirement: Retries and failures are logged with their cause
Each retry SHALL be logged with the attempt number out of the maximum, and each final failure with the number of attempts made. Both SHALL show the time elapsed since the first attempt and the full chain of causes (e.g. `httpx.ConnectError <- httpcore.ConnectError <- anyio.EndOfStream`). Transport errors often have an empty message, and the real reason is visible only in the underlying cause. The final timeout log SHALL also show the client's timeout settings.

#### Scenario: Empty error message
- **WHEN** an attempt fails with a connect error whose message is empty
- **THEN** the log line shows the attempt number, the elapsed time and the chain down to the underlying cause
