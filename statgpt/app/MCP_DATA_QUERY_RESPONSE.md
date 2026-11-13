# Data Query MCP response

Field reference and examples for version 3 of the Data Query tool's MCP response. The rules
behind it (which surface carries what, when a field is present, the config toggles and their
defaults, how `queryId` and the resource URIs are built) are specified in
[`data-query-mcp-response`](../../openspec/specs/data-query-mcp-response/spec.md).
See [the migration guide](MCP_DATA_QUERY_RESPONSE_MIGRATION.md) for what changed from version 2.

A result has three surfaces: `content` (the `structuredContent` as JSON text, the `text/csv` and
`text/markdown` resources, then the discovery datasets block), `structuredContent` (for the
calling model) and `result._meta` (for the MCP-App widget and programmatic clients).

## `structuredContent`

Null fields are omitted.

| Field | Notes |
|---|---|
| `status` | The pipeline outcome; see [Examples](#examples) for each one. |
| `message` | What the model should know about the outcome. |
| `executedAt` | When the queries were executed (ISO 8601, UTC). |
| `queries[].queryId` | Short id of the query within the response. Joins it to its resources and to the `_meta` payloads. |
| `queries[].datasetUrn` | URN of the queried dataset. |
| `queries[].datasetName` | Dataset name, when known. |
| `queries[].isOfficial` | Whether the dataset is official. |
| `queries[].provider` / `datasetLastUpdated` / `datasetUrl` | The dataset's provider, last-updated date (ISO 8601) and link. |
| `queries[].querySummary` | A short summary of what the query asks for. |
| `queries[].executed` | `false` for a query that was constructed but never ran. |
| `queries[].filters[]` | One entry per filtered dimension: `dimensionId`, `dimensionName`, `operator`, `values[].id` / `values[].name`. A dimension with no filter is not listed. |
| `queries[].filters[].isIndicator` | Whether the dimension is one of the dataset's indicators. |
| `queries[].filters[].isDefault` | Whether the filter is the dimension's default, applied because the user did not specify it. |
| `queries[].filters[].valueCount` | How many values the filter applies. |
| `queries[].requestedPeriod` | `startPeriod` / `endPeriod`, named after the SDMX REST query parameters, and `isDefault` when the dataset's default period was applied. |
| `queries[].invalidity` | Why a constructed query cannot run: `reason` (`invalid_time_period` or `missing_dimensions`), an `explanation` in words, and the `rejectedPeriod` (`rejectedBound`, its `requestedValue`, the `availablePeriod`) or the `missingDimensions` (as in the top-level `missingDimensions`). |
| `queries[].factualPeriod` | The period the returned data actually covers. |
| `queries[].seriesCount` | Number of series returned. |
| `queries[].execution` | How the execution went: `result` (`data_received`, `partially_parsed`, `parsing_failed`, `request_failed`, `no_data`), with a `reason` and an `advice` unless the data was received. |
| `queries[].dataExplorerUrl` | Deep link to the query's data. |
| `missingDimensions` | `datasetUrn` and the `dimensions` a follow-up query must specify: `dimensionId`, `name`, `totalValues` and `sampleValues`. |
| `candidateDatasets[]` | Datasets to narrow the query to: `id`, `name`, `isOfficial`, and the `query` that would run against it. |

## `result._meta`

Each payload sits under a namespaced key, `statgpt.dialx.ai/...` by default, and carries
`version` (`3`).

### `{namespace}/mcp-app`

What the UI widget renders and edits. Null fields are kept.

| Field | Notes |
|---|---|
| `status` | The pipeline outcome. |
| `message` | The pipeline's full response text, as the agent would read it. |
| `queries[]` | The SDMX query model: `urn`, `filters[]` (`componentCode`, `operator`, `values`, the time period included), `metadata`, `sdmx1Source`, `disabled`, and its `queryId`. |
| `candidateDatasets[]` | `id`, `name`, `description`, `isOfficial`. |
| `missingDimensions` | The missing dimensions with every available value. |
| `pythonCode` | A reproducible sdmx1 snippet for the queries. |
| `tools.sdmxProxy` | Name of the channel's SDMX proxy tool. |

### `{namespace}/client`

What a programmatic client needs to present and navigate the result. Null fields are omitted.

| Field | Notes |
|---|---|
| `status` | The pipeline outcome. |
| `queries[].queryId` / `urn` / `datasetName` | The query and its dataset. |
| `queries[].dataExplorerUrl` / `datasetUrl` | Deep link to the query's data, and the dataset link. |
| `queries[].resourceUris` | URIs of the resources `content` carries for the query. |
| `queries[].seriesCount` | Number of series returned. |

## Resource values

The `text/csv` and `text/markdown` resources carry the values the LLM reads, transformed as the
dataset config sets. The Supreme Agent reads the same values. The attachments the user gets in a
chat (the UI grid, the charts, the table and the downloaded CSV) keep the values the source stores,
at full precision, as do the SDMX query, the data explorer link and the Python code, which reproduce
the request to the source.

A dataset whose series store a percentage as a fraction of one (`0.028` for 2.8%) can have them
converted to percentages with `fractionToPercent` in its config. `component` names the dimension or
attribute that tells the series apart, and `codes` lists its codes for the fraction series. A series
dimension is the safe choice: an attribute such as a type often groups fractions with other values
(a ratio type can also hold basis points or a leverage ratio). An attribute component must be listed
in `includeAttributes`. `unitAttribute` names the attribute set to `unit` (`%` by default) on the
converted series, so no reader takes `2.8` for a fraction or converts it again. It must be listed in
`includeAttributes`, and is added to the table when the source leaves it out, empty for the other
series. A response that lacks `component` keeps its fractions and logs a warning.

`decimalPlaces` (0 to 15) rounds the values to that many decimal places, after the conversion, so
the precision applies to the percentage. A value too small for that precision keeps its first
significant digit instead of reading as `0`: with 3 decimal places `0.0004` stays `0.0004` and
`0.00049` becomes `0.0005`. Unset, the values are not rounded.

```yaml
includeAttributes: ["UNIT"]
fractionToPercent:
  component: "SERIES"
  codes: ["GDP_GROWTH", "INFLATION"]
  unitAttribute: "UNIT"
decimalPlaces: 3
```

## Examples

One per pipeline status. The `content` blocks are omitted for brevity; `resourceUris` lists the
resources `content` carries. `…` marks text shortened here. The examples show a channel with a
widget, the client payload enabled and `dataQueryExecutedMcpOnly` configured.

### `data_available`

The queries ran and returned data.

```json
{
  "structuredContent": {
    "status": "data_available",
    "message": "Do not reproduce the returned table: the user already sees the data in the widget.",
    "executedAt": "2026-09-23T10:00:00.000000+00:00",
    "queries": [
      {
        "queryId": "dq_333e6e65fc",
        "datasetUrn": "IMF.RES:WEO(9.0.0)",
        "datasetName": "World Economic Outlook (WEO)",
        "provider": "IMF Research Department (RES)",
        "datasetLastUpdated": "2026-04-14",
        "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
        "querySummary": "For the United States, Gross Domestic Product (GDP) in current prices, in domestic currency and US dollars, was retrieved from 2021 to 2026 from the World Economic Outlook.",
        "executed": true,
        "filters": [
          {
            "dimensionId": "COUNTRY",
            "dimensionName": "Country",
            "operator": "in",
            "values": [
              {
                "id": "USA",
                "name": "United States"
              }
            ],
            "isIndicator": false,
            "isDefault": false,
            "valueCount": 1
          },
          {
            "dimensionId": "INDICATOR",
            "dimensionName": "Indicator",
            "operator": "in",
            "values": [
              {
                "id": "NGDP",
                "name": "Gross domestic product (GDP), Current prices, Domestic currency"
              },
              {
                "id": "NGDPD",
                "name": "Gross domestic product (GDP), Current prices, US dollar"
              }
            ],
            "isIndicator": true,
            "isDefault": false,
            "valueCount": 2
          }
        ],
        "requestedPeriod": {
          "startPeriod": "2021-01-01",
          "endPeriod": "2026-12-31",
          "isDefault": false
        },
        "factualPeriod": {
          "startPeriod": "2021",
          "endPeriod": "2025"
        },
        "seriesCount": 2,
        "execution": {
          "result": "data_received"
        },
        "dataExplorerUrl": "https://data.imf.org/en/Data-Explorer?datasetUrn=IMF.RES:WEO(9.0.0)&timeseriesName=USA.NGDP+NGDPD.*&startPeriod=2021-01-01&endPeriod=2026-12-31"
      }
    ],
    "candidateDatasets": []
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "data_available",
      "message": "The following queries were executed:\n\n…\n\nDo not reproduce the returned table: the user already sees the data in the widget.\n[Data Query executed at 2026-09-23T10:00:00.000000+00:00]",
      "queries": [
        {
          "urn": "IMF.RES:WEO(9.0.0)",
          "filters": [
            {
              "componentCode": "COUNTRY",
              "operator": "in",
              "values": [
                "USA"
              ]
            },
            {
              "componentCode": "INDICATOR",
              "operator": "in",
              "values": [
                "NGDP",
                "NGDPD"
              ]
            },
            {
              "componentCode": "TIME_PERIOD",
              "operator": "between",
              "values": [
                "2021-01-01",
                "2026-12-31"
              ]
            }
          ],
          "metadata": {
            "countryDimension": "COUNTRY",
            "indicatorDimensions": [
              "INDICATOR"
            ],
            "timePeriodDimension": "TIME_PERIOD",
            "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
            "keyDimensionIdsInDsdOrder": [
              "COUNTRY",
              "INDICATOR",
              "FREQUENCY"
            ]
          },
          "sdmx1Source": "IMF_DATA",
          "disabled": false,
          "queryId": "dq_333e6e65fc"
        }
      ],
      "candidateDatasets": [],
      "missingDimensions": null,
      "pythonCode": "# Uses the [sdmx1 library](https://pypi.org/project/sdmx1/)\n# Install with:\n# ```bash\n# pip install sdmx1\n# ```\n\nimport sdmx\n\nprovider = sdmx.Client(\"IMF_DATA\")\ndata_msg = provider.data(\n    \"IMF.RES,WEO,9.0.0\",\n    key=\"USA.NGDP+NGDPD.\",\n    params={'detail': 'full', 'startPeriod': '2021-01-01', 'endPeriod': '2026-12-31'}\n)",
      "tools": {
        "sdmxProxy": "sdmx_proxy"
      },
      "version": 3
    },
    "statgpt.dialx.ai/client": {
      "status": "data_available",
      "queries": [
        {
          "queryId": "dq_333e6e65fc",
          "urn": "IMF.RES:WEO(9.0.0)",
          "datasetName": "World Economic Outlook (WEO)",
          "dataExplorerUrl": "https://data.imf.org/en/Data-Explorer?datasetUrn=IMF.RES:WEO(9.0.0)&timeseriesName=USA.NGDP+NGDPD.*&startPeriod=2021-01-01&endPeriod=2026-12-31",
          "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
          "resourceUris": [
            "statgpt://data_query/IMF.RES%3AWEO%289.0.0%29/dq_333e6e65fc.csv",
            "statgpt://data_query/IMF.RES%3AWEO%289.0.0%29/dq_333e6e65fc.md"
          ],
          "seriesCount": 2
        }
      ],
      "version": 3
    }
  },
  "resourceUris": [
    "statgpt://data_query/IMF.RES%3AWEO%289.0.0%29/dq_333e6e65fc.csv",
    "statgpt://data_query/IMF.RES%3AWEO%289.0.0%29/dq_333e6e65fc.md"
  ]
}
```

### `executed_no_data`

The queries ran and returned nothing. The model still sees what was asked, and `seriesCount` is absent.

```json
{
  "structuredContent": {
    "status": "executed_no_data",
    "message": "Do not reproduce the returned table: the user already sees the data in the widget.",
    "executedAt": "2026-09-23T10:00:00.000000+00:00",
    "queries": [
      {
        "queryId": "dq_333e6e65fc",
        "datasetUrn": "IMF.RES:WEO(9.0.0)",
        "datasetName": "World Economic Outlook (WEO)",
        "provider": "IMF Research Department (RES)",
        "datasetLastUpdated": "2026-04-14",
        "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
        "querySummary": "For the United States, Gross Domestic Product (GDP) in current prices, in domestic currency and US dollars, was retrieved from 2021 to 2026 from the World Economic Outlook.",
        "executed": true,
        "filters": [
          {
            "dimensionId": "COUNTRY",
            "dimensionName": "Country",
            "operator": "in",
            "values": [
              {
                "id": "USA",
                "name": "United States"
              }
            ],
            "isIndicator": false,
            "isDefault": false,
            "valueCount": 1
          },
          {
            "dimensionId": "INDICATOR",
            "dimensionName": "Indicator",
            "operator": "in",
            "values": [
              {
                "id": "NGDP",
                "name": "Gross domestic product (GDP), Current prices, Domestic currency"
              },
              {
                "id": "NGDPD",
                "name": "Gross domestic product (GDP), Current prices, US dollar"
              }
            ],
            "isIndicator": true,
            "isDefault": false,
            "valueCount": 2
          }
        ],
        "requestedPeriod": {
          "startPeriod": "2021-01-01",
          "endPeriod": "2026-12-31",
          "isDefault": false
        },
        "execution": {
          "result": "no_data",
          "reason": "A response was received, but it does not contain any data.",
          "advice": "Most likely, the query is generally correct, but there is no data for the specified time period. You may want to try selecting a different time period. Another option is to try to find relevant data in other datasets or using other tools."
        },
        "dataExplorerUrl": "https://data.imf.org/en/Data-Explorer?datasetUrn=IMF.RES:WEO(9.0.0)&timeseriesName=USA.NGDP+NGDPD.*&startPeriod=2021-01-01&endPeriod=2026-12-31"
      }
    ],
    "candidateDatasets": []
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "executed_no_data",
      "message": "The following queries were executed:\n\n…\n\nDo not reproduce the returned table: the user already sees the data in the widget.\n[Data Query executed at 2026-09-23T10:00:00.000000+00:00]",
      "queries": [
        {
          "urn": "IMF.RES:WEO(9.0.0)",
          "filters": [
            {
              "componentCode": "COUNTRY",
              "operator": "in",
              "values": [
                "USA"
              ]
            },
            {
              "componentCode": "INDICATOR",
              "operator": "in",
              "values": [
                "NGDP",
                "NGDPD"
              ]
            },
            {
              "componentCode": "TIME_PERIOD",
              "operator": "between",
              "values": [
                "2021-01-01",
                "2026-12-31"
              ]
            }
          ],
          "metadata": {
            "countryDimension": "COUNTRY",
            "indicatorDimensions": [
              "INDICATOR"
            ],
            "timePeriodDimension": "TIME_PERIOD",
            "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
            "keyDimensionIdsInDsdOrder": [
              "COUNTRY",
              "INDICATOR",
              "FREQUENCY"
            ]
          },
          "sdmx1Source": "IMF_DATA",
          "disabled": false,
          "queryId": "dq_333e6e65fc"
        }
      ],
      "candidateDatasets": [],
      "missingDimensions": null,
      "pythonCode": "# Uses the [sdmx1 library](https://pypi.org/project/sdmx1/)\n# Install with:\n# ```bash\n# pip install sdmx1\n# ```\n\nimport sdmx\n\nprovider = sdmx.Client(\"IMF_DATA\")\ndata_msg = provider.data(\n    \"IMF.RES,WEO,9.0.0\",\n    key=\"USA.NGDP+NGDPD.\",\n    params={'detail': 'full', 'startPeriod': '2021-01-01', 'endPeriod': '2026-12-31'}\n)",
      "tools": {
        "sdmxProxy": "sdmx_proxy"
      },
      "version": 3
    },
    "statgpt.dialx.ai/client": {
      "status": "executed_no_data",
      "queries": [
        {
          "queryId": "dq_333e6e65fc",
          "urn": "IMF.RES:WEO(9.0.0)",
          "datasetName": "World Economic Outlook (WEO)",
          "dataExplorerUrl": "https://data.imf.org/en/Data-Explorer?datasetUrn=IMF.RES:WEO(9.0.0)&timeseriesName=USA.NGDP+NGDPD.*&startPeriod=2021-01-01&endPeriod=2026-12-31",
          "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
          "resourceUris": []
        }
      ],
      "version": 3
    }
  }
}
```

### `failed`

The fetch or the parsing failed. Also the default status, in which case there are no queries to report at all.

```json
{
  "structuredContent": {
    "status": "failed",
    "message": "Do not reproduce the returned table: the user already sees the data in the widget.",
    "executedAt": "2026-09-23T10:00:00.000000+00:00",
    "queries": [
      {
        "queryId": "dq_333e6e65fc",
        "datasetUrn": "IMF.RES:WEO(9.0.0)",
        "datasetName": "World Economic Outlook (WEO)",
        "provider": "IMF Research Department (RES)",
        "datasetLastUpdated": "2026-04-14",
        "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
        "querySummary": "For the United States, Gross Domestic Product (GDP) in current prices, in domestic currency and US dollars, was retrieved from 2021 to 2026 from the World Economic Outlook.",
        "executed": true,
        "filters": [
          {
            "dimensionId": "COUNTRY",
            "dimensionName": "Country",
            "operator": "in",
            "values": [
              {
                "id": "USA",
                "name": "United States"
              }
            ],
            "isIndicator": false,
            "isDefault": false,
            "valueCount": 1
          },
          {
            "dimensionId": "INDICATOR",
            "dimensionName": "Indicator",
            "operator": "in",
            "values": [
              {
                "id": "NGDP",
                "name": "Gross domestic product (GDP), Current prices, Domestic currency"
              },
              {
                "id": "NGDPD",
                "name": "Gross domestic product (GDP), Current prices, US dollar"
              }
            ],
            "isIndicator": true,
            "isDefault": false,
            "valueCount": 2
          }
        ],
        "requestedPeriod": {
          "startPeriod": "2021-01-01",
          "endPeriod": "2026-12-31",
          "isDefault": false
        },
        "execution": {
          "result": "request_failed",
          "reason": "The request to the data source failed.",
          "advice": "This looks like a temporary issue with the data source. You may want to retry the query, or try again shortly."
        },
        "dataExplorerUrl": "https://data.imf.org/en/Data-Explorer?datasetUrn=IMF.RES:WEO(9.0.0)&timeseriesName=USA.NGDP+NGDPD.*&startPeriod=2021-01-01&endPeriod=2026-12-31"
      }
    ],
    "candidateDatasets": []
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "failed",
      "message": "The following queries were executed:\n\n…\n\nDo not reproduce the returned table: the user already sees the data in the widget.\n[Data Query executed at 2026-09-23T10:00:00.000000+00:00]",
      "queries": [
        {
          "urn": "IMF.RES:WEO(9.0.0)",
          "filters": [
            {
              "componentCode": "COUNTRY",
              "operator": "in",
              "values": [
                "USA"
              ]
            },
            {
              "componentCode": "INDICATOR",
              "operator": "in",
              "values": [
                "NGDP",
                "NGDPD"
              ]
            },
            {
              "componentCode": "TIME_PERIOD",
              "operator": "between",
              "values": [
                "2021-01-01",
                "2026-12-31"
              ]
            }
          ],
          "metadata": {
            "countryDimension": "COUNTRY",
            "indicatorDimensions": [
              "INDICATOR"
            ],
            "timePeriodDimension": "TIME_PERIOD",
            "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
            "keyDimensionIdsInDsdOrder": [
              "COUNTRY",
              "INDICATOR",
              "FREQUENCY"
            ]
          },
          "sdmx1Source": "IMF_DATA",
          "disabled": false,
          "queryId": "dq_333e6e65fc"
        }
      ],
      "candidateDatasets": [],
      "missingDimensions": null,
      "pythonCode": "# Uses the [sdmx1 library](https://pypi.org/project/sdmx1/)\n# Install with:\n# ```bash\n# pip install sdmx1\n# ```\n\nimport sdmx\n\nprovider = sdmx.Client(\"IMF_DATA\")\ndata_msg = provider.data(\n    \"IMF.RES,WEO,9.0.0\",\n    key=\"USA.NGDP+NGDPD.\",\n    params={'detail': 'full', 'startPeriod': '2021-01-01', 'endPeriod': '2026-12-31'}\n)",
      "tools": {
        "sdmxProxy": "sdmx_proxy"
      },
      "version": 3
    },
    "statgpt.dialx.ai/client": {
      "status": "failed",
      "queries": [
        {
          "queryId": "dq_333e6e65fc",
          "urn": "IMF.RES:WEO(9.0.0)",
          "datasetName": "World Economic Outlook (WEO)",
          "dataExplorerUrl": "https://data.imf.org/en/Data-Explorer?datasetUrn=IMF.RES:WEO(9.0.0)&timeseriesName=USA.NGDP+NGDPD.*&startPeriod=2021-01-01&endPeriod=2026-12-31",
          "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
          "resourceUris": []
        }
      ],
      "version": 3
    }
  }
}
```

### `not_executed`

The queries were constructed but never ran: `executed` is `false`, and there is no response to read the display names, the explorer link or the resources off.

```json
{
  "structuredContent": {
    "status": "not_executed",
    "message": "data queries constructed. queries were not executed, their status (valid/invalid) is unknown, because data query post-processing is disabled in config",
    "queries": [
      {
        "queryId": "dq_6dd977de4c",
        "datasetUrn": "IMF.RES:WEO(9.0.0)",
        "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
        "executed": false,
        "filters": [
          {
            "dimensionId": "COUNTRY",
            "operator": "in",
            "values": [
              {
                "id": "USA"
              }
            ],
            "isIndicator": false,
            "isDefault": false,
            "valueCount": 1
          },
          {
            "dimensionId": "INDICATOR",
            "operator": "in",
            "values": [
              {
                "id": "NGDP"
              },
              {
                "id": "NGDPD"
              }
            ],
            "isIndicator": true,
            "isDefault": false,
            "valueCount": 2
          }
        ],
        "requestedPeriod": {
          "startPeriod": "2021-01-01",
          "endPeriod": "2026-12-31",
          "isDefault": false
        }
      }
    ],
    "candidateDatasets": []
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "not_executed",
      "message": "data queries constructed. queries were not executed, their status (valid/invalid) is unknown, because data query post-processing is disabled in config",
      "queries": [
        {
          "urn": "IMF.RES:WEO(9.0.0)",
          "filters": [
            {
              "componentCode": "COUNTRY",
              "operator": "in",
              "values": [
                "USA"
              ]
            },
            {
              "componentCode": "INDICATOR",
              "operator": "in",
              "values": [
                "NGDP",
                "NGDPD"
              ]
            },
            {
              "componentCode": "TIME_PERIOD",
              "operator": "between",
              "values": [
                "2021-01-01",
                "2026-12-31"
              ]
            }
          ],
          "metadata": {
            "countryDimension": "COUNTRY",
            "indicatorDimensions": [
              "INDICATOR"
            ],
            "timePeriodDimension": "TIME_PERIOD",
            "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
            "keyDimensionIdsInDsdOrder": [
              "COUNTRY",
              "INDICATOR",
              "FREQUENCY"
            ]
          },
          "sdmx1Source": "IMF_DATA",
          "disabled": false,
          "queryId": "dq_6dd977de4c"
        }
      ],
      "candidateDatasets": [],
      "missingDimensions": null,
      "pythonCode": "# Uses the [sdmx1 library](https://pypi.org/project/sdmx1/)\n# Install with:\n# ```bash\n# pip install sdmx1\n# ```\n\nimport sdmx\n\nprovider = sdmx.Client(\"IMF_DATA\")\ndata_msg = provider.data(\n    \"IMF.RES,WEO,9.0.0\",\n    key=\"USA.NGDP+NGDPD.\",\n    params={'detail': 'full', 'startPeriod': '2021-01-01', 'endPeriod': '2026-12-31'}\n)",
      "tools": {
        "sdmxProxy": "sdmx_proxy"
      },
      "version": 3
    },
    "statgpt.dialx.ai/client": {
      "status": "not_executed",
      "queries": [
        {
          "queryId": "dq_6dd977de4c",
          "urn": "IMF.RES:WEO(9.0.0)",
          "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
          "resourceUris": []
        }
      ],
      "version": 3
    }
  }
}
```

### `dataset_selection_required`

The query matched several datasets. The model gets the ids to narrow it down, each with the query that would run against it; the widget gets the descriptions as well.

```json
{
  "structuredContent": {
    "status": "dataset_selection_required",
    "message": "**Important**: at that point **no data is provided either to you or to user**, only query info. You may select one of the datasets without user's input, whenever you think it's possible, or ask user to select one of the datasets to proceed with query execution. When user selected something, call the same tool mentioning the dataset name or id in the tool call arguments.",
    "queries": [],
    "candidateDatasets": [
      {
        "id": "IMF.RES:WEO(9.0.0)",
        "name": "World Economic Outlook (WEO)",
        "query": {
          "queryId": "dq_6dd977de4c",
          "datasetUrn": "IMF.RES:WEO(9.0.0)",
          "datasetName": "World Economic Outlook (WEO)",
          "provider": "IMF Research Department (RES)",
          "datasetLastUpdated": "2026-04-14",
          "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
          "querySummary": "For the United States, Gross Domestic Product (GDP) in current prices, in domestic currency and US dollars, was retrieved from 2021 to 2026 from the World Economic Outlook.",
          "executed": false,
          "filters": [
            {
              "dimensionId": "COUNTRY",
              "dimensionName": "Country",
              "operator": "in",
              "values": [
                {
                  "id": "USA",
                  "name": "United States"
                }
              ],
              "isIndicator": false,
              "isDefault": false,
              "valueCount": 1
            }
          ],
          "requestedPeriod": {
            "startPeriod": "2021-01-01",
            "endPeriod": "2026-12-31",
            "isDefault": false
          }
        }
      },
      {
        "id": "IMF.STA:NSDP(7.0.0)",
        "name": "National Summary Data Page (NSDP)"
      }
    ]
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "dataset_selection_required",
      "message": "Relevant data can be pulled from the following datasets:\n…\n\n**Important**: at that point **no data is provided either to you or to user**, only query info. You may select one of the datasets without user's input, whenever you think it's possible, or ask user to select one of the datasets to proceed with query execution. When user selected something, call the same tool mentioning the dataset name or id in the tool call arguments.",
      "queries": [],
      "candidateDatasets": [
        {
          "id": "IMF.RES:WEO(9.0.0)",
          "name": "World Economic Outlook (WEO)",
          "description": "Macroeconomic projections.",
          "isOfficial": true
        },
        {
          "id": "IMF.STA:NSDP(7.0.0)",
          "name": "National Summary Data Page (NSDP)",
          "description": null,
          "isOfficial": false
        }
      ],
      "missingDimensions": null,
      "pythonCode": null,
      "tools": {
        "sdmxProxy": "sdmx_proxy"
      },
      "version": 3
    },
    "statgpt.dialx.ai/client": {
      "status": "dataset_selection_required",
      "queries": [],
      "version": 3
    }
  }
}
```

### `missing_dimensions`

The query is incomplete. The model gets a bounded sample of each dimension's values, the widget gets every one of them.

```json
{
  "structuredContent": {
    "status": "missing_dimensions",
    "message": "Which country are you interested in? For example: Country 0, Country 1 or Country 2.",
    "queries": [],
    "missingDimensions": {
      "datasetUrn": "IMF.STA:NSDP(7.0.0)",
      "dimensions": [
        {
          "dimensionId": "COUNTRY",
          "name": "Country",
          "totalValues": 12,
          "sampleValues": [
            {
              "id": "C00",
              "name": "Country 0"
            },
            {
              "id": "C01",
              "name": "Country 1"
            },
            {
              "id": "C02",
              "name": "Country 2"
            },
            {
              "id": "C03",
              "name": "Country 3"
            },
            {
              "id": "C04",
              "name": "Country 4"
            },
            {
              "id": "C05",
              "name": "Country 5"
            },
            {
              "id": "C06",
              "name": "Country 6"
            },
            {
              "id": "C07",
              "name": "Country 7"
            },
            {
              "id": "C08",
              "name": "Country 8"
            },
            {
              "id": "C09",
              "name": "Country 9"
            }
          ]
        }
      ]
    },
    "candidateDatasets": []
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "missing_dimensions",
      "message": "Which country are you interested in? For example: Country 0, Country 1 or Country 2.",
      "queries": [],
      "candidateDatasets": [],
      "missingDimensions": {
        "datasetId": "30f1edda-4f1c-4046-b98d-e844ea152db1",
        "datasetUrn": "IMF.STA:NSDP(7.0.0)",
        "dimensions": [
          {
            "dimensionId": "COUNTRY",
            "name": "Country",
            "availableValues": [
              {
                "id": "C00",
                "name": "Country 0",
                "description": null
              },
              {
                "id": "C01",
                "name": "Country 1",
                "description": null
              },
              {
                "id": "C02",
                "name": "Country 2",
                "description": null
              },
              {
                "id": "C03",
                "name": "Country 3",
                "description": null
              },
              {
                "id": "C04",
                "name": "Country 4",
                "description": null
              },
              {
                "id": "C05",
                "name": "Country 5",
                "description": null
              },
              {
                "id": "C06",
                "name": "Country 6",
                "description": null
              },
              {
                "id": "C07",
                "name": "Country 7",
                "description": null
              },
              {
                "id": "C08",
                "name": "Country 8",
                "description": null
              },
              {
                "id": "C09",
                "name": "Country 9",
                "description": null
              },
              {
                "id": "C10",
                "name": "Country 10",
                "description": null
              },
              {
                "id": "C11",
                "name": "Country 11",
                "description": null
              }
            ]
          }
        ]
      },
      "pythonCode": null,
      "tools": {
        "sdmxProxy": "sdmx_proxy"
      },
      "version": 3
    },
    "statgpt.dialx.ai/client": {
      "status": "missing_dimensions",
      "queries": [],
      "version": 3
    }
  }
}
```

### `invalid_time_period`

The requested period is outside the dataset's range. The model gets the constructed queries, each with the `invalidity` that keeps it from running: a rejected period was never applied, so `requestedPeriod` is absent and `invalidity.rejectedPeriod` carries it instead. A query that misses a required dimension reports `missing_dimensions` there instead. The widget gets no queries.

```json
{
  "structuredContent": {
    "status": "invalid_time_period",
    "message": "The created query contains data according to the selected filters, but the values are only available for a different time period. Please adjust the time period or modify the query.",
    "queries": [
      {
        "queryId": "dq_6dd977de4c",
        "datasetUrn": "IMF.RES:WEO(9.0.0)",
        "datasetName": "World Economic Outlook (WEO)",
        "provider": "IMF Research Department (RES)",
        "datasetLastUpdated": "2026-04-14",
        "datasetUrl": "https://data.imf.org/en/datasets/IMF.RES:WEO",
        "querySummary": "For the United States, Gross Domestic Product (GDP) in current prices, in domestic currency and US dollars, was retrieved from 2021 to 2026 from the World Economic Outlook.",
        "executed": false,
        "filters": [
          {
            "dimensionId": "COUNTRY",
            "dimensionName": "Country",
            "operator": "in",
            "values": [
              {
                "id": "USA",
                "name": "United States"
              }
            ],
            "isIndicator": false,
            "isDefault": false,
            "valueCount": 1
          }
        ],
        "invalidity": {
          "reason": "invalid_time_period",
          "explanation": "The requested start period 2035 is after the last period the dataset has data for (2030).",
          "rejectedPeriod": {
            "rejectedBound": "startPeriod",
            "requestedValue": "2035",
            "availablePeriod": {
              "startPeriod": "1980",
              "endPeriod": "2030"
            }
          }
        }
      }
    ],
    "candidateDatasets": []
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "invalid_time_period",
      "message": "## Result of query construction\n\nThe created query contains data according to the selected filters, but the values are only available for a different time period. Please adjust the time period or modify the query.\n\n## Constructed queries for datasets\n\n…",
      "queries": [],
      "candidateDatasets": [],
      "missingDimensions": null,
      "pythonCode": null,
      "tools": {
        "sdmxProxy": "sdmx_proxy"
      },
      "version": 3
    },
    "statgpt.dialx.ai/client": {
      "status": "invalid_time_period",
      "queries": [],
      "version": 3
    }
  }
}
```

### `no_data`

Nothing relevant was found, and no query was built.

```json
{
  "structuredContent": {
    "status": "no_data",
    "message": "No relevant data found for the query. Try to change the query.",
    "queries": [],
    "candidateDatasets": []
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "no_data",
      "message": "No relevant data found for the query. Try to change the query.",
      "queries": [],
      "candidateDatasets": [],
      "missingDimensions": null,
      "pythonCode": null,
      "tools": {
        "sdmxProxy": "sdmx_proxy"
      },
      "version": 3
    },
    "statgpt.dialx.ai/client": {
      "status": "no_data",
      "queries": [],
      "version": 3
    }
  }
}
```
