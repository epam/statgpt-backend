# Data Query MCP response

The Data Query tool answers one `tools/call` with three surfaces, each written for a different
reader:

| Surface | Reader | Contents |
|---|---|---|
| `content` | the model, and the user through it | The rendered text response, plus one `text/csv` and/or `text/markdown` resource per dataset (see [`mcpResources`](README.md#mcp-server)). |
| `structuredContent` | the calling model | The queries the pipeline produced, or what a follow-up query would need when it produced none. Validated against the tool's declared `outputSchema`. |
| `result._meta` | the clients | One namespaced payload per audience: the MCP-App widget and programmatic clients (e.g. Deep Research). |

The whole response carries one version number, `3`, in `structuredContent.version` and in every
`_meta` payload. See [the migration guide](MCP_DATA_QUERY_RESPONSE_MIGRATION.md) for what changed
from version 2.

## `structuredContent`

| Field | Notes |
|---|---|
| `queries[].queryId` | Short id of the query within the response. Joins it to its resources and to the `_meta` payloads. |
| `queries[].datasetUrn` | URN of the queried dataset. |
| `queries[].datasetName` | Dataset name, when a response carried one. |
| `queries[].executed` | `false` for a query that was constructed but never ran. |
| `queries[].filters[]` | One entry per filtered dimension: `dimensionId`, `dimensionName`, `operator`, `values[].id` / `values[].name`. A dimension with no filter is not listed - every one of its values is included. |
| `queries[].filters[].totalValues` | Present only when `values` was truncated to the first 10. |
| `queries[].requestedPeriod` | `startPeriod` / `endPeriod`, named after the SDMX REST query parameters. The time period is reported here, not as another filter. |
| `queries[].factualPeriod` | The period the returned data actually covers. |
| `queries[].seriesCount` | Number of series returned, absent when the query returned no data. |
| `missingDimensions` | The dimensions a follow-up query must specify, with `totalValues` and up to 10 `sampleValues` each. |
| `candidateDatasets[]` | Datasets to narrow the query to, as `id` / `name` / `isOfficial`. |
| `version` | `3`. |

Null fields are omitted. The pipeline status, the python snippet and the companion tool names are
not here: the text block explains the outcome to the model, and the clients read `_meta`.

## `result._meta`

Two payloads, each under a namespaced key and each toggleable per channel:

```yaml
details:
  mcpMeta:
    namespace: "statgpt.dialx.ai"       # supports $env:{VAR}
    mcpApp:
      enabledStr: "True"
    client:
      enabledStr: "True"
```

- **`{namespace}/mcp-app`** - what the UI widget renders and edits: the pipeline `status`, the
  `message`, the SDMX query model (`urn`, `filters`, `metadata`, `sdmx1Source`, `disabled`) with its
  `queryId`, the full `candidateDatasets` / `missingDimensions` value lists, the reproducible
  `pythonCode`, and the companion `tools`. Null fields are kept, so the payload's shape does not
  change with the outcome.
- **`{namespace}/client`** - what a programmatic client needs to present and navigate the result:
  the `status`, the `message`, and per query the `dataExplorerUrl`, the `datasetUrl`, the
  `resourceUris` and the `seriesCount`. Null fields are omitted.

`_meta` is omitted entirely when both audiences are disabled.

## Resource URIs

A resource URI is `statgpt://data_query/{dataset}/{queryId}.{csv|md}`, so a client can join a table
to the query that produced it. The `queryId` is derived from the query itself plus the date it ran:
the same query executed again on the same date gets the same id, while the same query a month later
- which may well return different data - gets a new one. A response that carries no query keeps a
timestamp stem instead.

## Examples

One per pipeline status. The `content` blocks are omitted for brevity; `resourceUris` lists the
resources `content` carries.

### `data_available`

The queries ran and returned data.

```json
{
  "structuredContent": {
    "queries": [
      {
        "queryId": "dq_333e6e65fc",
        "datasetUrn": "IMF.RES:WEO(9.0.0)",
        "datasetName": "World Economic Outlook (WEO)",
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
            ]
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
            ]
          }
        ],
        "requestedPeriod": {
          "startPeriod": "2021-01-01",
          "endPeriod": "2026-12-31"
        },
        "factualPeriod": {
          "startPeriod": "2021",
          "endPeriod": "2025"
        },
        "seriesCount": 2
      }
    ],
    "candidateDatasets": [],
    "version": 3
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "data_available",
      "message": null,
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
    "queries": [
      {
        "queryId": "dq_333e6e65fc",
        "datasetUrn": "IMF.RES:WEO(9.0.0)",
        "datasetName": "World Economic Outlook (WEO)",
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
            ]
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
            ]
          }
        ],
        "requestedPeriod": {
          "startPeriod": "2021-01-01",
          "endPeriod": "2026-12-31"
        }
      }
    ],
    "candidateDatasets": [],
    "version": 3
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "executed_no_data",
      "message": null,
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
    "queries": [
      {
        "queryId": "dq_333e6e65fc",
        "datasetUrn": "IMF.RES:WEO(9.0.0)",
        "datasetName": "World Economic Outlook (WEO)",
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
            ]
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
            ]
          }
        ],
        "requestedPeriod": {
          "startPeriod": "2021-01-01",
          "endPeriod": "2026-12-31"
        }
      }
    ],
    "candidateDatasets": [],
    "version": 3
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "failed",
      "message": null,
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

The queries were constructed but never ran: `executed` is `false`, and there is no response to read the display names, the links or the resources off.

```json
{
  "structuredContent": {
    "queries": [
      {
        "queryId": "dq_6dd977de4c",
        "datasetUrn": "IMF.RES:WEO(9.0.0)",
        "executed": false,
        "filters": [
          {
            "dimensionId": "COUNTRY",
            "operator": "in",
            "values": [
              {
                "id": "USA"
              }
            ]
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
            ]
          }
        ],
        "requestedPeriod": {
          "startPeriod": "2021-01-01",
          "endPeriod": "2026-12-31"
        }
      }
    ],
    "candidateDatasets": [],
    "version": 3
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "not_executed",
      "message": null,
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

The query matched several datasets. The model gets the ids to narrow it down; the widget gets the descriptions as well.

```json
{
  "structuredContent": {
    "queries": [],
    "candidateDatasets": [
      {
        "id": "IMF.RES:WEO(9.0.0)",
        "name": "World Economic Outlook (WEO)",
        "isOfficial": true
      },
      {
        "id": "IMF.STA:NSDP(7.0.0)",
        "name": "National Summary Data Page (NSDP)",
        "isOfficial": false
      }
    ],
    "version": 3
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "dataset_selection_required",
      "message": "Several datasets match your query. Which one should I use?",
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
      "message": "Several datasets match your query. Which one should I use?",
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
    "candidateDatasets": [],
    "version": 3
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "missing_dimensions",
      "message": "Your query is missing the required dimension \"Country\".",
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
      "message": "Your query is missing the required dimension \"Country\".",
      "queries": [],
      "version": 3
    }
  }
}
```

### `invalid_time_period`

The requested period is outside the dataset's range. The constructed queries are deliberately not reported: the rejected period was never applied to them, so they would describe a query the user did not ask for.

```json
{
  "structuredContent": {
    "queries": [],
    "candidateDatasets": [],
    "version": 3
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "invalid_time_period",
      "message": "The selected end date (2030) is outside the available range.",
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
      "message": "The selected end date (2030) is outside the available range.",
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
    "queries": [],
    "candidateDatasets": [],
    "version": 3
  },
  "_meta": {
    "statgpt.dialx.ai/mcp-app": {
      "status": "no_data",
      "message": "No relevant data was found for the provided query.",
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
      "message": "No relevant data was found for the provided query.",
      "queries": [],
      "version": 3
    }
  }
}
```
