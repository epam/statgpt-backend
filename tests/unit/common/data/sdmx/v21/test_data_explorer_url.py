from unittest.mock import MagicMock, patch
from urllib.parse import unquote

import pytest

from statgpt.common.data.sdmx.common.data_explorer_url import DataExplorerUrlConfig
from statgpt.common.data.sdmx.v21.data_explorer_url import (
    build_data_explorer_dataset_url,
    build_data_explorer_url_query,
)
from statgpt.common.data.sdmx.v21.query import (
    SdmxDataSetQuery,
    SdmxQueryReadinessStatus,
    TimeDimensionQuery,
)

EXPLORER_BASE = "https://example.net/data-viewer"
SHORT_URN = "AGENCY.REG:DS(1.0.0)"


def _sample_sdmx_query() -> SdmxDataSetQuery:
    return SdmxDataSetQuery(
        status=SdmxQueryReadinessStatus.READY,
        categorical_dimensions={"A": ["x"]},
        time_dimension_query=TimeDimensionQuery(
            time_dimension_id="TIME_PERIOD",
            start_period="2021-01-01",
            end_period="2026-12-31",
        ),
        missing_dimensions=[],
    )


@patch(
    "statgpt.common.data.sdmx.v21.data_explorer_url.convert_keys_to_str",
    return_value=".A.x.*.B.+",
)
def test_build_data_explorer_url_query_sdmx_key_default(_) -> None:
    dsd = MagicMock()
    q = _sample_sdmx_query()
    url = build_data_explorer_url_query(EXPLORER_BASE, SHORT_URN, dsd, q.get_key(), q, config=None)
    assert url.startswith(f"{EXPLORER_BASE}?")
    assert "urn=" in url
    assert "filter=" in url
    assert "startPeriod" in url
    assert "endPeriod" in url


def test_build_data_explorer_url_query_dataset_urn_param_no_series_no_time() -> None:
    dsd = MagicMock()
    q = _sample_sdmx_query()
    cfg = DataExplorerUrlConfig(
        dataset_urn_param="datasetUrn",
        include_series_key_filter=False,
        time_encoding="none",
    )
    url = build_data_explorer_url_query(
        EXPLORER_BASE, "AGENCY.REG:FLOW(1.0.0)", dsd, q.get_key(), q, config=cfg
    )
    assert url.startswith(f"{EXPLORER_BASE}?")
    assert "datasetUrn=" in url
    assert "filter=" not in url
    assert "startPeriod" not in url


def test_build_data_explorer_dataset_url_defaults() -> None:
    url = build_data_explorer_dataset_url(EXPLORER_BASE, SHORT_URN, config=None)
    assert url.startswith(f"{EXPLORER_BASE}?urn=")
    assert "AGENCY" in url


def test_aggregated_filter_includes_time_segment() -> None:
    c1, c2 = MagicMock(), MagicMock()
    c1.id, c2.id = "COUNTRY", "COUNTERPART_COUNTRY"
    dsd = MagicMock()
    dsd.dimensions.components = [c1, c2]

    q = SdmxDataSetQuery(
        status=SdmxQueryReadinessStatus.READY,
        categorical_dimensions={
            "COUNTERPART_COUNTRY": ["1E"],
            "COUNTRY": ["DE"],
        },
        time_dimension_query=TimeDimensionQuery(
            time_dimension_id="TIME_PERIOD",
            start_period="2020-01-01",
            end_period="2021-12-31",
        ),
        missing_dimensions=[],
    )

    cfg = DataExplorerUrlConfig(
        include_dataflow_urn_param=False,
        filter_format="key_value_aggregated",
        time_encoding="in_aggregated_filter",
        aggregated_time_param="TIMESPAN",
        series_key_filter_param="filter",
        aggregated_dimension_param_names={
            "COUNTRY": "AREA_TXT",
            "COUNTERPART_COUNTRY": "COUNTERPART_AREA",
        },
    )
    url = build_data_explorer_url_query(
        "https://example.org/topics/ABC/data", SHORT_URN, dsd, q.get_key(), q, config=cfg
    )
    assert "?" in url
    assert "filter=" in url
    decoded = unquote(url)
    assert "COUNTERPART_AREA=1E" in decoded
    assert "AREA_TXT=DE" in decoded
    assert "TIMESPAN=2020-01-01_2021-12-31" in decoded
    assert "urn=" not in url


def test_aggregated_filter_requires_time_param() -> None:
    with pytest.raises(ValueError, match="aggregated"):
        DataExplorerUrlConfig(
            filter_format="key_value_aggregated",
            time_encoding="in_aggregated_filter",
        )


def test_dataset_url_without_urn_uses_base_only() -> None:
    cfg = DataExplorerUrlConfig(include_dataflow_urn_param=False)
    u = build_data_explorer_dataset_url("https://example.org/topics/ABC/data", "ignored-urn", cfg)
    assert u == "https://example.org/topics/ABC/data"


def test_aggregated_filter_encodes_plus_as_data_not_space() -> None:
    dsd = MagicMock()
    d1 = MagicMock()
    d1.id = "EER_TYPE"
    dsd.dimensions.components = [d1]
    q = SdmxDataSetQuery(
        status=SdmxQueryReadinessStatus.READY,
        categorical_dimensions={"EER_TYPE": ["R", "N"]},
        time_dimension_query=None,
        missing_dimensions=[],
    )
    cfg = DataExplorerUrlConfig(
        include_dataflow_urn_param=False,
        filter_format="key_value_aggregated",
        include_series_key_filter=True,
        time_encoding="none",
    )
    url = build_data_explorer_url_query(EXPLORER_BASE, SHORT_URN, dsd, q.get_key(), q, config=cfg)
    assert "%2B" in url
    assert "%20" not in url
    assert "EER_TYPE=R+N" in unquote(url)


def test_aggregated_filter_supports_custom_values_separator() -> None:
    dsd = MagicMock()
    d1 = MagicMock()
    d1.id = "EER_TYPE"
    dsd.dimensions.components = [d1]
    q = SdmxDataSetQuery(
        status=SdmxQueryReadinessStatus.READY,
        categorical_dimensions={"EER_TYPE": ["N", "R"]},
        time_dimension_query=None,
        missing_dimensions=[],
    )
    cfg = DataExplorerUrlConfig(
        include_dataflow_urn_param=False,
        filter_format="key_value_aggregated",
        include_series_key_filter=True,
        time_encoding="none",
        aggregated_values_separator="|",
    )
    url = build_data_explorer_url_query(EXPLORER_BASE, SHORT_URN, dsd, q.get_key(), q, config=cfg)
    decoded = unquote(url)
    assert "EER_TYPE=N|R" in decoded
    assert "EER_TYPE=N+R" not in decoded


def test_aggregated_filter_can_use_dimension_names() -> None:
    dsd = MagicMock()
    d1 = MagicMock()
    d1.id = "REF_AREA"
    dsd.dimensions.components = [d1]
    q = SdmxDataSetQuery(
        status=SdmxQueryReadinessStatus.READY,
        categorical_dimensions={"REF_AREA": ["CA"]},
        time_dimension_query=None,
        missing_dimensions=[],
    )
    cfg = DataExplorerUrlConfig(
        include_dataflow_urn_param=False,
        filter_format="key_value_aggregated",
        include_series_key_filter=True,
        time_encoding="none",
        aggregated_dimension_param_names={"REF_AREA": "REF_AREA_TXT"},
        aggregated_dimension_value_mode={"REF_AREA": "name"},
    )
    url = build_data_explorer_url_query(
        EXPLORER_BASE,
        SHORT_URN,
        dsd,
        q.get_key(),
        q,
        config=cfg,
        dimension_values_resolver=lambda dim_id, values: (
            ["Canada"] if dim_id == "REF_AREA" else values
        ),
    )
    assert "REF_AREA_TXT=Canada" in unquote(url)
