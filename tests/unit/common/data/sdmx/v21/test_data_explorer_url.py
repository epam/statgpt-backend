from unittest.mock import MagicMock

import pytest

from statgpt.common.data.sdmx.common.data_explorer_url import (
    DataExplorerUrlConfig,
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


def test_build_data_explorer_url_query_sdmx_key_default() -> None:
    dsd = MagicMock()
    q = _sample_sdmx_query()
    url = build_data_explorer_url_query(
        EXPLORER_BASE,
        SHORT_URN,
        dsd,
        q.get_key(),
        q,
        config=None,
        sdmx_key_builder=lambda _d, _k: ".A.x.*.B.+",
    )
    assert (
        url == "https://example.net/data-viewer?"
        "urn=AGENCY.REG:DS%281.0.0%29&"
        "filter=.A.x.*.B.+&"
        "startPeriod=2021-01-01&"
        "endPeriod=2026-12-31"
    )


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
    assert url == "https://example.net/data-viewer?datasetUrn=AGENCY.REG:FLOW%281.0.0%29"


def test_build_data_explorer_dataset_url_defaults() -> None:
    url = build_data_explorer_dataset_url(EXPLORER_BASE, SHORT_URN, config=None)
    assert url == "https://example.net/data-viewer?urn=AGENCY.REG:DS(1.0.0)"


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
    assert (
        url == "https://example.org/topics/ABC/data?"
        "filter=AREA_TXT%3DDE%5ECOUNTERPART_AREA%3D1E%5ETIMESPAN%3D2020-01-01_2021-12-31"
    )


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
    assert url == "https://example.net/data-viewer?filter=EER_TYPE%3DR%2BN"


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
    assert url == "https://example.net/data-viewer?filter=EER_TYPE%3DN%7CR"


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
    assert url == "https://example.net/data-viewer?filter=REF_AREA_TXT%3DCanada"
