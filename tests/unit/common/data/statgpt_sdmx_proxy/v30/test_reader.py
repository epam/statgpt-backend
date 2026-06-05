"""Tests for :class:`StatGptSdmxProxyDataReader`.

The standard fixtures exercise SDMX-JSON 2.0.0 payloads returned by non-proxy
SDMX 3.0 APIs. ``data_all_attribute_levels.json`` is a small hand-crafted proxy
message that covers the proxy-only branches documented in the reader: the
``dimensionGroup`` attribute category, uncoded dimension values using
``{"value": "..."}``, and uncoded raw-string attributes in data arrays.
"""

import io
import json
from pathlib import Path

import pytest
from sdmx.message import DataMessage
from sdmx.model.v21 import ActionType, AllDimensions, AttributeValue, Code

from statgpt.common.data.statgpt_sdmx_proxy.v30.reader import StatGptSdmxProxyDataReader
from statgpt.common.data.statgpt_sdmx_proxy.v30.sdmx_client import (
    _dataset_level_attribute_map_from_data_message,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _parse(filename: str) -> DataMessage:
    content = io.BytesIO((_FIXTURES / filename).read_bytes())
    return StatGptSdmxProxyDataReader().convert(content)


def _attr_code_id(av: AttributeValue) -> str | None:
    val = av.value
    return val.id if isinstance(val, Code) else None


class TestStubEerReader:
    """Tests for the STUB_ORG TEST_EER fixture (monthly + daily EER series)."""

    @pytest.fixture(scope="class")
    def msg(self) -> DataMessage:
        return _parse("stub_eer.json")

    @pytest.fixture(scope="class")
    def dataset(self, msg: DataMessage):
        return msg.data[0]

    @pytest.fixture(scope="class")
    def monthly_key(self, dataset):
        return next(sk for sk in dataset.series if sk.values["FREQ"].value == "M")

    @pytest.fixture(scope="class")
    def daily_key(self, dataset):
        return next(sk for sk in dataset.series if sk.values["FREQ"].value == "D")

    # --- Header ---

    def test_returns_data_message(self, msg: DataMessage) -> None:
        assert isinstance(msg, DataMessage)

    def test_header_id(self, msg: DataMessage) -> None:
        assert msg.header.id == "stub-fixture-eer"

    def test_header_prepared_year(self, msg: DataMessage) -> None:
        assert msg.header.prepared is not None
        assert msg.header.prepared.year == 2026

    def test_header_sender_id(self, msg: DataMessage) -> None:
        assert msg.header.sender is not None
        assert msg.header.sender.id == "unknown"

    # --- Dataset ---

    def test_single_dataset(self, msg: DataMessage) -> None:
        assert len(msg.data) == 1

    def test_action_is_information(self, dataset) -> None:
        assert dataset.action == ActionType.information

    # --- Dimensions ---

    def test_dimension_ids(self, msg: DataMessage) -> None:
        assert {d.id for d in msg.structure.dimensions} == {
            "FREQ",
            "EER_TYPE",
            "EER_BASKET",
            "REF_AREA",
            "TIME_PERIOD",
        }

    def test_observation_dimension_is_time_period(self, msg: DataMessage) -> None:
        obs_dims = msg.observation_dimension
        assert obs_dims is not None
        assert len(obs_dims) == 1
        assert obs_dims[0].id == "TIME_PERIOD"

    # --- Series count and keys ---

    def test_two_series(self, dataset) -> None:
        assert len(dataset.series) == 2

    def test_monthly_series_dimension_values(self, monthly_key) -> None:
        assert monthly_key.values["FREQ"].value == "M"
        assert monthly_key.values["EER_TYPE"].value == "N"
        assert monthly_key.values["EER_BASKET"].value == "B"
        assert monthly_key.values["REF_AREA"].value == "DE"

    def test_daily_series_dimension_values(self, daily_key) -> None:
        assert daily_key.values["FREQ"].value == "D"
        assert daily_key.values["EER_TYPE"].value == "N"
        assert daily_key.values["EER_BASKET"].value == "B"
        assert daily_key.values["REF_AREA"].value == "DE"

    # --- Observations ---

    def test_monthly_series_observation_count(self, dataset, monthly_key) -> None:
        assert len(dataset.series[monthly_key]) == 5

    def test_daily_series_observation_count(self, dataset, daily_key) -> None:
        assert len(dataset.series[daily_key]) == 5

    def test_monthly_first_obs_time_period(self, dataset, monthly_key) -> None:
        obs = dataset.series[monthly_key][0]
        assert obs.key.values["TIME_PERIOD"].value == "2024-M01"

    def test_daily_first_obs_time_period(self, dataset, daily_key) -> None:
        obs = dataset.series[daily_key][0]
        assert obs.key.values["TIME_PERIOD"].value == "2024-01-01"

    def test_monthly_first_obs_value(self, dataset, monthly_key) -> None:
        obs = dataset.series[monthly_key][0]
        assert obs.value == "102.23"

    def test_monthly_obs_time_periods_sequential(self, dataset, monthly_key) -> None:
        periods = [obs.key.values["TIME_PERIOD"].value for obs in dataset.series[monthly_key]]
        assert periods == ["2024-M01", "2024-M02", "2024-M03", "2024-M04", "2024-M05"]

    # --- Series-level attributes ---

    def test_monthly_series_collection_attribute(self, monthly_key) -> None:
        assert _attr_code_id(monthly_key.attrib["COLLECTION"]) == "A"

    def test_monthly_series_unit_measure_attribute(self, monthly_key) -> None:
        assert _attr_code_id(monthly_key.attrib["UNIT_MEASURE"]) == "882"

    def test_monthly_series_title_ts_attribute(self, monthly_key) -> None:
        assert (
            _attr_code_id(monthly_key.attrib["TITLE_TS"])
            == "Germany - Nominal - Broad (64 economies)"
        )

    def test_time_format_absent_from_series_attrib(self, monthly_key) -> None:
        assert "TIME_FORMAT" not in monthly_key.attrib

    # --- Observation-level attributes ---

    def test_monthly_first_obs_status_normal(self, dataset, monthly_key) -> None:
        obs = dataset.series[monthly_key][0]
        assert _attr_code_id(obs.attrib["OBS_STATUS"]) == "A"

    def test_monthly_first_obs_conf_free(self, dataset, monthly_key) -> None:
        obs = dataset.series[monthly_key][0]
        assert _attr_code_id(obs.attrib["OBS_CONF"]) == "F"


class TestImfWeoReader:
    """Tests for the IMF WEO fixture (sparse series attribute arrays, annual time periods)."""

    @pytest.fixture(scope="class")
    def msg(self) -> DataMessage:
        return _parse("imf_weo.json")

    @pytest.fixture(scope="class")
    def dataset(self, msg: DataMessage):
        return msg.data[0]

    @pytest.fixture(scope="class")
    def ngdp_r_key(self, dataset):
        return next(sk for sk in dataset.series if sk.values["INDICATOR"].value == "NGDP_R")

    @pytest.fixture(scope="class")
    def ngdp_rpch_key(self, dataset):
        return next(sk for sk in dataset.series if sk.values["INDICATOR"].value == "NGDP_RPCH")

    # --- Header ---

    def test_returns_data_message(self, msg: DataMessage) -> None:
        assert isinstance(msg, DataMessage)

    def test_header_id(self, msg: DataMessage) -> None:
        assert msg.header.id == "7eee9c36-87cb-47ae-b942-25ebf0569ace"

    def test_header_sender_id(self, msg: DataMessage) -> None:
        assert msg.header.sender is not None
        assert msg.header.sender.id == "unknown"

    # --- Dataset ---

    def test_single_dataset(self, msg: DataMessage) -> None:
        assert len(msg.data) == 1

    def test_action_is_replace(self, dataset) -> None:
        assert dataset.action == ActionType.replace

    # --- Dimensions ---

    def test_dimension_ids(self, msg: DataMessage) -> None:
        assert {d.id for d in msg.structure.dimensions} == {
            "COUNTRY",
            "INDICATOR",
            "FREQUENCY",
            "TIME_PERIOD",
        }

    def test_observation_dimension_is_time_period(self, msg: DataMessage) -> None:
        obs_dims = msg.observation_dimension
        assert obs_dims is not None
        assert len(obs_dims) == 1
        assert obs_dims[0].id == "TIME_PERIOD"

    # --- Series count and keys ---

    def test_two_series(self, dataset) -> None:
        assert len(dataset.series) == 2

    def test_ngdp_r_country_usa(self, ngdp_r_key) -> None:
        assert ngdp_r_key.values["COUNTRY"].value == "USA"

    def test_ngdp_r_frequency_annual(self, ngdp_r_key) -> None:
        assert ngdp_r_key.values["FREQUENCY"].value == "A"

    def test_ngdp_rpch_country_usa(self, ngdp_rpch_key) -> None:
        assert ngdp_rpch_key.values["COUNTRY"].value == "USA"

    # --- Observations ---

    def test_ngdp_r_three_observations(self, dataset, ngdp_r_key) -> None:
        assert len(dataset.series[ngdp_r_key]) == 3

    def test_ngdp_rpch_three_observations(self, dataset, ngdp_rpch_key) -> None:
        assert len(dataset.series[ngdp_rpch_key]) == 3

    def test_ngdp_r_obs_time_periods_are_annual(self, dataset, ngdp_r_key) -> None:
        periods = [obs.key.values["TIME_PERIOD"].value for obs in dataset.series[ngdp_r_key]]
        assert periods == ["2026", "2027", "2028"]

    def test_ngdp_r_first_obs_value(self, dataset, ngdp_r_key) -> None:
        obs = dataset.series[ngdp_r_key][0]
        assert obs.value == "24407270882000"

    def test_ngdp_rpch_first_obs_value(self, dataset, ngdp_rpch_key) -> None:
        obs = dataset.series[ngdp_rpch_key][0]
        assert obs.value == "2.323695"

    # --- Series-level attributes resolved from sparse nullable arrays ---

    def test_ngdp_r_scale_is_billions(self, ngdp_r_key) -> None:
        assert _attr_code_id(ngdp_r_key.attrib["SCALE"]) == "9"

    def test_ngdp_rpch_scale_is_units(self, ngdp_rpch_key) -> None:
        assert _attr_code_id(ngdp_rpch_key.attrib["SCALE"]) == "0"

    def test_ngdp_r_decimals_displayed(self, ngdp_r_key) -> None:
        assert _attr_code_id(ngdp_r_key.attrib["DECIMALS_DISPLAYED"]) == "3"

    def test_ngdp_r_overlap_resolved(self, ngdp_r_key) -> None:
        assert _attr_code_id(ngdp_r_key.attrib["OVERLAP"]) == "OL"

    def test_ngdp_r_country_update_date(self, ngdp_r_key) -> None:
        assert _attr_code_id(ngdp_r_key.attrib["COUNTRY_UPDATE_DATE"]) == "9/30/2025"

    def test_attributes_with_empty_values_list_absent(self, ngdp_r_key) -> None:
        assert "FUNCTIONAL_CAT" not in ngdp_r_key.attrib


class TestImfWeoDatasetLevelAttributesReader:
    """Tests for the IMF WEO dataset-level attributes fixture.

    This payload is returned by the ``?attributes=dataset&measures=none``
    endpoint.  The ``dataSets[0].attributes`` array drives
    ``_make_dataset_level_attrs``; attributes beyond the array length are
    resolved implicitly when their coded-values list has exactly one entry.
    """

    @pytest.fixture(scope="class")
    def msg(self) -> DataMessage:
        return _parse("imf_weo_dataset_attrs.json")

    @pytest.fixture(scope="class")
    def dataset(self, msg: DataMessage):
        return msg.data[0]

    @pytest.fixture(scope="class")
    def attr_map(self, msg: DataMessage) -> dict:
        return _dataset_level_attribute_map_from_data_message(msg)

    # --- Header / message envelope ---

    def test_returns_data_message(self, msg: DataMessage) -> None:
        assert isinstance(msg, DataMessage)

    def test_header_id(self, msg: DataMessage) -> None:
        assert msg.header.id == "725c23b7-3be7-4baf-9354-6a56e425f72e"

    def test_single_dataset(self, msg: DataMessage) -> None:
        assert len(msg.data) == 1

    # --- Core UPDATE_DATE value (the main purpose of this endpoint) ---

    def test_update_date_present_in_dataset_attrib(self, dataset) -> None:
        assert "UPDATE_DATE" in dataset.attrib

    def test_update_date_is_correct(self, dataset) -> None:
        assert _attr_code_id(dataset.attrib["UPDATE_DATE"]) == "2026-04-15T13:00:00Z"

    def test_update_date_in_attr_map(self, attr_map: dict) -> None:
        assert attr_map["UPDATE_DATE"] == "2026-04-15T13:00:00Z"

    # --- Other explicitly indexed dataset-level attributes ---

    def test_publisher_resolved(self, dataset) -> None:
        assert _attr_code_id(dataset.attrib["PUBLISHER"]) == "IMF"

    def test_department_resolved(self, dataset) -> None:
        assert _attr_code_id(dataset.attrib["DEPARTMENT"]) == "RES"

    def test_contact_point_resolved(self, dataset) -> None:
        assert _attr_code_id(dataset.attrib["CONTACT_POINT"]) == "datahelp@imf.org"

    def test_language_resolved(self, dataset) -> None:
        assert _attr_code_id(dataset.attrib["LANGUAGE"]) == "EN"

    def test_publication_date_resolved(self, dataset) -> None:
        assert _attr_code_id(dataset.attrib["PUBLICATION_DATE"]) == "2026-04-14T13:00:00Z"

    def test_access_sharing_level_resolved(self, dataset) -> None:
        assert _attr_code_id(dataset.attrib["ACCESS_SHARING_LEVEL"]) == "PUBLIC_OPEN"

    # --- Attributes with empty coded-values list resolve to None ---

    def test_doi_is_none(self, dataset) -> None:
        assert dataset.attrib["DOI"].value is None

    def test_author_is_none(self, dataset) -> None:
        assert dataset.attrib["AUTHOR"].value is None

    def test_doi_none_in_attr_map(self, attr_map: dict) -> None:
        assert attr_map["DOI"] is None

    # --- Implicit single-value inference (index beyond the array length) ---

    def test_security_classification_implicit(self, dataset) -> None:
        assert _attr_code_id(dataset.attrib["SECURITY_CLASSIFICATION"]) == "PUB"

    def test_short_source_citation_implicit(self, dataset) -> None:
        assert _attr_code_id(dataset.attrib["SHORT_SOURCE_CITATION"]) == "IMF staff calculations."

    def test_suggested_citation_implicit(self, attr_map: dict) -> None:
        assert attr_map["SUGGESTED_CITATION"] is not None
        assert "World Economic Outlook" in attr_map["SUGGESTED_CITATION"]

    # --- Attributes with empty values and beyond the array are absent ---

    def test_full_source_citation_absent(self, dataset) -> None:
        assert "FULL_SOURCE_CITATION" not in dataset.attrib

    # --- Flat attr_map mirrors ds.attrib resolved values ---

    def test_attr_map_publisher(self, attr_map: dict) -> None:
        assert attr_map["PUBLISHER"] == "IMF"

    def test_attr_map_update_date_matches_attrib(self, dataset, attr_map: dict) -> None:
        assert attr_map["UPDATE_DATE"] == _attr_code_id(dataset.attrib["UPDATE_DATE"])


FIXTURE = Path(__file__).parent / "data_all_attribute_levels.json"


def _parse_payload(payload: dict) -> DataMessage:
    return StatGptSdmxProxyDataReader().convert(io.BytesIO(json.dumps(payload).encode()))


def _parse_fixture() -> DataMessage:
    return StatGptSdmxProxyDataReader().convert(io.BytesIO(FIXTURE.read_bytes()), structure=None)


def _attr_display(attrib: dict) -> dict[str, str | None]:
    return {
        key: (value.value.id if hasattr(value.value, "id") else value.value)
        for key, value in attrib.items()
    }


def _series_by_country(dataset) -> dict:
    return {sk.values["COUNTRY"].value: (sk, obs) for sk, obs in dataset.series.items()}


def _obs_by_time(observations) -> dict:
    return {o.dimension.values["TIME_PERIOD"].value: o for o in observations}


def test_proxy_fixture_uses_dimension_group_attribute_category() -> None:
    """Proxy payloads keep group attributes separate from series attributes."""
    reader = StatGptSdmxProxyDataReader()
    reader.convert(io.BytesIO(FIXTURE.read_bytes()))
    attr_levels = {attr.id: level for attr, level in reader._attr_level.items()}

    assert attr_levels["NA_STO"] == "dimensionGroup"
    assert attr_levels["SERIES_NAME"] == "dimensionGroup"
    assert attr_levels["BASE_YEAR"] == "dimensionGroup"


def test_proxy_fixture_parses_uncoded_dimension_value_objects() -> None:
    """TIME_PERIOD values use the proxy ``{"value": "..."}`` syntax."""
    series = _series_by_country(_parse_fixture().data[0])
    usa_obs = _obs_by_time(series["USA"][1])

    assert sorted(usa_obs) == ["2020", "2021"]


def test_reader_parses_proxy_payload_with_only_observation_dimensions() -> None:
    """Proxy responses can carry all dimensions directly on bare observations."""
    msg = _parse_payload(
        {
            "meta": {
                "id": "OBSERVATION-ONLY",
                "prepared": "2026-06-02T00:00:00",
                "sender": {"id": "TEST"},
            },
            "data": {
                "dataSets": [
                    {
                        "action": "Replace",
                        "observations": {
                            "0:0": ["42", 0],
                        },
                    }
                ],
                "structures": [
                    {
                        "dimensions": {
                            "observation": [
                                {
                                    "id": "TIME_PERIOD",
                                    "keyPosition": 0,
                                    "values": [{"value": "2026"}],
                                },
                                {
                                    "id": "REF_AREA",
                                    "keyPosition": 1,
                                    "values": [{"id": "US"}],
                                },
                            ]
                        },
                        "attributes": {
                            "observation": [
                                {
                                    "id": "OBS_STATUS",
                                    "relationship": {"observation": {}},
                                    "values": [{"id": "A"}],
                                }
                            ]
                        },
                        "measures": {
                            "observation": [
                                {
                                    "id": "OBS_VALUE",
                                    "values": [],
                                }
                            ]
                        },
                    }
                ],
            },
        }
    )

    dataset = msg.data[0]
    assert msg.observation_dimension is AllDimensions
    assert len(dataset.series) == 0
    assert len(dataset.obs) == 1

    obs = dataset.obs[0]
    assert obs.dimension.values["TIME_PERIOD"].value == "2026"
    assert obs.dimension.values["REF_AREA"].value == "US"
    assert obs.value == "42"
    assert _attr_display(obs.attached_attribute) == {"OBS_STATUS": "A"}


def test_reader_parses_dataset_level_attributes() -> None:
    """dataSet-level attributes resolve across coded, inline-string, localized and null slots."""
    attrib = _attr_display(_parse_fixture().data[0].attrib)

    assert attrib["SOURCE_DATASET"] == "IMF"  # coded index -> code id
    assert attrib["CONTACT"] == "data@imf.org"  # inline plain string
    assert attrib["TITLE"] == "World Economic Outlook"  # inline localized {"en": ...}
    assert attrib["NOTE"] is None  # null slot


def test_reader_parses_series_level_attributes() -> None:
    """Series-level attributes resolve per series; null slots are skipped."""
    dataset = _parse_fixture().data[0]
    assert len(dataset.series) == 2
    series = _series_by_country(dataset)

    usa_key, _ = series["USA"]
    usa_attrib = _attr_display(usa_key.attrib)
    assert usa_attrib["SCALE"] == "9"
    assert usa_attrib["DECIMALS"] == "3"
    assert usa_attrib["COUNTRY_UPDATE_DATE"] == "2025-09-30"

    fra_key, _ = series["FRA"]
    fra_attrib = _attr_display(fra_key.attrib)
    assert fra_attrib["SCALE"] == "9"
    assert fra_attrib["COUNTRY_UPDATE_DATE"] == "2025-10-01"
    assert "DECIMALS" not in fra_attrib  # null slot skipped


def test_reader_reads_dimension_group_attributes_onto_matching_series() -> None:
    """``dimensionGroupAttributes`` are resolved and attached to every matching series.

    The INDICATOR-only group (``":0:"``) is a wildcard on COUNTRY, so it attaches
    to both series; the COUNTRY+INDICATOR group (``"0:0:"``) attaches only to USA.
    Coded slots resolve to the code id; inline slots (plain string and localized
    text) resolve to a display string.
    """
    series = _series_by_country(_parse_fixture().data[0])

    usa_attrib = _attr_display(series["USA"][0].attrib)
    assert usa_attrib["NA_STO"] == "B1GQ"  # coded index 1 (not 0), INDICATOR group
    assert usa_attrib["SERIES_NAME"] == "Gross Domestic Product"  # localized, INDICATOR group
    assert usa_attrib["BASE_YEAR"] == "2015"  # inline string, COUNTRY+INDICATOR group

    fra_attrib = _attr_display(series["FRA"][0].attrib)
    assert fra_attrib["NA_STO"] == "B1GQ"  # wildcard group reaches FRA too
    assert fra_attrib["SERIES_NAME"] == "Gross Domestic Product"
    assert "BASE_YEAR" not in fra_attrib  # COUNTRY+INDICATOR group is USA-specific


def test_reader_parses_observation_level_attributes() -> None:
    """Observation-level attributes resolve per observation; null slots are skipped."""
    series = _series_by_country(_parse_fixture().data[0])

    usa_obs = _obs_by_time(series["USA"][1])
    assert usa_obs["2020"].value == "100"
    # coded attributes resolve via index 0; OBS_NOTE is a raw string in the data array
    assert _attr_display(usa_obs["2020"].attached_attribute) == {
        "OBS_STATUS": "A",
        "OBS_CONF": "F",
        "OBS_NOTE": "flash",
    }
    # OBS_STATUS resolves via coded index 1 (not 0); OBS_CONF omitted (null slot)
    assert _attr_display(usa_obs["2021"].attached_attribute) == {
        "OBS_STATUS": "H",
        "OBS_NOTE": "revision",
    }

    # raw strings are still resolved when coded attributes are null
    fra_obs = _obs_by_time(series["FRA"][1])
    assert _attr_display(fra_obs["2020"].attached_attribute) == {"OBS_NOTE": "preliminary"}
