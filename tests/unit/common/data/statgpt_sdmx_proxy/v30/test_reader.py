"""Tests for StatGptSdmxProxyDataReader with real SDMX-JSON 2.0.0 payloads.

Three fixture files exercise different structural patterns of the standard
SDMX-JSON 2.0.0 format:

* ``stub_eer.json`` — series-level dimensions + series/observation attributes,
  two series with different frequencies, monthly and daily time periods.

* ``imf_weo.json`` — large sparse series attribute arrays (many ``null``
  entries), observation time periods in annual format, ``Replace`` action.

* ``imf_weo_dataset_attrs.json`` — dataset-level ``attributes`` array with
  coded and empty-list values; exercises ``_make_dataset_level_attrs`` and the
  ``_dataset_level_attribute_map_from_data_message`` helper, including implicit
  single-value inference for attributes omitted from the array.
"""

import io
from pathlib import Path

import pytest
from sdmx.message import DataMessage
from sdmx.model.v21 import ActionType, AttributeValue, Code

from statgpt.common.data.statgpt_sdmx_proxy.v30.reader import StatGptSdmxProxyDataReader
from statgpt.common.data.statgpt_sdmx_proxy.v30.sdmx_client import (
    _dataset_level_attribute_map_from_data_message,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _parse(filename: str) -> DataMessage:
    content = io.BytesIO((_FIXTURES / filename).read_bytes())
    content.default_size = -1
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
        assert msg.header.prepared.year == 2026

    def test_header_sender_id(self, msg: DataMessage) -> None:
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
        assert len(msg.observation_dimension) == 1
        assert msg.observation_dimension[0].id == "TIME_PERIOD"

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
        assert len(msg.observation_dimension) == 1
        assert msg.observation_dimension[0].id == "TIME_PERIOD"

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
