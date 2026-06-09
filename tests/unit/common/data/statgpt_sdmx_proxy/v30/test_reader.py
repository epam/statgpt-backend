"""Tests for :class:`StatGptSdmxProxyDataReader`.

The reader normalises two SDMX-JSON 2.0.0 variants into identical models.  Two
small hand-crafted fixtures keep each variant isolated so a regression in one
cannot be masked by the other:

* ``data_all_attribute_levels.json`` — **proxy** format.  Covers the proxy-only
  branches: the ``dimensionGroup`` attribute category, uncoded dimension values
  using ``{"value": "..."}``, uncoded raw-string attributes in data arrays, and
  inline localized text.
* ``standard_attribute_levels.json`` — **standard** (non-proxy) format.  Covers
  the standard-only branches: the lowercase ``dataset`` level key, coded value
  objects carrying extra keys (``name``/``description``/``start``/``end``),
  implicit single-value dataSet-attribute inference, and empty coded-values
  lists resolving to ``None``.
"""

import io
import json
from pathlib import Path

import pytest
from sdmx.message import DataMessage
from sdmx.model.v21 import AllDimensions, AttributeValue, Code

from statgpt.common.data.statgpt_sdmx_proxy.v30.reader import StatGptSdmxProxyDataReader
from statgpt.common.data.statgpt_sdmx_proxy.v30.sdmx_client import (
    _dataset_level_attribute_map_from_data_message,
)

STANDARD_FIXTURE = Path(__file__).parent / "standard_attribute_levels.json"


def _parse_standard() -> DataMessage:
    return StatGptSdmxProxyDataReader().convert(io.BytesIO(STANDARD_FIXTURE.read_bytes()))


def _attr_code_id(av: AttributeValue) -> str | None:
    val = av.value
    return val.id if isinstance(val, Code) else None


class TestStandardFormatReader:
    """Standard (non-proxy) SDMX-JSON 2.0.0 variant.

    One series over two annual observations.  Each test asserts a single
    representative value per reader branch — covering every branch the variant
    exercises uniquely, without re-asserting the same code path field by field.
    """

    @pytest.fixture(scope="class")
    def msg(self) -> DataMessage:
        return _parse_standard()

    @pytest.fixture(scope="class")
    def dataset(self, msg: DataMessage):
        return msg.data[0]

    @pytest.fixture(scope="class")
    def series_key(self, dataset):
        return next(iter(dataset.series))

    def test_returns_single_dataset(self, msg: DataMessage) -> None:
        assert isinstance(msg, DataMessage)
        assert len(msg.data) == 1

    def test_lowercase_dataset_level_resolved(self, dataset) -> None:
        # lowercase ``dataset`` level key is normalised so dataSet attrs resolve
        assert _attr_code_id(dataset.attrib["UPDATE_DATE"]) == "2026-04-15T13:00:00Z"

    def test_coded_dimension_value(self, series_key) -> None:
        assert series_key.values["COUNTRY"].value == "USA"

    def test_value_object_extra_keys_filtered(self, series_key) -> None:
        # SCALE's value carries start/end/order; _code_from_value must drop them
        # (else TypeError) while keeping the code id.
        assert _attr_code_id(series_key.attrib["SCALE"]) == "9"

    def test_sparse_series_null_slot_skipped(self, series_key) -> None:
        assert "DECIMALS" not in series_key.attrib

    def test_dataset_attr_empty_coded_list_is_none(self, dataset) -> None:
        assert dataset.attrib["DOI"].value is None

    def test_dataset_attr_implicit_single_value(self, dataset) -> None:
        # PUBLISHER is beyond the data array; its single coded value is inferred.
        assert _attr_code_id(dataset.attrib["PUBLISHER"]) == "IMF"

    def test_dataset_attr_empty_beyond_array_absent(self, dataset) -> None:
        # AUTHOR is beyond the array with no coded values -> not emitted.
        assert "AUTHOR" not in dataset.attrib

    def test_dataset_attribute_map_helper(self, msg: DataMessage) -> None:
        attr_map = _dataset_level_attribute_map_from_data_message(msg)
        assert attr_map["UPDATE_DATE"] == "2026-04-15T13:00:00Z"  # Code -> id
        assert attr_map["DOI"] is None  # None -> None
        assert attr_map["PUBLISHER"] == "IMF"  # implicit inference


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
