"""Tests for :class:`SdmxJsonDataReader` on SDMX-JSON 1.0 messages (SDMX 2.1 APIs).

The SDMX-JSON 2.0.0 variants are covered in ``test_json_reader_2_0.py``.

``sdmx_json_1_0_attribute_levels.json`` has two series that differ only by ``INDICATOR``
and ``SCALE``.  The empty ``UNIT`` attribute sits between ``CURRENCY`` and ``SCALE``
so that discarding it would shift ``SCALE`` out of position.  ``NOTE`` is referenced by
the first series only, ``METHODOLOGY`` by none.
"""

import io
from pathlib import Path

import pytest
from sdmx.message import DataMessage
from sdmx.model.v21 import AttributeValue, DataSet, SeriesKey

from statgpt.common.data.sdmx.common.json_reader import SdmxJsonDataReader

FIXTURE = Path(__file__).parent / "sdmx_json_1_0_attribute_levels.json"


def _attr_ids(series_key: SeriesKey) -> dict[str, str]:
    return {attr_id: _value_id(av) for attr_id, av in series_key.attrib.items()}


def _value_id(av: AttributeValue) -> str:
    return getattr(av.value, "id", av.value)  # type: ignore[arg-type]


class TestSdmxJson10Reader:
    @pytest.fixture(scope="class")
    def msg(self) -> DataMessage:
        return SdmxJsonDataReader().convert(io.BytesIO(FIXTURE.read_bytes()))

    @pytest.fixture(scope="class")
    def dataset(self, msg: DataMessage) -> DataSet:
        return msg.data[0]

    @pytest.fixture(scope="class")
    def series(self, dataset: DataSet) -> dict[str, dict[str, str]]:
        return {sk.values["INDICATOR"].value: _attr_ids(sk) for sk in dataset.series}

    def test_reads_observations(self, dataset: DataSet) -> None:
        assert len(dataset.obs) == 4

    def test_dataset_attributes_are_attached(self, dataset: DataSet) -> None:
        assert _value_id(dataset.attrib["COMMENT"]) == "Test comment"

    def test_series_attributes_come_from_each_series(
        self, series: dict[str, dict[str, str]]
    ) -> None:
        assert series["IND_A"]["SCALE"] == "9"
        assert series["IND_B"]["SCALE"] == "6"
        assert series["IND_A"]["CURRENCY"] == series["IND_B"]["CURRENCY"] == "USD"

    def test_null_slot_is_not_resolved(self, series: dict[str, dict[str, str]]) -> None:
        assert "UNIT" not in series["IND_A"]
        assert series["IND_A"]["NOTE"] == "EST"
        assert "NOTE" not in series["IND_B"]

    def test_unreferenced_single_value_attribute_applies_to_every_series(
        self, series: dict[str, dict[str, str]]
    ) -> None:
        assert series["IND_A"]["METHODOLOGY"] == series["IND_B"]["METHODOLOGY"] == "SNA2008"
