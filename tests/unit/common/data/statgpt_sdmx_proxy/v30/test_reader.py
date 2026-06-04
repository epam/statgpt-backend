"""Tests for :class:`StatGptSdmxProxyDataReader`.

``data_all_attribute_levels.json`` is a small hand-crafted proxy SDMX-JSON data
message that exercises all four SDMX-JSON 2.0.0 attribute levels and every value
encoding the reader supports:

* ``dataSet`` — coded index, inline string, inline localized text, ``null``.
* ``dimensionGroup`` — group attributes carried in ``dimensionGroupAttributes``
  keyed by partial dimension keys; a wildcard (INDICATOR-only) group that matches
  both series and a specific (COUNTRY+INDICATOR) group that matches one.
* ``series`` — coded index and ``null`` skip.
* ``observation`` — coded index and ``null`` skip.

It has two series (``USA, GDP`` and ``FRA, GDP``) over two time periods, with
just enough values to cover each case without redundancy.
"""

import io
from pathlib import Path

from sdmx.message import DataMessage

from statgpt.common.data.statgpt_sdmx_proxy.v30.reader import StatGptSdmxProxyDataReader

FIXTURE = Path(__file__).parent / "data_all_attribute_levels.json"


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

    fra_key, _ = series["FRA"]
    fra_attrib = _attr_display(fra_key.attrib)
    assert fra_attrib["SCALE"] == "9"
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
    # both attributes resolve via coded index 0
    assert _attr_display(usa_obs["2020"].attached_attribute) == {"OBS_STATUS": "A", "OBS_CONF": "F"}
    # OBS_STATUS resolves via coded index 1 (not 0); OBS_CONF omitted (null slot)
    assert _attr_display(usa_obs["2021"].attached_attribute) == {"OBS_STATUS": "H"}

    # an observation with no attribute array carries no observation-level attributes
    fra_obs = _obs_by_time(series["FRA"][1])
    assert fra_obs["2020"].attached_attribute == {}
