"""The opt-in conversion of the values a dataset stores as fractions of one to percentages: its
config, the conversion `Sdmx21DataSet` runs as the data arrives, and the rounding that follows it.
"""

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from statgpt.common.data.base import DataResponseStatus
from statgpt.common.data.sdmx.common import SdmxDataSetConfig
from statgpt.common.data.sdmx.v21.dataset import Sdmx21DataResponse, Sdmx21DataSet
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus

_TIME = "TIME_PERIOD"
_TYPE = "TYPE"
_FACTOR = "FACTOR"
_UNIT = "UNIT"
_FRACTION = "FRACTION"
_SHARE = "SHARE"
_LEVEL = "LEVEL"


def _config(**details: Any) -> SdmxDataSetConfig:
    return SdmxDataSetConfig.model_validate(
        {
            "urn": {"agencyId": "AGENCY", "resourceId": "FLOW"},
            "dimensions": {
                "SERIES": {"dimensionType": "INDICATOR"},
                "FREQ": {"dimensionType": "NON_INDICATOR", "subtype": "FREQUENCY"},
                _TIME: {"dimensionType": "TIME_PERIOD"},
            },
            **details,
        }
    )


def _converting_config(**conversion: Any) -> SdmxDataSetConfig:
    return _config(
        includeAttributes=[_FACTOR, _TYPE, _UNIT],
        fractionToPercent={
            "attribute": _TYPE,
            "codes": [_FRACTION, _SHARE],
            "unitAttribute": _UNIT,
            **conversion,
        },
    )


class TestConfig:
    def test_conversion_is_off_by_default(self) -> None:
        assert _config().fraction_to_percent is None

    def test_conversion_is_read_from_the_dataset_details(self) -> None:
        conversion = _converting_config().fraction_to_percent

        assert conversion is not None
        assert (conversion.attribute, conversion.codes) == (_TYPE, [_FRACTION, _SHARE])
        assert (conversion.unit_attribute, conversion.unit) == (_UNIT, "%")

    def test_the_unit_can_be_set(self) -> None:
        conversion = _converting_config(unit="PC").fraction_to_percent

        assert conversion is not None
        assert conversion.unit == "PC"

    @pytest.mark.parametrize("include_attributes", [None, [_FACTOR], [_TYPE], [_UNIT]])
    def test_both_attributes_must_be_included(self, include_attributes: list[str] | None) -> None:
        with pytest.raises(ValidationError, match="must be listed in include_attributes"):
            _config(
                includeAttributes=include_attributes,
                fractionToPercent={
                    "attribute": _TYPE,
                    "codes": [_FRACTION],
                    "unitAttribute": _UNIT,
                },
            )

    def test_at_least_one_code_is_required(self) -> None:
        with pytest.raises(ValidationError, match="codes"):
            _converting_config(codes=[])

    def test_the_unit_attribute_is_required(self) -> None:
        with pytest.raises(ValidationError, match="unitAttribute"):
            _config(
                includeAttributes=[_TYPE],
                fractionToPercent={"attribute": _TYPE, "codes": [_FRACTION]},
            )

    def test_the_unit_attribute_must_differ_from_the_attribute(self) -> None:
        with pytest.raises(ValidationError, match="must differ from attribute"):
            _converting_config(unitAttribute=_TYPE)


# ~~~ The conversion ~~~


def _dataset(config: SdmxDataSetConfig) -> Sdmx21DataSet:
    """A dataset stub carrying only what the parsing of the data message reads."""
    dataset = Sdmx21DataSet.__new__(Sdmx21DataSet)
    dataset._config = config
    dataset._short_urn = "AGENCY:FLOW(1.0)"
    return dataset


def _source_frame(
    rows: list[tuple[str, str, float]], units: list[str | None] | None = None
) -> pd.DataFrame:
    """The shape `sdmx.to_pandas` gives with attributes: dimensions in the index, the observation
    value and one column per attribute, each cell holding an `AttributeValue`-like object or `None`.
    Without `units`, the source leaves the unit attribute out."""
    index = pd.MultiIndex.from_tuples(
        [(series, "2024") for series, _, _ in rows], names=["SERIES", _TIME]
    )
    columns: dict[str, list[Any]] = {
        "value": [value for _, _, value in rows],
        _FACTOR: [SimpleNamespace(value="0")] * len(rows),
        _TYPE: [SimpleNamespace(value=type_) for _, type_, _ in rows],
    }
    if units is not None:
        columns[_UNIT] = [None if unit is None else SimpleNamespace(value=unit) for unit in units]
    return pd.DataFrame(columns, index=index)


def _parse(dataset: Sdmx21DataSet, source: pd.DataFrame) -> pd.DataFrame:
    """Parse a data message as `Sdmx21DataSet.query` does, `sdmx.to_pandas` giving `source`."""
    with patch.object(dataset, "_data_msg_to_dataframe_sync", return_value=source):
        return dataset._parse_data_msg_sync(Mock())


def _parse_rows(dataset: Sdmx21DataSet, rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return _parse(dataset, _source_frame(rows))


def _values(df: pd.DataFrame) -> dict[str, float]:
    return dict(zip(df.index.get_level_values("SERIES"), df["value"]))


def _units(df: pd.DataFrame) -> dict[str, str]:
    return dict(zip(df.index.get_level_values("SERIES"), df.index.get_level_values(_UNIT)))


class TestConversion:
    def test_fraction_series_are_converted_and_the_other_series_kept(self) -> None:
        df = _parse_rows(
            _dataset(_converting_config()),
            [
                ("GDP_GROWTH", _FRACTION, 0.028),
                ("EXPORTS_SHARE", _SHARE, 0.1188),
                ("CPI", _LEVEL, 132.3),
            ],
        )

        assert _values(df) == {"GDP_GROWTH": 2.8, "EXPORTS_SHARE": 11.88, "CPI": 132.3}

    @pytest.mark.parametrize(
        ("fraction", "percent"),
        [
            # A float multiplication would give `2.8000000000000003` and `3.6999999999999997`.
            (0.028, 2.8),
            (0.037, 3.7),
            (0.04167, 4.167),
            # A fraction above 1 is still a fraction.
            (1.028, 102.8),
            (-0.0356, -3.56),
            (0.0, 0.0),
        ],
    )
    def test_the_decimal_point_is_shifted_exactly(self, fraction: float, percent: float) -> None:
        df = _parse_rows(_dataset(_converting_config()), [("S", _FRACTION, fraction)])

        assert df["value"].tolist() == [percent]

    def test_a_missing_value_stays_missing(self) -> None:
        df = _parse_rows(_dataset(_converting_config()), [("S", _FRACTION, float("nan"))])

        assert pd.isna(df["value"].iloc[0])

    def test_the_attribute_is_kept_in_the_table(self) -> None:
        df = _parse_rows(_dataset(_converting_config()), [("S", _FRACTION, 0.028)])

        assert df.index.get_level_values(_TYPE).tolist() == [_FRACTION]

    def test_the_unit_is_set_on_the_converted_series_only(self) -> None:
        df = _parse(
            _dataset(_converting_config()),
            _source_frame(
                [("GDP_GROWTH", _FRACTION, 0.028), ("EXPORTS", _LEVEL, 5.0)], units=[None, "USD"]
            ),
        )

        assert _units(df) == {"GDP_GROWTH": "%", "EXPORTS": "USD"}

    def test_the_configured_unit_is_set(self) -> None:
        df = _parse_rows(_dataset(_converting_config(unit="PC")), [("S", _FRACTION, 0.028)])

        assert _units(df) == {"S": "PC"}

    def test_the_unit_is_added_when_the_source_leaves_it_out(self) -> None:
        # Empty rather than missing for the other series, which the charts would otherwise drop.
        df = _parse_rows(
            _dataset(_converting_config()),
            [("GDP_GROWTH", _FRACTION, 0.028), ("CPI", _LEVEL, 132.3)],
        )

        assert _units(df) == {"GDP_GROWTH": "%", "CPI": ""}

    def test_the_unit_is_added_without_a_fraction_series(self) -> None:
        # So every response of the dataset has the same index levels, e.g. to be merged.
        df = _parse_rows(_dataset(_converting_config()), [("CPI", _LEVEL, 132.3)])

        assert _values(df) == {"CPI": 132.3}
        assert _units(df) == {"CPI": ""}

    def test_nothing_is_converted_without_the_config(self) -> None:
        df = _parse_rows(
            _dataset(_config(includeAttributes=[_FACTOR, _TYPE])), [("S", _FRACTION, 0.028)]
        )

        assert df["value"].tolist() == [0.028]
        assert _UNIT not in df.index.names

    def test_nothing_is_converted_when_the_response_lacks_the_attribute(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        source = _source_frame([("S", _FRACTION, 0.028)]).drop(columns=[_TYPE])

        with caplog.at_level(logging.WARNING):
            df = _parse(_dataset(_converting_config()), source)

        assert df["value"].tolist() == [0.028]
        assert "lacks the 'TYPE' attribute" in caplog.text

    def test_the_source_frame_is_left_untouched(self) -> None:
        dataset = _dataset(_converting_config())
        parsed = dataset._include_attributes_sync(_source_frame([("S", _FRACTION, 0.028)]))

        dataset._convert_fractions_to_percent_sync(parsed)

        assert parsed["value"].tolist() == [0.028]
        assert _UNIT not in parsed.index.names


# ~~~ Conversion, then rounding ~~~


def test_rounding_applies_to_the_percentage() -> None:
    # Rounded first, the fraction `0.04167` would keep only its first significant digit with 1
    # decimal place, `0.04`, and read 4%; converted first, it reads `4.2`, the precision asked for.
    df = _parse_rows(_dataset(_converting_config()), [("S", _FRACTION, 0.04167)])
    dataset = Mock()
    dataset.dimensions.return_value = [
        SimpleNamespace(entity_id="SERIES", name="Series"),
        SimpleNamespace(entity_id=_TIME, name="Time period"),
    ]
    dataset.attributes.return_value = [
        SimpleNamespace(entity_id=_FACTOR, name="Factor"),
        SimpleNamespace(entity_id=_TYPE, name="Type"),
        SimpleNamespace(entity_id=_UNIT, name="Unit"),
    ]
    dataset.get_time_dimension.return_value = SimpleNamespace(entity_id=_TIME)
    dataset.map_component_values_id_2_name.return_value = None
    response = Sdmx21DataResponse(
        dataset=dataset,
        sdmx_query=Mock(),
        df=df,
        url=None,
        status=DataResponseStatus(
            request_status=DataRequestStatus.SUCCESS, parsing_status=DataParsingStatus.SUCCESS
        ),
    )

    assert response.rounded_csv_dataframe(1)["value"].tolist() == [4.2]
    visual_row = response.rounded_visual_dataframe(1).iloc[0]
    assert (visual_row["2024"], visual_row[_UNIT]) == (4.2, "%")
