"""The opt-in conversion of the values a dataset stores as fractions of one to percentages: its
config, the conversion of the values the LLM reads, the rounding that follows it, and the user's
attachments that keep the fractions.
"""

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import ValidationError

from statgpt.common.data.base import DataResponseStatus
from statgpt.common.data.sdmx.common import SdmxDataSetConfig
from statgpt.common.data.sdmx.v21.dataset import Sdmx21DataResponse
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


def _converting_config(decimal_places: int | None = None, **conversion: Any) -> SdmxDataSetConfig:
    return _config(
        includeAttributes=[_FACTOR, _TYPE, _UNIT],
        fractionToPercent={
            "component": _TYPE,
            "codes": [_FRACTION, _SHARE],
            "unitAttribute": _UNIT,
            **conversion,
        },
        decimalPlaces=decimal_places,
    )


def _converting_config_with(
    include_attributes: list[str] | None, component: str
) -> SdmxDataSetConfig:
    return _config(
        includeAttributes=include_attributes,
        fractionToPercent={"component": component, "codes": ["X"], "unitAttribute": _UNIT},
    )


class TestConfig:
    def test_conversion_is_off_by_default(self) -> None:
        assert _config().fraction_to_percent is None

    def test_conversion_is_read_from_the_dataset_details(self) -> None:
        conversion = _converting_config().fraction_to_percent

        assert conversion is not None
        assert (conversion.component, conversion.codes) == (_TYPE, [_FRACTION, _SHARE])
        assert (conversion.unit_attribute, conversion.unit) == (_UNIT, "%")

    def test_the_unit_can_be_set(self) -> None:
        conversion = _converting_config(unit="PC").fraction_to_percent

        assert conversion is not None
        assert conversion.unit == "PC"

    @pytest.mark.parametrize("include_attributes", [None, [_FACTOR, _UNIT]])
    def test_an_attribute_component_must_be_included(
        self, include_attributes: list[str] | None
    ) -> None:
        with pytest.raises(ValidationError, match="must be a dimension of the dataset or listed"):
            _converting_config_with(include_attributes, component=_TYPE)

    def test_a_dimension_component_needs_no_including(self) -> None:
        conversion = _converting_config_with([_UNIT], component="SERIES").fraction_to_percent

        assert conversion is not None
        assert conversion.component == "SERIES"

    def test_an_unknown_component_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be a dimension of the dataset or listed"):
            _converting_config_with([_TYPE, _UNIT], component="REF_AREA")

    @pytest.mark.parametrize("include_attributes", [None, [_TYPE], [_FACTOR, _TYPE]])
    def test_the_unit_attribute_must_be_included(
        self, include_attributes: list[str] | None
    ) -> None:
        with pytest.raises(ValidationError, match="unit_attribute='UNIT' must be listed"):
            _converting_config_with(include_attributes, component="SERIES")

    def test_at_least_one_code_is_required(self) -> None:
        with pytest.raises(ValidationError, match="codes"):
            _converting_config(codes=[])

    def test_the_unit_attribute_is_required(self) -> None:
        with pytest.raises(ValidationError, match="unitAttribute"):
            _config(
                includeAttributes=[_TYPE],
                fractionToPercent={"component": _TYPE, "codes": [_FRACTION]},
            )

    def test_the_unit_attribute_must_differ_from_the_component(self) -> None:
        with pytest.raises(ValidationError, match="must differ from component"):
            _converting_config(unitAttribute=_TYPE)


# ~~~ The conversion ~~~


def _frame(
    rows: list[tuple[str, str, float]], units: list[str | None] | None = None
) -> pd.DataFrame:
    """The shape `Sdmx21DataSet.query` gives with the included attributes: the dimensions and the
    attributes in the index, the observation values in the single `value` column. Without
    `units`, the source leaves the unit attribute out."""
    names = ["SERIES", _TIME, _FACTOR, _TYPE]
    keys: list[tuple[Any, ...]] = [(series, "2024", "0", type_) for series, type_, _ in rows]
    if units is not None:
        names.append(_UNIT)
        keys = [(*key, unit) for key, unit in zip(keys, units)]
    return pd.DataFrame(
        {"value": [value for _, _, value in rows]},
        index=pd.MultiIndex.from_tuples(keys, names=names),
    )


def _response(config: SdmxDataSetConfig, df: pd.DataFrame) -> Sdmx21DataResponse:
    dataset = Mock()
    dataset.entity_id = "ds1"
    dataset.config = config
    dataset.short_urn = "AGENCY:FLOW(1.0)"
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
    return Sdmx21DataResponse(
        dataset=dataset,
        sdmx_query=Mock(),
        df=df,
        url=None,
        status=DataResponseStatus(
            request_status=DataRequestStatus.SUCCESS, parsing_status=DataParsingStatus.SUCCESS
        ),
    )


def _llm_rows(
    rows: list[tuple[str, str, float]], config: SdmxDataSetConfig | None = None
) -> pd.DataFrame:
    """The observations the LLM reads for `rows`."""
    return _response(config or _converting_config(), _frame(rows)).llm_csv_dataframe


def _values(df: pd.DataFrame) -> dict[str, float]:
    return dict(zip(df["SERIES"], df["value"]))


def _units(df: pd.DataFrame) -> dict[str, str]:
    return dict(zip(df["SERIES"], df[_UNIT]))


class TestConversion:
    def test_fraction_series_are_converted_and_the_other_series_kept(self) -> None:
        df = _llm_rows(
            [
                ("GDP_GROWTH", _FRACTION, 0.028),
                ("EXPORTS_SHARE", _SHARE, 0.1188),
                ("CPI", _LEVEL, 132.3),
            ]
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
        df = _llm_rows([("S", _FRACTION, fraction)])

        assert df["value"].tolist() == [percent]

    def test_a_missing_value_stays_missing(self) -> None:
        df = _llm_rows([("S", _FRACTION, float("nan"))])

        assert pd.isna(df["value"].iloc[0])

    def test_series_are_told_apart_by_a_dimension_too(self) -> None:
        # The attribute codes need not match how the values are stored: only the listed series
        # are converted, whatever their type.
        config = _converting_config(component="SERIES", codes=["GDP_GROWTH"])
        source = _frame(
            [("GDP_GROWTH", _LEVEL, 0.028), ("SPREAD_BP", _FRACTION, 100.0)], units=[None, "BP"]
        )

        df = _response(config, source).llm_csv_dataframe

        assert _values(df) == {"GDP_GROWTH": 2.8, "SPREAD_BP": 100.0}
        assert _units(df) == {"GDP_GROWTH": "%", "SPREAD_BP": "BP"}

    def test_the_attribute_is_kept_in_the_table(self) -> None:
        df = _llm_rows([("S", _FRACTION, 0.028)])

        assert df[_TYPE].tolist() == [_FRACTION]

    def test_the_unit_is_set_on_the_converted_series_only(self) -> None:
        source = _frame(
            [("GDP_GROWTH", _FRACTION, 0.028), ("EXPORTS", _LEVEL, 5.0)], units=[None, "USD"]
        )

        df = _response(_converting_config(), source).llm_csv_dataframe

        assert _units(df) == {"GDP_GROWTH": "%", "EXPORTS": "USD"}

    def test_the_configured_unit_is_set(self) -> None:
        df = _llm_rows([("S", _FRACTION, 0.028)], _converting_config(unit="PC"))

        assert _units(df) == {"S": "PC"}

    def test_the_unit_is_added_when_the_source_leaves_it_out(self) -> None:
        # Empty rather than missing for the other series, which a grouping would drop.
        df = _llm_rows([("GDP_GROWTH", _FRACTION, 0.028), ("CPI", _LEVEL, 132.3)])

        assert _units(df) == {"GDP_GROWTH": "%", "CPI": ""}

    def test_the_unit_is_added_without_a_fraction_series(self) -> None:
        # So the LLM reads the same columns in every response of the dataset.
        df = _llm_rows([("CPI", _LEVEL, 132.3)])

        assert _values(df) == {"CPI": 132.3}
        assert _units(df) == {"CPI": ""}

    def test_the_time_periods_table_is_converted_too(self) -> None:
        response = _response(
            _converting_config(), _frame([("GDP_GROWTH", _FRACTION, 0.028), ("CPI", _LEVEL, 5.0)])
        )

        rows = response.llm_visual_dataframe.set_index("SERIES")

        assert rows.loc["GDP_GROWTH", ["2024", _UNIT]].tolist() == [2.8, "%"]
        assert rows.loc["CPI", ["2024", _UNIT]].tolist() == [5.0, ""]

    def test_a_merged_response_is_converted(self) -> None:
        # The responses are merged as the source gives them, and converted once merged.
        config = _converting_config()
        first = _response(config, _frame([("GDP_GROWTH", _FRACTION, 0.028)]))
        second = _response(config, _frame([("CPI", _LEVEL, 132.3)]))

        df = first.merge(second).llm_csv_dataframe

        assert _values(df) == {"GDP_GROWTH": 2.8, "CPI": 132.3}
        assert _units(df) == {"GDP_GROWTH": "%", "CPI": ""}

    def test_nothing_is_converted_without_the_config(self) -> None:
        df = _llm_rows([("S", _FRACTION, 0.028)], _config(includeAttributes=[_FACTOR, _TYPE]))

        assert df["value"].tolist() == [0.028]
        assert _UNIT not in df.columns

    def test_nothing_is_converted_when_the_response_lacks_the_component(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        source = _frame([("S", _FRACTION, 0.028)]).droplevel(_TYPE)

        with caplog.at_level(logging.WARNING):
            df = _response(_converting_config(), source).llm_csv_dataframe

        assert df["value"].tolist() == [0.028]
        assert "lacks the 'TYPE' component" in caplog.text


# ~~~ What the user gets ~~~


class TestUserAttachments:
    @pytest.fixture
    def response(self) -> Sdmx21DataResponse:
        response = _response(
            _converting_config(decimal_places=1),
            _frame([("GDP_GROWTH", _FRACTION, 0.04167), ("CPI", _LEVEL, 132.345)]),
        )
        # Read first, so a conversion done in place would show in the user's frames.
        _ = response.llm_csv_dataframe, response.llm_visual_dataframe
        return response

    def test_the_source_frame_is_left_untouched(self, response: Sdmx21DataResponse) -> None:
        assert _values(response.df.reset_index()) == {"GDP_GROWTH": 0.04167, "CPI": 132.345}
        assert _UNIT not in response.df.index.names

    def test_the_csv_download_keeps_the_fractions_at_full_precision(
        self, response: Sdmx21DataResponse
    ) -> None:
        for df in (response.csv_dataframe, response.dataframe):
            assert _values(df) == {"GDP_GROWTH": 0.04167, "CPI": 132.345}
            assert _UNIT not in df.columns

    def test_the_grid_keeps_the_fractions_at_full_precision(
        self, response: Sdmx21DataResponse
    ) -> None:
        rows = response.visual_dataframe.set_index("SERIES")
        assert rows["2024"].to_dict() == {"GDP_GROWTH": 0.04167, "CPI": 132.345}
        assert _UNIT not in rows.columns

        grid = response.plotly_grid
        assert grid is not None
        table = grid.data[0]
        columns = dict(zip(table.header.values, table.cells.values))
        assert dict(zip(columns["SERIES"], columns["2024"])) == {
            "GDP_GROWTH": 0.04167,
            "CPI": 132.345,
        }


# ~~~ Conversion, then rounding ~~~


def test_rounding_applies_to_the_percentage() -> None:
    # Rounded first, the fraction `0.04167` would keep only its first significant digit with 1
    # decimal place, `0.04`, and read 4%; converted first, it reads `4.2`, the precision asked for.
    response = _response(_converting_config(decimal_places=1), _frame([("S", _FRACTION, 0.04167)]))

    assert response.llm_csv_dataframe["value"].tolist() == [4.2]
    visual_row = response.llm_visual_dataframe.iloc[0]
    assert (visual_row["2024"], visual_row[_UNIT]) == (4.2, "%")
