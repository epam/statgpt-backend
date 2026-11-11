"""The rounding of the values the LLM reads, set by `decimalPlaces` in the dataset config, next to
the full-precision frames the user's attachments are built from."""

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
# A numeric, non-integer attribute: it qualifies the values, so it must never be rounded.
_FACTOR = "ADJUSTMENT_FACTOR"


def _config(**details: Any) -> SdmxDataSetConfig:
    return SdmxDataSetConfig.model_validate(
        {
            "urn": {"agencyId": "AGENCY", "resourceId": "FLOW"},
            "dimensions": {
                "INDICATOR": {"dimensionType": "INDICATOR"},
                "FREQ": {"dimensionType": "NON_INDICATOR", "subtype": "FREQUENCY"},
                _TIME: {"dimensionType": "TIME_PERIOD"},
            },
            **details,
        }
    )


class TestConfig:
    def test_rounding_is_off_by_default(self) -> None:
        assert _config().decimal_places is None

    def test_rounding_is_read_from_the_dataset_details(self) -> None:
        assert _config(decimalPlaces=3).decimal_places == 3

    @pytest.mark.parametrize("value", [-1, 16, 2.5])
    def test_decimal_places_out_of_range_are_rejected(self, value: float) -> None:
        with pytest.raises(ValidationError, match="decimalPlaces"):
            _config(decimalPlaces=value)


def _dataset(decimal_places: int | None) -> Mock:
    dataset = Mock()
    dataset.entity_id = "ds1"
    dataset.config = SimpleNamespace(decimal_places=decimal_places, fraction_to_percent=None)
    dataset.dimensions.return_value = [
        SimpleNamespace(entity_id="REF_AREA", name="Reference area"),
        SimpleNamespace(entity_id=_TIME, name="Time period"),
    ]
    dataset.attributes.return_value = [SimpleNamespace(entity_id=_FACTOR, name="Factor")]
    dataset.get_time_dimension.return_value = SimpleNamespace(entity_id=_TIME)
    dataset.map_component_values_id_2_name.side_effect = lambda value_ids, component_id: (
        {"FR": "France"} if component_id == "REF_AREA" else None
    )
    return dataset


def _response(
    values: list[float], periods: list[str], decimal_places: int | None = 3
) -> Sdmx21DataResponse:
    # The shape `Sdmx21DataSet.query` produces: dimensions and attributes in the index, the
    # observation values in the single `value` column.
    index = pd.MultiIndex.from_tuples(
        [("FR", period, 1.23456) for period in periods], names=["REF_AREA", _TIME, _FACTOR]
    )
    return Sdmx21DataResponse(
        dataset=_dataset(decimal_places),
        sdmx_query=Mock(),
        df=pd.DataFrame({"value": values}, index=index),
        url=None,
        status=DataResponseStatus(
            request_status=DataRequestStatus.SUCCESS, parsing_status=DataParsingStatus.SUCCESS
        ),
    )


@pytest.fixture
def response() -> Sdmx21DataResponse:
    return _response([0.9431918, 3.521599, float("nan")], ["2020", "2021", "2022"])


def test_llm_csv_dataframe_rounds_the_values(response: Sdmx21DataResponse) -> None:
    values = response.llm_csv_dataframe["value"].tolist()

    assert values[:2] == [0.943, 3.522]
    assert pd.isna(values[2])


def test_llm_visual_dataframe_rounds_the_time_period_columns(
    response: Sdmx21DataResponse,
) -> None:
    row = response.llm_visual_dataframe.iloc[0]

    assert (row["2020"], row["2021"]) == (0.943, 3.522)
    assert pd.isna(row["2022"])


def test_the_user_frames_keep_full_precision(response: Sdmx21DataResponse) -> None:
    # Read after the LLM's ones, so a rounding done in place would show here.
    _ = response.llm_csv_dataframe, response.llm_visual_dataframe

    assert response.csv_dataframe["value"].tolist()[:2] == [0.9431918, 3.521599]
    assert response.dataframe["value"].tolist()[:2] == [0.9431918, 3.521599]
    assert response.visual_dataframe.iloc[0]["2020"] == 0.9431918


def test_dimensions_and_attributes_are_not_rounded(response: Sdmx21DataResponse) -> None:
    for df in (response.llm_csv_dataframe, response.llm_visual_dataframe):
        assert df[_FACTOR].tolist() == [1.23456] * len(df)
        assert set(df["REF_AREA_Name"]) == {"France"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Too small for 3 decimal places: the first significant digit is kept instead of a 0.
        (0.0004, 0.0004),
        (0.00049, 0.0005),
        (-0.0004, -0.0004),
        (1.234e-7, 1e-7),
        # A tie rounds to 0 with 3 decimal places, so it keeps its digit too.
        (0.0005, 0.0005),
        # Big enough for 3 decimal places: plain rounding.
        (0.0006, 0.001),
        (0.0, 0.0),
    ],
)
def test_small_values_keep_their_first_significant_digit(value: float, expected: float) -> None:
    response = _response([value], ["2020"])

    assert response.llm_csv_dataframe["value"].tolist() == [expected]
    assert response.llm_visual_dataframe.iloc[0]["2020"] == expected


def test_a_negative_zero_is_written_as_zero() -> None:
    response = _response([-0.0], ["2020"])

    csv = response.llm_csv_dataframe.to_csv(index=False)

    assert "-0.0" not in csv


def test_zero_decimal_places_still_keeps_small_values() -> None:
    response = _response([2.6, 0.4], ["2020", "2021"], decimal_places=0)

    assert response.llm_csv_dataframe["value"].tolist() == [3.0, 0.4]


def test_no_decimal_places_keeps_full_precision() -> None:
    response = _response([0.9431918, 3.521599], ["2020", "2021"], decimal_places=None)

    assert response.llm_csv_dataframe["value"].tolist() == [0.9431918, 3.521599]
    assert response.llm_visual_dataframe.iloc[0]["2020"] == 0.9431918


def test_plotly_grid_shows_full_precision(response: Sdmx21DataResponse) -> None:
    grid = response.plotly_grid
    assert grid is not None
    table = grid.data[0]
    columns = dict(zip(table.header.values, table.cells.values))

    assert (columns["2020"][0], columns["2021"][0]) == (0.9431918, 3.521599)
    assert columns[_FACTOR][0] == 1.23456
