"""The rounded dataframes of `Sdmx21DataResponse`: what the model reads and the UI grid shows, next
to the full-precision frames the downloadable files are built from."""

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from statgpt.common.data.base import DataResponseStatus
from statgpt.common.data.sdmx.v21.dataset import Sdmx21DataResponse
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus

_TIME = "TIME_PERIOD"
# A numeric, non-integer attribute: it qualifies the values, so it must never be rounded.
_FACTOR = "ADJUSTMENT_FACTOR"


def _dataset() -> Mock:
    dataset = Mock()
    dataset.entity_id = "ds1"
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
    values: list[float], periods: list[str], dataset: Mock | None = None
) -> Sdmx21DataResponse:
    # The shape `Sdmx21DataSet.query` produces: dimensions and attributes in the index, the
    # observation values in the single `value` column.
    index = pd.MultiIndex.from_tuples(
        [("FR", period, 1.23456) for period in periods], names=["REF_AREA", _TIME, _FACTOR]
    )
    return Sdmx21DataResponse(
        dataset=dataset or _dataset(),
        sdmx_query=Mock(),
        df=pd.DataFrame({"value": values}, index=index),
        url=None,
        status=DataResponseStatus(
            request_status=DataRequestStatus.SUCCESS, parsing_status=DataParsingStatus.SUCCESS
        ),
    )


@pytest.fixture
def response() -> Sdmx21DataResponse:
    response = _response([0.9431918, 3.521599, float("nan")], ["2020", "2021", "2022"])
    response.value_decimal_places = 3
    return response


def test_rounded_csv_dataframe_rounds_the_values(response: Sdmx21DataResponse) -> None:
    values = response.rounded_csv_dataframe["value"].tolist()

    assert values[:2] == [0.943, 3.522]
    assert pd.isna(values[2])


def test_rounded_visual_dataframe_rounds_the_time_period_columns(
    response: Sdmx21DataResponse,
) -> None:
    row = response.rounded_visual_dataframe.iloc[0]

    assert (row["2020"], row["2021"]) == (0.943, 3.522)
    assert pd.isna(row["2022"])


def test_full_precision_dataframes_are_left_untouched(response: Sdmx21DataResponse) -> None:
    # Read after the rounded ones, so a rounding done in place would show here.
    _ = response.rounded_csv_dataframe, response.rounded_visual_dataframe

    assert response.csv_dataframe["value"].tolist()[:2] == [0.9431918, 3.521599]
    assert response.visual_dataframe.iloc[0]["2020"] == 0.9431918


def test_dimensions_and_attributes_are_not_rounded(response: Sdmx21DataResponse) -> None:
    for df in (response.rounded_csv_dataframe, response.rounded_visual_dataframe):
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
    response.value_decimal_places = 3

    assert response.rounded_csv_dataframe["value"].tolist() == [expected]
    assert response.rounded_visual_dataframe.iloc[0]["2020"] == expected


def test_a_negative_zero_is_written_as_zero() -> None:
    response = _response([-0.0], ["2020"])
    response.value_decimal_places = 3

    csv = response.rounded_csv_dataframe.to_csv(index=False)

    assert "-0.0" not in csv


def test_zero_decimal_places_still_keeps_small_values() -> None:
    response = _response([2.6, 0.4], ["2020", "2021"])
    response.value_decimal_places = 0

    assert response.rounded_csv_dataframe["value"].tolist() == [3.0, 0.4]


def _grid_columns(response: Sdmx21DataResponse) -> dict[str, tuple]:
    grid = response.plotly_grid
    assert grid is not None
    table = grid.data[0]
    return dict(zip(table.header.values, table.cells.values))


def test_plotly_grid_shows_the_values_the_model_reads(response: Sdmx21DataResponse) -> None:
    columns = _grid_columns(response)

    assert (columns["2020"][0], columns["2021"][0]) == (0.943, 3.522)
    assert columns[_FACTOR][0] == 1.23456


def test_plotly_grid_keeps_full_precision_when_rounding_is_off() -> None:
    response = _response([0.9431918], ["2020"])

    assert _grid_columns(response)["2020"][0] == 0.9431918


def test_rounding_is_off_until_the_pipeline_sets_it() -> None:
    response = _response([0.9431918], ["2020"])

    assert response.value_decimal_places is None
    assert response.rounded_csv_dataframe["value"].tolist() == [0.9431918]


def test_merge_keeps_the_rounding() -> None:
    dataset = _dataset()
    first = _response([0.9431918], ["2020"], dataset)
    second = _response([3.521599], ["2021"], dataset)
    first.value_decimal_places = second.value_decimal_places = 3

    merged = first.merge(second)

    assert merged.value_decimal_places == 3
    assert merged.rounded_csv_dataframe["value"].tolist() == [0.943, 3.522]
