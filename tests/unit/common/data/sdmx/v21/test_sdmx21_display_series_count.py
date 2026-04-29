"""Tests for ``Sdmx21DataResponse.get_display_series_count``."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from statgpt.common.data.base.dataset import DataResponseStatus
from statgpt.common.data.sdmx.v21.dataset import Sdmx21DataResponse
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus


def _dim(entity_id: str) -> MagicMock:
    m = MagicMock()
    m.entity_id = entity_id
    return m


@pytest.mark.parametrize("stored", (1, 5))
def test_get_display_series_count_returns_stored_value(stored: int) -> None:
    time = _dim("TIME_PERIOD")
    dataset = MagicMock()
    dataset.dimensions.return_value = [time]
    dataset.get_time_dimension.return_value = time
    dataset.map_component_values_id_2_name = MagicMock(return_value=None)
    index = pd.MultiIndex.from_tuples([("2022-07-01",), ("2022-07-02",)], names=["TIME_PERIOD"])
    df = pd.DataFrame({"value": [1.0, 2.0]}, index=index)
    resp = Sdmx21DataResponse(
        dataset=dataset,
        sdmx_query=MagicMock(),
        df=df,
        url=None,
        status=DataResponseStatus(
            request_status=DataRequestStatus.SUCCESS,
            parsing_status=DataParsingStatus.SUCCESS,
        ),
        display_series_count=stored,
    )
    assert resp.get_display_series_count() == stored


def test_get_display_series_count_defaults_to_zero() -> None:
    resp = Sdmx21DataResponse(
        dataset=MagicMock(),
        sdmx_query=MagicMock(),
        df=pd.DataFrame(),
        url=None,
        status=DataResponseStatus(
            request_status=DataRequestStatus.FAILED,
            parsing_status=DataParsingStatus.NA,
        ),
    )
    assert resp.get_display_series_count() == 0
