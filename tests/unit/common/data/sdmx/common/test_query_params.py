"""Tests for version-specific SDMX data query params on SdmxDataSetQuery."""

from statgpt.common.data.sdmx.common import (
    SdmxDataSetQuery,
    SdmxQueryReadinessStatus,
    TimeDimensionQuery,
)


def _query(start: str | None = None, end: str | None = None) -> SdmxDataSetQuery:
    time_query = None
    if start is not None or end is not None:
        time_query = TimeDimensionQuery(
            time_dimension_id="TIME_PERIOD", start_period=start, end_period=end
        )
    return SdmxDataSetQuery(
        status=SdmxQueryReadinessStatus.READY,
        categorical_dimensions={},
        time_dimension_query=time_query,
        missing_dimensions=[],
    )


class TestGetParamsV21:
    def test_no_time(self) -> None:
        assert _query().get_params_v21() == {"detail": "full"}

    def test_start_and_end(self) -> None:
        assert _query(start="2020", end="2024").get_params_v21() == {
            "detail": "full",
            "startPeriod": "2020",
            "endPeriod": "2024",
        }

    def test_start_only(self) -> None:
        assert _query(start="2020").get_params_v21() == {
            "detail": "full",
            "startPeriod": "2020",
        }

    def test_end_only(self) -> None:
        assert _query(end="2024").get_params_v21() == {
            "detail": "full",
            "endPeriod": "2024",
        }


class TestGetParamsV30:
    def test_no_time(self) -> None:
        assert _query().get_params_v30() == {"attributes": "all"}

    def test_start_and_end(self) -> None:
        params = _query(start="2020", end="2024").get_params_v30()
        assert params == {"attributes": "all", "c[TIME_PERIOD]": "ge:2020+le:2024"}

    def test_start_only(self) -> None:
        assert _query(start="2020").get_params_v30() == {
            "attributes": "all",
            "c[TIME_PERIOD]": "ge:2020",
        }

    def test_end_only(self) -> None:
        assert _query(end="2024").get_params_v30() == {
            "attributes": "all",
            "c[TIME_PERIOD]": "le:2024",
        }

    def test_never_includes_detail(self) -> None:
        assert "detail" not in _query(start="2020", end="2024").get_params_v30()
