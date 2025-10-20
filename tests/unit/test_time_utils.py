from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest

from common.utils.time_utils import (
    format_date_long,
    get_time_period_bounds,
    get_today_date_long,
    get_ts_now_str,
    get_ts_utcnow,
    get_ts_utcnow_str,
)


@pytest.mark.parametrize(
    "values, expected",
    [
        # Empty list
        ([], None),
        # Single values
        (["2023"], ("2023", "2023")),
        (["2023-Q1"], ("2023-Q1", "2023-Q1")),
        (["2023-M01"], ("2023-M01", "2023-M01")),
        # Annual only
        (["2021", "2023", "2022"], ("2021", "2023")),
        (["2025", "2020", "2022"], ("2020", "2025")),
        # Quarterly only
        (["2023-Q1", "2023-Q4", "2023-Q2"], ("2023-Q1", "2023-Q4")),
        (["2022-Q4", "2023-Q1", "2023-Q2"], ("2022-Q4", "2023-Q2")),
        # Monthly only
        (["2023-M01", "2023-M12", "2023-M06"], ("2023-M01", "2023-M12")),
        (["2022-M12", "2023-M01", "2023-M02"], ("2022-M12", "2023-M02")),
        # Mixed frequencies - annual and quarterly
        (["2021", "2021-Q4"], ("2021", "2021")),  # Prefer annual at end
        (["2021-Q1", "2021"], ("2021-Q1", "2021")),  # Annual exists for end year
        (["2020-Q4", "2021", "2021-Q2"], ("2020-Q4", "2021")),  # Annual at end
        (["2020", "2021-Q1", "2021-Q4"], ("2020", "2021-Q4")),  # No annual for 2021
        # Mixed frequencies - annual and monthly
        (["2021", "2021-M12"], ("2021", "2021")),  # Prefer annual at end
        (["2021-M01", "2021"], ("2021-M01", "2021")),  # Annual exists for end year
        (["2020-M12", "2021", "2021-M06"], ("2020-M12", "2021")),  # Annual at end
        # Mixed frequencies - quarterly and monthly
        (["2021-Q1", "2021-M10"], ("2021-Q1", "2021-M10")),  # Q1 (Jan-Mar) < M10 (Oct)
        (["2025-Q1", "2025-M10"], ("2025-Q1", "2025-M10")),  # Q1 < M10 within year
        (["2021-Q4", "2021-M01"], ("2021-M01", "2021-Q4")),  # M01 (Jan) < Q4 (Oct-Dec)
        (["2021-M01", "2021-Q1"], ("2021-M01", "2021-Q1")),  # M01 and Q1 both start Jan
        (["2021-M02", "2021-Q1"], ("2021-Q1", "2021-M02")),  # Q1 (Jan) < M02 (Feb)
        (["2021-M03", "2021-Q1"], ("2021-Q1", "2021-M03")),  # Q1 (Jan) < M03 (Mar)
        (["2021-M04", "2021-Q2"], ("2021-M04", "2021-Q2")),  # M04 and Q2 both start Apr
        # Mixed all frequencies
        (
            ["2022", "2021-Q1", "2022-Q2", "2022-Q3", "2023-M01", "2023-M02"],
            ("2021-Q1", "2023-M02"),
        ),
        (["2021", "2021-Q1", "2021-M12"], ("2021", "2021")),  # Annual preferred at end
        (["2020", "2021-Q1", "2021-M01", "2022"], ("2020", "2022")),  # Annual at both ends
        # Edge cases with sorting
        (["2024", "2023-Q4", "2024-M01"], ("2023-Q4", "2024")),  # Annual exists for 2024
        (["1998", "2005-Q1", "2024-Q4", "2025-M12"], ("1998", "2025-M12")),
        # Same year different frequencies
        (["2023-Q1", "2023-Q2", "2023-M06"], ("2023-Q1", "2023-M06")),  # Q1 < Q2 < M06
        (["2023", "2023-Q4", "2023-M12"], ("2023", "2023")),  # Annual preferred
        (["2021-Q1", "2021-Q2", "2021-M02"], ("2021-Q1", "2021-Q2")),  # Q1 < M02 < Q2
        # Cross-year boundaries
        (["2020-M12", "2021-Q1", "2021-M01"], ("2020-M12", "2021-Q1")),
        (["2020", "2021-Q1"], ("2020", "2021-Q1")),
        (["2020-Q4", "2021"], ("2020-Q4", "2021")),
    ],
)
def test_get_time_period_bounds(values, expected):
    result = get_time_period_bounds(values)
    assert result == expected


def test_get_time_period_bounds_invalid_format():
    with pytest.raises(ValueError, match="Invalid time period format"):
        get_time_period_bounds(["2023-W01"])  # Invalid format

    with pytest.raises(ValueError, match="Invalid time period format"):
        get_time_period_bounds(["2023-D01"])  # Invalid format

    with pytest.raises(ValueError, match="Invalid time period format"):
        get_time_period_bounds(["invalid"])  # Cannot parse as year

    with pytest.raises(ValueError, match="Invalid time period format"):
        get_time_period_bounds(["2023-QX"])  # Invalid quarter format

    with pytest.raises(ValueError, match="Invalid time period format"):
        get_time_period_bounds(["2023-MXX"])  # Invalid month format


class TestGetTsNowStr:
    @patch("common.utils.time_utils.datetime")
    def test_default_format(self, mock_datetime):
        mock_datetime.now.return_value = datetime(2023, 10, 15, 14, 30, 45)
        result = get_ts_now_str()
        assert result == "20231015-143045"

    @patch("common.utils.time_utils.datetime")
    def test_custom_format(self, mock_datetime):
        mock_datetime.now.return_value = datetime(2023, 10, 15, 14, 30, 45)
        result = get_ts_now_str(ts_format="%Y-%m-%d")
        assert result == "2023-10-15"

    @patch("common.utils.time_utils.datetime")
    def test_different_datetime(self, mock_datetime):
        mock_datetime.now.return_value = datetime(2000, 1, 1, 0, 0, 0)
        result = get_ts_now_str()
        assert result == "20000101-000000"


class TestGetTsUtcnow:
    @patch("common.utils.time_utils.datetime")
    def test_returns_utc_datetime(self, mock_datetime):
        utc_time = datetime(2023, 10, 15, 14, 30, 45, tzinfo=timezone.utc)
        mock_datetime.now.return_value = utc_time
        result = get_ts_utcnow()
        mock_datetime.now.assert_called_once_with(timezone.utc)
        assert result == utc_time


class TestGetTsUtcnowStr:
    @patch("common.utils.time_utils.get_ts_utcnow")
    def test_default_format(self, mock_get_ts_utcnow):
        mock_get_ts_utcnow.return_value = datetime(2023, 10, 15, 14, 30, 45, tzinfo=timezone.utc)
        result = get_ts_utcnow_str()
        assert result == "20231015-143045"

    @patch("common.utils.time_utils.get_ts_utcnow")
    def test_custom_format(self, mock_get_ts_utcnow):
        mock_get_ts_utcnow.return_value = datetime(2023, 10, 15, 14, 30, 45, tzinfo=timezone.utc)
        result = get_ts_utcnow_str(ts_format="%Y/%m/%d %H:%M")
        assert result == "2023/10/15 14:30"


class TestFormatDateLong:
    def test_single_digit_day(self):
        test_date = date(2023, 10, 5)
        result = format_date_long(test_date)
        assert result == "5 October 2023"

    def test_double_digit_day(self):
        test_date = date(2023, 12, 25)
        result = format_date_long(test_date)
        assert result == "25 December 2023"

    def test_all_months(self):
        expected = [
            "1 January 2023",
            "1 February 2023",
            "1 March 2023",
            "1 April 2023",
            "1 May 2023",
            "1 June 2023",
            "1 July 2023",
            "1 August 2023",
            "1 September 2023",
            "1 October 2023",
            "1 November 2023",
            "1 December 2023",
        ]
        for month_num, expected_str in enumerate(expected, 1):
            test_date = date(2023, month_num, 1)
            assert format_date_long(test_date) == expected_str

    def test_leap_year_february(self):
        test_date = date(2024, 2, 29)
        result = format_date_long(test_date)
        assert result == "29 February 2024"

    def test_different_years(self):
        test_date = date(1999, 12, 31)
        result = format_date_long(test_date)
        assert result == "31 December 1999"

        test_date = date(2000, 1, 1)
        result = format_date_long(test_date)
        assert result == "1 January 2000"


class TestGetTodayDateLong:
    @patch("common.utils.time_utils.date")
    def test_returns_todays_date_formatted(self, mock_date):
        mock_date.today.return_value = date(2023, 10, 15)
        result = get_today_date_long()
        assert result == "15 October 2023"

    @patch("common.utils.time_utils.date")
    def test_different_date(self, mock_date):
        mock_date.today.return_value = date(2024, 1, 1)
        result = get_today_date_long()
        assert result == "1 January 2024"
