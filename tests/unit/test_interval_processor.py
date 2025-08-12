from datetime import datetime

import pytest
from freezegun import freeze_time

from common.utils.interval_processor import IntervalProcessor


@pytest.fixture
def processor():
    return IntervalProcessor()


def _test_interval(
    processor: IntervalProcessor,
    interval_str: str,
    current_date: datetime,
    expected_start: datetime,
    expected_end: datetime,
) -> None:
    start, end = processor.get_interval(interval_str, current_date)
    assert start == expected_start
    assert end == expected_end


@pytest.mark.parametrize(
    "interval_str,current_date,expected_start,expected_end",
    [
        # Basic year/month tests
        (
            "-1y",
            datetime(2025, 10, 10),
            datetime(2024, 10, 10),
            datetime(2025, 10, 10),
        ),
        (
            "-1y2m",
            datetime(2025, 10, 10),
            datetime(2024, 8, 10),
            datetime(2025, 10, 10),
        ),
        (
            "-2years",
            datetime(2025, 10, 10),
            datetime(2023, 10, 10),
            datetime(2025, 10, 10),
        ),
        (
            "-3m",
            datetime(2025, 10, 10),
            datetime(2025, 7, 10),
            datetime(2025, 10, 10),
        ),
        # Special cases
        (
            "-month",
            datetime(2025, 10, 10),
            datetime(2025, 9, 10),
            datetime(2025, 10, 10),
        ),
        (
            "-year",
            datetime(2025, 10, 10),
            datetime(2024, 10, 10),
            datetime(2025, 10, 10),
        ),
        (
            "1y",
            datetime(2025, 10, 10),
            datetime(2025, 10, 10),
            datetime(2026, 10, 10),
        ),
        (
            "2m",
            datetime(2025, 10, 10),
            datetime(2025, 10, 10),
            datetime(2025, 12, 10),
        ),
        (
            "1y2m",
            datetime(2025, 10, 10),
            datetime(2025, 10, 10),
            datetime(2026, 12, 10),
        ),
        (
            "25m",
            datetime(2025, 10, 10),
            datetime(2025, 10, 10),
            datetime(2027, 11, 10),
        ),
        # potential month overflow (12->13)
        (
            "1m",
            datetime(2025, 12, 10),
            datetime(2025, 12, 10),
            datetime(2026, 1, 10),
        ),
        (
            "2m",
            datetime(2025, 11, 10),
            datetime(2025, 11, 10),
            datetime(2026, 1, 10),
        ),
        # correct end of month detection
        (
            "1m",
            datetime(2025, 1, 31),
            datetime(2025, 1, 31),
            datetime(2025, 2, 28),
        ),
        (
            "1m",
            datetime(2025, 2, 28),
            datetime(2025, 2, 28),
            datetime(2025, 3, 28),  # NOTE: must be 28th, not 31st
        ),
    ],
)
def test_regular(
    processor: IntervalProcessor,
    interval_str: str,
    current_date: datetime,
    expected_start: datetime,
    expected_end: datetime,
) -> None:
    _test_interval(processor, interval_str, current_date, expected_start, expected_end)


@pytest.mark.parametrize(
    "interval_str,current_date,expected_start,expected_end",
    [
        # to_date postfix tests
        (
            "y_to_date",
            datetime(2025, 10, 10),
            datetime(2025, 1, 1),
            datetime(2025, 10, 10),
        ),
        (
            "2y_to_date",
            datetime(2025, 10, 10),
            datetime(2024, 1, 1),
            datetime(2025, 10, 10),
        ),
        (
            "m_to_date",
            datetime(2025, 10, 10),
            datetime(2025, 10, 1),
            datetime(2025, 10, 10),
        ),
        (
            "2m_to_date",
            datetime(2025, 10, 10),
            datetime(2025, 9, 1),
            datetime(2025, 10, 10),
        ),
        (
            "25m_to_date",
            datetime(2025, 10, 10),
            datetime(2023, 10, 1),
            datetime(2025, 10, 10),
        ),
        # potential month overflow (12->13)
        (
            "m_to_date",
            datetime(2025, 12, 10),
            datetime(2025, 12, 1),
            datetime(2025, 12, 10),
        ),
    ],
)
def test_to_date(
    processor: IntervalProcessor,
    interval_str: str,
    current_date: datetime,
    expected_start: datetime,
    expected_end: datetime,
) -> None:
    _test_interval(processor, interval_str, current_date, expected_start, expected_end)


@pytest.mark.parametrize(
    "interval_str,current_date,expected_start,expected_end",
    [
        # from_now postfix tests
        (
            "y_from_now",
            datetime(2025, 10, 10),
            datetime(2025, 10, 10),
            datetime(2025, 12, 31),
        ),
        (
            "2y_from_now",
            datetime(2025, 10, 10),
            datetime(2025, 10, 10),
            datetime(2026, 12, 31),
        ),
        (
            "m_from_now",
            datetime(2025, 10, 10),
            datetime(2025, 10, 10),
            datetime(2025, 10, 31),
        ),
        (
            "2m_from_now",
            datetime(2025, 10, 10),
            datetime(2025, 10, 10),
            datetime(2025, 11, 30),
        ),
    ],
)
def test_from_now(
    processor: IntervalProcessor,
    interval_str: str,
    current_date: datetime,
    expected_start: datetime,
    expected_end: datetime,
) -> None:
    _test_interval(processor, interval_str, current_date, expected_start, expected_end)


@pytest.mark.parametrize(
    "interval_str,current_date,expected_start,expected_end",
    [
        # Last_ prefix tests
        (
            "last_month",
            datetime(2025, 10, 10),
            datetime(2025, 9, 1),
            datetime(2025, 9, 30),
        ),
        (
            "last_year",
            datetime(2025, 10, 10),
            datetime(2024, 1, 1),
            datetime(2024, 12, 31),
        ),
        (
            "last_1y",
            datetime(2025, 10, 10),
            datetime(2024, 1, 1),
            datetime(2024, 12, 31),
        ),
        (
            "last_2y",
            datetime(2025, 10, 10),
            datetime(2023, 1, 1),
            datetime(2024, 12, 31),
        ),
    ],
)
def test_last(
    processor: IntervalProcessor,
    interval_str: str,
    current_date: datetime,
    expected_start: datetime,
    expected_end: datetime,
) -> None:
    _test_interval(processor, interval_str, current_date, expected_start, expected_end)


@pytest.mark.parametrize(
    "interval_str,current_date,expected_start,expected_end",
    [
        # next prefix tests
        (
            "next_month",
            datetime(2025, 10, 10),
            datetime(2025, 11, 1),
            datetime(2025, 11, 30),
        ),
        (
            "next_year",
            datetime(2025, 10, 10),
            datetime(2026, 1, 1),
            datetime(2026, 12, 31),
        ),
        (
            "next_2y",
            datetime(2025, 10, 10),
            datetime(2026, 1, 1),
            datetime(2027, 12, 31),
        ),
        (
            "next_5m",
            datetime(2025, 10, 10),
            datetime(2025, 11, 1),
            datetime(2026, 3, 31),
        ),
        # potential month overflow (12->13)
        (
            "next_month",
            datetime(2025, 12, 10),
            datetime(2026, 1, 1),
            datetime(2026, 1, 31),
        ),
        # correct end of month detection
        (
            "next_month",
            datetime(2025, 1, 10),
            datetime(2025, 2, 1),
            datetime(2025, 2, 28),
        ),
        (
            "next_month",
            datetime(2028, 1, 10),
            datetime(2028, 2, 1),
            datetime(2028, 2, 29),
        ),
        (
            "next_month",
            datetime(2025, 2, 10),
            datetime(2025, 3, 1),
            datetime(2025, 3, 31),
        ),
        (
            "next_month",
            datetime(2025, 3, 10),
            datetime(2025, 4, 1),
            datetime(2025, 4, 30),
        ),
    ],
)
def test_next(
    processor: IntervalProcessor,
    interval_str: str,
    current_date: datetime,
    expected_start: datetime,
    expected_end: datetime,
) -> None:
    _test_interval(processor, interval_str, current_date, expected_start, expected_end)


@pytest.mark.parametrize(
    "interval_str",
    [
        "invalid",
        "1x",
        "1y1x",
        "lastyear",
        "1y1",
        "y1",
        "1years",
        "years",
        "1months",
        "months",
        "last_2y2m",
        "0m",
        "0y",
        "0y0m",
        "1y0m",
        "1m0y",
        "1m3m",
    ],
)
def test_invalid_interval_format(processor: IntervalProcessor, interval_str: str) -> None:
    with pytest.raises(ValueError, match="Invalid interval format"):
        processor.get_interval(interval_str)


@freeze_time("2025-10-10")
def test_default_current_date(processor: IntervalProcessor) -> None:
    start, end = processor.get_interval("-1y")
    assert start == datetime(2024, 10, 10)
    assert end == datetime(2025, 10, 10)


@pytest.mark.parametrize(
    "interval_str,expected_years,expected_months",
    [
        ("year", 1, 0),
        ("1y", 1, 0),
        ("1year", 1, 0),
        ("2year", 2, 0),  # should this be valid?
        ("2years", 2, 0),
        #
        ("month", 0, 1),
        ("1m", 0, 1),
        ("1month", 0, 1),
        ("2m", 0, 2),
        ("2month", 0, 2),  # should this be valid?
        ("2months", 0, 2),
        ("3m", 0, 3),
        #
        ("1y2m", 1, 2),
        #
        ("last_1y", 1, 0),
        ("last_2y", 2, 0),
        ("last_y", 1, 0),
        ("last_m", 0, 1),
    ],
)
def test_parse_duration(
    processor: IntervalProcessor,
    interval_str: str,
    expected_years: int,
    expected_months: int,
) -> None:
    duration = processor._parse_duration(interval_str)
    assert duration.years == expected_years
    assert duration.months == expected_months
