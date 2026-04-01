from datetime import datetime

import pytest
from freezegun import freeze_time

from statgpt.common.utils.interval_processor import IntervalProcessor


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
            datetime(2025, 1, 29),
            datetime(2025, 1, 29),
            datetime(2025, 2, 28),
        ),
        (
            "1m",
            datetime(2025, 1, 30),
            datetime(2025, 1, 30),
            datetime(2025, 2, 28),
        ),
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
        (
            "1m",
            datetime(2025, 2, 27),
            datetime(2025, 2, 27),
            datetime(2025, 3, 27),
        ),
        (
            "-1m",
            datetime(2025, 3, 29),
            datetime(2025, 2, 28),
            datetime(2025, 3, 29),
        ),
        (
            "-1m",
            datetime(2025, 3, 30),
            datetime(2025, 2, 28),
            datetime(2025, 3, 30),
        ),
        (
            "-1m",
            datetime(2025, 3, 31),
            datetime(2025, 2, 28),
            datetime(2025, 3, 31),
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
        # quarter tests — Q4 (Oct-Dec)
        (
            "q_to_date",
            datetime(2025, 10, 10),
            datetime(2025, 10, 1),
            datetime(2025, 10, 10),
        ),
        (
            "2q_to_date",
            datetime(2025, 10, 10),
            datetime(2025, 7, 1),
            datetime(2025, 10, 10),
        ),
        # Q3 (Jul-Sep)
        (
            "q_to_date",
            datetime(2025, 8, 15),
            datetime(2025, 7, 1),
            datetime(2025, 8, 15),
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
        # quarter tests — Q4 (Oct-Dec)
        (
            "q_from_now",
            datetime(2025, 10, 10),
            datetime(2025, 10, 10),
            datetime(2025, 12, 31),
        ),
        (
            "2q_from_now",
            datetime(2025, 10, 10),
            datetime(2025, 10, 10),
            datetime(2026, 3, 31),
        ),
        # Q3 (Jul-Sep)
        (
            "q_from_now",
            datetime(2025, 8, 15),
            datetime(2025, 8, 15),
            datetime(2025, 9, 30),
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
        # quarter tests
        (
            "last_quarter",
            datetime(2025, 10, 10),
            datetime(2025, 7, 1),
            datetime(2025, 9, 30),
        ),
        (
            "last_1q",
            datetime(2025, 10, 10),
            datetime(2025, 7, 1),
            datetime(2025, 9, 30),
        ),
        (
            "last_2q",
            datetime(2025, 10, 10),
            datetime(2025, 4, 1),
            datetime(2025, 9, 30),
        ),
        # year boundary: Q1 → previous Q4
        (
            "last_quarter",
            datetime(2025, 2, 15),
            datetime(2024, 10, 1),
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
        # quarter tests
        (
            "next_quarter",
            datetime(2025, 10, 10),
            datetime(2026, 1, 1),
            datetime(2026, 3, 31),
        ),
        (
            "next_1q",
            datetime(2025, 10, 10),
            datetime(2026, 1, 1),
            datetime(2026, 3, 31),
        ),
        (
            "next_2q",
            datetime(2025, 10, 10),
            datetime(2026, 1, 1),
            datetime(2026, 6, 30),
        ),
        # Q3 → next is Q4 (same year)
        (
            "next_quarter",
            datetime(2025, 8, 15),
            datetime(2025, 10, 1),
            datetime(2025, 12, 31),
        ),
        # Q4 → next is Q1 of next year
        (
            "next_quarter",
            datetime(2025, 12, 10),
            datetime(2026, 1, 1),
            datetime(2026, 3, 31),
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
        # quarter regular types are forbidden
        "q",
        "0q",
        "1q",
        "2q",
        "quarter",
        "2quarters",
        "-q",
        "-1q",
        "-2q",
        "-quarter",
        "-1quarter",
        "-2quarters",
        "+q",
        "+1q",
        "+2q",
        "+quarter",
        "+2quarters",
        # mixed with quarters are forbidden
        "1y1q",
        "2y1q",
        "-1y1q",
        #
        "1q2m",
        "1q1y",
        #
        "1m1q",
        "-1m1q",
        # plural with count < 2
        "quarters",
        "1quarters",
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
    "interval_str,current_date,expected_date",
    [
        # ── now ──
        ("now", datetime(2025, 10, 10), datetime(2025, 10, 10)),
        # ── regular negative → returns start date ──
        ("-1y", datetime(2025, 10, 10), datetime(2024, 10, 10)),
        ("-1y2m", datetime(2025, 10, 10), datetime(2024, 8, 10)),
        ("-3m", datetime(2025, 10, 10), datetime(2025, 7, 10)),
        ("-month", datetime(2025, 10, 10), datetime(2025, 9, 10)),
        ("-1m", datetime(2025, 3, 31), datetime(2025, 2, 28)),
        # ── regular positive → returns end date ──
        ("1y", datetime(2025, 10, 10), datetime(2026, 10, 10)),
        ("3m", datetime(2025, 10, 10), datetime(2026, 1, 10)),
        ("1y2m", datetime(2025, 10, 10), datetime(2026, 12, 10)),
        ("1m", datetime(2025, 12, 10), datetime(2026, 1, 10)),
        ("1m", datetime(2025, 1, 31), datetime(2025, 2, 28)),
        # ── to_date → returns start of period ──
        ("y_to_date", datetime(2025, 10, 10), datetime(2025, 1, 1)),
        ("2y_to_date", datetime(2025, 10, 10), datetime(2024, 1, 1)),
        ("m_to_date", datetime(2025, 10, 10), datetime(2025, 10, 1)),
        ("2m_to_date", datetime(2025, 10, 10), datetime(2025, 9, 1)),
        ("q_to_date", datetime(2025, 10, 10), datetime(2025, 10, 1)),
        ("2q_to_date", datetime(2025, 10, 10), datetime(2025, 7, 1)),
        ("q_to_date", datetime(2025, 8, 15), datetime(2025, 7, 1)),
        # ── from_now → returns end of period ──
        ("y_from_now", datetime(2025, 10, 10), datetime(2025, 12, 31)),
        ("2y_from_now", datetime(2025, 10, 10), datetime(2026, 12, 31)),
        ("m_from_now", datetime(2025, 10, 10), datetime(2025, 10, 31)),
        ("2m_from_now", datetime(2025, 10, 10), datetime(2025, 11, 30)),
        ("q_from_now", datetime(2025, 10, 10), datetime(2025, 12, 31)),
        ("2q_from_now", datetime(2025, 10, 10), datetime(2026, 3, 31)),
        ("q_from_now", datetime(2025, 8, 15), datetime(2025, 9, 30)),
    ],
)
def test_get_absolute_date(
    processor: IntervalProcessor,
    interval_str: str,
    current_date: datetime,
    expected_date: datetime,
) -> None:
    result = processor.get_absolute_date(interval_str, current_date)
    assert result == expected_date


@pytest.mark.parametrize(
    "interval_str",
    [
        "last_year",
        "last_quarter",
        "last_month",
        "next_year",
        "next_quarter",
        "next_month",
    ],
)
def test_get_absolute_date_invalid_for_last_next(
    processor: IntervalProcessor, interval_str: str
) -> None:
    with pytest.raises(ValueError, match="Invalid relative format"):
        processor.get_absolute_date(interval_str, datetime(2025, 10, 10))


@pytest.mark.parametrize(
    "interval_str,expected_years,expected_months,expected_quarters",
    [
        # ── Regular positive (years) ──
        ("year", 1, 0, 0),
        ("1y", 1, 0, 0),
        ("1year", 1, 0, 0),
        ("2y", 2, 0, 0),
        ("2year", 2, 0, 0),  # should this be valid?
        ("2years", 2, 0, 0),
        # ── Regular positive (months) ──
        ("month", 0, 1, 0),
        ("1m", 0, 1, 0),
        ("1month", 0, 1, 0),
        ("2m", 0, 2, 0),
        ("2month", 0, 2, 0),  # should this be valid?
        ("2months", 0, 2, 0),
        ("3m", 0, 3, 0),
        # ── Regular negative (years) ──
        ("-year", 1, 0, 0),
        ("-1y", 1, 0, 0),
        ("-1year", 1, 0, 0),
        ("-2y", 2, 0, 0),
        ("-2years", 2, 0, 0),
        # ── Regular negative (months) ──
        ("-month", 0, 1, 0),
        ("-1m", 0, 1, 0),
        ("-1month", 0, 1, 0),
        ("-2m", 0, 2, 0),
        ("-2months", 0, 2, 0),
        # ── Regular (combined) ──
        ("1y2m", 1, 2, 0),
        ("-1y2m", 1, 2, 0),
        # ── Last ──
        ("last_y", 1, 0, 0),
        ("last_year", 1, 0, 0),
        ("last_1y", 1, 0, 0),
        ("last_1year", 1, 0, 0),
        ("last_2y", 2, 0, 0),
        ("last_2years", 2, 0, 0),
        ("last_m", 0, 1, 0),
        ("last_month", 0, 1, 0),
        ("last_1m", 0, 1, 0),
        ("last_2m", 0, 2, 0),
        ("last_2months", 0, 2, 0),
        ("last_q", 0, 0, 1),
        ("last_quarter", 0, 0, 1),
        ("last_1q", 0, 0, 1),
        ("last_1quarter", 0, 0, 1),
        ("last_2q", 0, 0, 2),
        ("last_2quarters", 0, 0, 2),
        # ── Next ──
        ("next_y", 1, 0, 0),
        ("next_year", 1, 0, 0),
        ("next_1y", 1, 0, 0),
        ("next_1year", 1, 0, 0),
        ("next_2y", 2, 0, 0),
        ("next_2years", 2, 0, 0),
        ("next_m", 0, 1, 0),
        ("next_month", 0, 1, 0),
        ("next_1m", 0, 1, 0),
        ("next_2m", 0, 2, 0),
        ("next_2months", 0, 2, 0),
        ("next_5m", 0, 5, 0),
        ("next_q", 0, 0, 1),
        ("next_quarter", 0, 0, 1),
        ("next_1q", 0, 0, 1),
        ("next_1quarter", 0, 0, 1),
        ("next_2q", 0, 0, 2),
        ("next_2quarters", 0, 0, 2),
        # ── To date ──
        ("y_to_date", 1, 0, 0),
        ("year_to_date", 1, 0, 0),
        ("1y_to_date", 1, 0, 0),
        ("1year_to_date", 1, 0, 0),
        ("2y_to_date", 2, 0, 0),
        ("2years_to_date", 2, 0, 0),
        ("m_to_date", 0, 1, 0),
        ("month_to_date", 0, 1, 0),
        ("1m_to_date", 0, 1, 0),
        ("2m_to_date", 0, 2, 0),
        ("2months_to_date", 0, 2, 0),
        ("q_to_date", 0, 0, 1),
        ("quarter_to_date", 0, 0, 1),
        ("1q_to_date", 0, 0, 1),
        ("2q_to_date", 0, 0, 2),
        ("2quarters_to_date", 0, 0, 2),
        # ── From now ──
        ("y_from_now", 1, 0, 0),
        ("year_from_now", 1, 0, 0),
        ("1y_from_now", 1, 0, 0),
        ("1year_from_now", 1, 0, 0),
        ("2y_from_now", 2, 0, 0),
        ("2years_from_now", 2, 0, 0),
        ("m_from_now", 0, 1, 0),
        ("month_from_now", 0, 1, 0),
        ("1m_from_now", 0, 1, 0),
        ("2m_from_now", 0, 2, 0),
        ("2months_from_now", 0, 2, 0),
        ("q_from_now", 0, 0, 1),
        ("quarter_from_now", 0, 0, 1),
        ("1q_from_now", 0, 0, 1),
        ("2q_from_now", 0, 0, 2),
        ("2quarters_from_now", 0, 0, 2),
    ],
)
def test_parse_duration(
    processor: IntervalProcessor,
    interval_str: str,
    expected_years: int,
    expected_months: int,
    expected_quarters: int,
) -> None:
    duration = processor._parse_duration(interval_str)
    assert duration.years == expected_years
    assert duration.months == expected_months
    assert duration.quarters == expected_quarters
