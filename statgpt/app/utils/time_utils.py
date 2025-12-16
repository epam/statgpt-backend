import typing as t
from datetime import date, datetime

from dateutil.relativedelta import relativedelta


def format_date_freq_a(date: t.Union[date, datetime]) -> str:
    """Format date for Annual frequency"""
    return f"{date.year}"


def format_date_freq_q(date: t.Union[date, datetime]) -> str:
    """Format date for Quarterly frequency"""
    year = date.year
    month = date.month
    quarter = (month - 1) // 3 + 1
    return f"{year}-Q{quarter}"


def format_date_freq_m(date: t.Union[date, datetime]) -> str:
    """Format date for Monthly frequency"""
    year = date.year
    month = date.month
    return f"{year}-M{month :0>2}"


def get_relative_quarter(ref_date: t.Union[date, datetime], quarter_offset: int) -> str:
    """
    Get quarter relative to a 'ref_date'.
    'ref_date' - reference date (for example today)
    'quarter_offset' - offset (in quarters) from 'ref_date'.
        could be any integer (positive, negative)
    """
    # 'new_date' may be an arbitraty month of a new quarter - it's ok
    delta = relativedelta(months=3 * quarter_offset)
    new_date = ref_date + delta
    # 'format_date_freq_q' returns str in format '2023-Q3'.
    # thus it's not important which month of the quarter was assigned to be a 'new_date' above,
    # as long as the month belonged to a correct quarter.
    res = format_date_freq_q(new_date)
    return res
