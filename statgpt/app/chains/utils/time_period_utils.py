import re

from statgpt.common.data.base import Query
from statgpt.common.utils import IntervalProcessor

ABSOLUTE_FILTER_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}$")


def get_relative_aware_time_period_query(default_query: Query) -> Query:
    values = default_query.values
    if all(ABSOLUTE_FILTER_DATE_RE.fullmatch(v) for v in values):
        return default_query
    interval_processor = IntervalProcessor()
    time_periods = [
        get_relative_aware_time_period(time_period, interval_processor)
        for time_period in default_query.values
    ]
    return Query(values=time_periods, operator=default_query.operator)


def get_relative_aware_time_period(
    time_period: str, interval_processor: IntervalProcessor | None = None
) -> str:
    if ABSOLUTE_FILTER_DATE_RE.fullmatch(time_period):
        return time_period
    if not interval_processor:
        interval_processor = IntervalProcessor()
    return interval_processor.get_absolute_date_str(time_period)
