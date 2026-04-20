from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timezone

MONTH_SHORT_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# Coarser (wider) periods have a higher rank: year > quarter > month > day
_G_DAY = 1
_G_MONTH = 2
_G_QUARTER = 3
_G_YEAR = 4


@dataclass(frozen=True)
class _Period:
    start: date
    end: date
    granularity: int
    original: str


def _last_day_of_month(year: int, month: int) -> int:
    return monthrange(year, month)[1]


def _quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    first_month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
    last_month = {1: 3, 2: 6, 3: 9, 4: 12}[quarter]
    start = date(year, first_month, 1)
    end = date(year, last_month, _last_day_of_month(year, last_month))
    return start, end


def _parse_period(time_val: str) -> _Period:
    """
    Parse a time string into calendar bounds and granularity.
    Granularity rank: annual (_G_YEAR) > quarter > month > day (_G_DAY).
    """
    parts = time_val.split('-')
    try:
        year = int(parts[0])
    except ValueError:
        raise ValueError(f"Invalid time period format: {time_val}")

    if len(parts) == 1:
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        return _Period(start, end, _G_YEAR, time_val)

    if len(parts) == 2:
        if parts[1].startswith('Q'):
            try:
                quarter = int(parts[1][1:])
                start, end = _quarter_bounds(year, quarter)
                return _Period(start, end, _G_QUARTER, time_val)
            except (ValueError, KeyError):
                raise ValueError(f"Invalid time period format: {time_val}")
        if parts[1].startswith('M'):
            try:
                month = int(parts[1][1:])
            except ValueError:
                raise ValueError(f"Invalid time period format: {time_val}")
            start = date(year, month, 1)
            end = date(year, month, _last_day_of_month(year, month))
            return _Period(start, end, _G_MONTH, time_val)
        if parts[1].isdigit():
            month = int(parts[1])
            if 1 <= month <= 12:
                start = date(year, month, 1)
                end = date(year, month, _last_day_of_month(year, month))
                return _Period(start, end, _G_MONTH, time_val)

    if len(parts) == 3:
        try:
            month = int(parts[1])
            day = int(parts[2])
            d = date(year, month, day)
            return _Period(d, d, _G_DAY, time_val)
        except ValueError:
            raise ValueError(f"Invalid time period format: {time_val}")

    raise ValueError(f"Invalid time period format: {time_val}")


def _is_subsumed_by_coarser(finer: _Period, coarser: _Period) -> bool:
    """True if coarser strictly outranks finer and fully covers its [start, end]."""
    if coarser.granularity <= finer.granularity:
        return False
    return finer.start >= coarser.start and finer.end <= coarser.end


def get_ts_now_str(ts_format="%Y%m%d-%H%M%S") -> str:
    return datetime.now().strftime(ts_format)


def get_ts_utcnow():
    return datetime.now(timezone.utc)


def get_ts_utcnow_str(ts_format="%Y%m%d-%H%M%S") -> str:
    return get_ts_utcnow().strftime(ts_format)


def format_date_long(date_: date) -> str:
    """
    Format date in the following format: "10 October 2023"
    """
    return f"{date_.day} {MONTH_NAMES[date_.month - 1]} {date_.year}"


def get_today_date_long() -> str:
    return format_date_long(date_=date.today())


def get_time_period_bounds(values: list[str]) -> tuple[str, str] | None:
    """
    Get the time period bounds from a list of time period values.

    Coarser periods (year > quarter > month > day) subsume finer ones when the finer
    span lies fully inside the coarser span (same calendar semantics). Bounds are then
    the earliest start and latest end among the remaining periods; ties break by
    lexicographic order of the original strings.
    """
    if not values:
        return None

    periods = [_parse_period(v) for v in values]

    def kept(p: _Period) -> bool:
        return not any(_is_subsumed_by_coarser(p, q) for q in periods if q is not p)

    remaining = [p for p in periods if kept(p)]
    if not remaining:
        # e.g. empty input already handled; single impossible case — fallback to full list
        remaining = periods

    start_p = min(remaining, key=lambda p: (p.start, p.original))
    end_p = max(remaining, key=lambda p: (p.end, p.original))
    return start_p.original, end_p.original
