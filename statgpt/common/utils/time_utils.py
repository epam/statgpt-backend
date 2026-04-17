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


def _parse_time_value(time_val: str) -> tuple[int, int, str]:
    """
    Parse time value into (year, period_order, original_value).
    period_order: months 1-12, quarters mapped to starting month (Q1=1, Q2=4, Q3=7, Q4=10), annual 999.
    Monthly periods may be SDMX-style (YYYY-MNN) or ISO calendar month (YYYY-MM).
    """
    parts = time_val.split('-')
    try:
        year = int(parts[0])
    except ValueError:
        raise ValueError(f"Invalid time period format: {time_val}")

    if len(parts) == 1:
        return (year, 999, time_val)  # Annual

    if len(parts) == 2:
        if parts[1].startswith('Q'):
            try:
                quarter = int(parts[1][1:])
                # Map quarters to their starting month: Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
                quarter_to_month = {1: 1, 2: 4, 3: 7, 4: 10}
                return (year, quarter_to_month[quarter], time_val)
            except (ValueError, KeyError):
                raise ValueError(f"Invalid time period format: {time_val}")
        elif parts[1].startswith('M'):
            try:
                month = int(parts[1][1:])
                return (year, month, time_val)
            except ValueError:
                raise ValueError(f"Invalid time period format: {time_val}")
        elif parts[1].isdigit():
            # ISO-style calendar month YYYY-MM (e.g. 1964-01), common in SDMX / tabular time
            month = int(parts[1])
            if 1 <= month <= 12:
                return (year, month, time_val)

    if len(parts) == 3:
        try:
            month = int(parts[1])
            day = int(parts[2])
            # Validate full date periods (YYYY-MM-DD) and keep period_order aligned with month.
            date(year, month, day)
            return (year, month, time_val)
        except ValueError:
            raise ValueError(f"Invalid time period format: {time_val}")

    raise ValueError(f"Invalid time period format: {time_val}")


def get_time_period_bounds(values: list[str]) -> tuple[str, str] | None:
    """
    Get the time period bounds from a list of time period values.
    Handles annual (2023), quarterly (2024-Q1), monthly (2023-M01 or 1964-01), and date (2023-10-03) formats.
    """
    if not values:
        return None

    parsed_values = [_parse_time_value(v) for v in values]
    parsed_values.sort()

    start_year, start_order, start_value = parsed_values[0]
    end_year, end_order, end_value = parsed_values[-1]

    values_set = set(values)
    start_year_str = str(start_year)
    end_year_str = str(end_year)

    # Prefer annual at start unless it's Q1/M01 with only one sub-year period
    if start_year_str in values_set:
        sub_year_count = sum(1 for v in values if v.startswith(f"{start_year}-"))
        if not (start_order == 1 and sub_year_count == 1):
            start_value = start_year_str

    # Always prefer annual at end if it exists
    if end_year_str in values_set:
        end_value = end_year_str

    return start_value, end_value
