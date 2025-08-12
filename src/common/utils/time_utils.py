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


# def get_ts_now_str_with_timezone(ts_format="%Y%m%d-%H%M%S") -> str:
#     """add local timezone to the final string"""
#     # NOTE: it seems to be not cross-platform

#     timezone = get_localzone()
#     now = datetime.now(timezone)

#     # check if timezone is present and include it in the string
#     if now.strftime("%Z"):
#         res = now.strftime(f"{ts_format}-%Z")
#     else:
#         res = now.strftime(ts_format)

#     return res


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
