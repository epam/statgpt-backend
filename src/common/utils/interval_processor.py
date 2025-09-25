import re
from datetime import datetime
from enum import StrEnum
from typing import Self

from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field, model_validator


class IntervalType(StrEnum):
    LAST = "last"
    NEXT = "next"
    TO_DATE = "to_date"
    FROM_NOW = "from_now"
    REGULAR_POSITIVE = "regular_positive"
    REGULAR_NEGATIVE = "regular_negative"


class Duration(BaseModel):
    years: int = Field()
    months: int = Field()
    interval_type: IntervalType = Field()

    @model_validator(mode="after")
    def validate_duration(self) -> Self:
        if self.years <= 0 and self.months <= 0:
            raise ValueError("Either years or months must be greater than 0")
        return self


class IntervalProcessor:
    """
    Process interval strings to generate date ranges.

    Supported interval formats:

        Note: All examples assume current date is 2025-10-10

        - "last <n> years/months"
            Calendar years/months preceding current date
            Example: "last 1 year" → 2024-01-01 to 2024-12-31

        - "next <n> years/months"
            Calendar years/months following current date
            Example: "next 1 year" → 2026-01-01 to 2026-12-31

        - "<n> years/months to date"
            From start of past calendar year/month until current date
            Example: "1 year to date" → 2025-01-01 to 2025-10-10

        - "<n> years/months from now"
            From current date until end of future calendar year/month
            Example: "1 year from now" → 2025-10-10 to 2026-12-31

        - "(+-)?<n> years/months"
            Relative period from current date
            Example: "1 year" → 2025-10-10 to 2026-10-10
            Example: "-1 year" → 2024-10-10 to 2025-10-10
    """

    _NUMBER = r"[1-9]\d*"
    _YEARS = r"y(?:ears?)?"
    _MONTHS = r"m(?:onths?)?"
    _UNITS = f"{_YEARS}|{_MONTHS}"
    _SPACE = r"[_ ]"
    _SIGN = r"[-+]"
    _LAST_NEXT = (
        f"^(?P<prefix>last|next){_SPACE}(?P<number>{_NUMBER})?{_SPACE}?(?P<units>{_UNITS})$"
    )
    _TO_DATE_FROM_NOW = f"^(?P<number>{_NUMBER})?{_SPACE}?(?P<units>{_UNITS}){_SPACE}(?P<postfix>to{_SPACE}date|from{_SPACE}now)$"
    _REGULAR_YEARS = f"^(?P<sign>{_SIGN})?(?P<number>{_NUMBER})?{_SPACE}?(?P<units>{_YEARS}){_SPACE}?((?P<additional_months>{_NUMBER}){_SPACE}?{_MONTHS})?$"
    _REGULAR_MONTHS = f"^(?P<sign>{_SIGN})?(?P<number>{_NUMBER})?{_SPACE}?(?P<units>{_MONTHS})$"

    def __init__(self) -> None:
        # last/next calendar year/month from date
        self._last_next_pattern = re.compile(self._LAST_NEXT)
        # calendar year/month to date/from date
        self._to_date_from_now_pattern = re.compile(self._TO_DATE_FROM_NOW)
        # any interval with years, months and +/- sign from date
        self._regular_years_pattern = re.compile(self._REGULAR_YEARS)
        # any interval with months only and +/- sign from date
        self._regular_months_pattern = re.compile(self._REGULAR_MONTHS)

    def _parse_duration(self, interval_str: str) -> Duration:
        """Parse the interval string to get years and months"""

        if match := self._last_next_pattern.match(interval_str):
            prefix = match.group("prefix")
            interval_type = IntervalType(prefix)
            units = match.group("units")
            if units.startswith('y'):
                years = int(match.group("number") or 1)
                months = 0
            elif units.startswith('m'):
                years = 0
                months = int(match.group("number") or 1)
            else:
                raise ValueError(f"Invalid interval format: {interval_str}")
        elif match := self._to_date_from_now_pattern.match(interval_str):
            postfix = match.group("postfix").replace(" ", "_")
            interval_type = IntervalType(postfix)
            units = match.group("units")
            if units.startswith('y'):
                years = int(match.group("number") or 1)
                months = 0
            elif units.startswith('m'):
                years = 0
                months = int(match.group("number") or 1)
            else:
                raise ValueError(f"Invalid interval format: {interval_str}")
        elif match := (
            self._regular_years_pattern.match(interval_str)
            or self._regular_months_pattern.match(interval_str)
        ):
            sign = match.group("sign")
            if not sign or sign == "+":
                interval_type = IntervalType.REGULAR_POSITIVE
            elif sign == "-":
                interval_type = IntervalType.REGULAR_NEGATIVE
            else:
                raise ValueError(f"Invalid interval format: {interval_str}")
            units = match.group("units")
            if units.startswith('y'):
                years = int(match.group("number") or 1)
                months = 0
            elif units.startswith('m'):
                years = 0
                months = int(match.group("number") or 1)
            else:
                raise ValueError(f"Invalid interval format: {interval_str}")
            if "additional_months" in match.groupdict() and (
                additional_months := match.group("additional_months")
            ):
                months += int(additional_months)
        else:
            raise ValueError(f"Invalid interval format: {interval_str}")

        if years < 2 and units == 'years':
            raise ValueError(f"Invalid interval format: {interval_str}")
        if months < 2 and units == 'months':
            raise ValueError(f"Invalid interval format: {interval_str}")

        return Duration(years=years, months=months, interval_type=interval_type)

    def _process_to_date(
        self, years: int, months: int, date: datetime, interval_str: str
    ) -> tuple[datetime, datetime]:
        if years > 0:
            start_date = datetime(date.year + 1, 1, 1) - relativedelta(years=years)
        elif months > 0:
            # use relativedelta in case `date` is December (12th month) to avoid overflow
            next_month_start = (date + relativedelta(months=1)).replace(day=1)
            start_date = next_month_start - relativedelta(months=months)
        else:
            raise ValueError(f"Invalid interval format: {interval_str}")
        return start_date, date

    def _process_from_now(
        self, years: int, months: int, date: datetime, interval_str: str
    ) -> tuple[datetime, datetime]:
        if years > 0:
            end_date = datetime(date.year - 1, 12, 31) + relativedelta(years=years)
        elif months > 0:
            end_date = (
                datetime(date.year, date.month, 1)
                + relativedelta(months=months)
                - relativedelta(days=1)
            )
        else:
            raise ValueError(f"Invalid interval format: {interval_str}")
        return date, end_date

    def _process_last(
        self, years: int, months: int, date: datetime, interval_str: str
    ) -> tuple[datetime, datetime]:
        if years > 0:
            start_date = datetime(date.year, 1, 1) - relativedelta(years=years)
            end_date = datetime(date.year, 1, 1) - relativedelta(days=1)
        elif months > 0:
            start_date = datetime(date.year, date.month, 1) - relativedelta(months=months)
            end_date = datetime(date.year, date.month, 1) - relativedelta(days=1)
        else:
            raise ValueError(f"Invalid interval format: {interval_str}")
        return start_date, end_date

    def _process_next(
        self, years: int, months: int, date: datetime, interval_str: str
    ) -> tuple[datetime, datetime]:
        if years > 0:
            start_date = datetime(date.year, 1, 1) + relativedelta(years=1)
            end_date = (
                datetime(date.year, 1, 1) + relativedelta(years=years + 1) - relativedelta(days=1)
            )
        elif months > 0:
            start_date = datetime(date.year, date.month, 1) + relativedelta(months=1)
            # use relativedelta in case `date` is December (12th month) to avoid overflow
            next_month_start = (date + relativedelta(months=1)).replace(day=1)
            end_date = next_month_start + relativedelta(months=months) - relativedelta(days=1)
        else:
            raise ValueError(f"Invalid interval format: {interval_str}")
        return start_date, end_date

    def _process_regular_negative(
        self, years: int, months: int, date: datetime
    ) -> tuple[datetime, datetime]:
        end_date = date
        delta = relativedelta(years=years, months=months)
        start_date = date - delta
        return start_date, end_date

    def _process_regular_positive(
        self, years: int, months: int, date: datetime
    ) -> tuple[datetime, datetime]:
        start_date = date
        delta = relativedelta(years=years, months=months)
        end_date = date + delta
        return start_date, end_date

    def get_interval(
        self, interval_str: str, date: datetime | None = None
    ) -> tuple[datetime, datetime]:
        """
        Get the start and end date for the given interval string
        :param interval_str: The interval string
        :param date: The reference date
        :return: The start and end date
        """
        if date is None:
            date = datetime.now()

        duration = self._parse_duration(interval_str)
        years, months, interval_type = duration.years, duration.months, duration.interval_type

        if interval_type == IntervalType.TO_DATE:
            return self._process_to_date(years, months, date, interval_str)
        elif interval_type == IntervalType.FROM_NOW:
            return self._process_from_now(years, months, date, interval_str)
        elif interval_type == IntervalType.LAST:
            return self._process_last(years, months, date, interval_str)
        elif interval_type == IntervalType.NEXT:
            return self._process_next(years, months, date, interval_str)
        elif interval_type == IntervalType.REGULAR_NEGATIVE:
            return self._process_regular_negative(years, months, date)
        elif interval_type == IntervalType.REGULAR_POSITIVE:
            return self._process_regular_positive(years, months, date)
        else:
            raise ValueError(f"Invalid interval format: {interval_str}")
