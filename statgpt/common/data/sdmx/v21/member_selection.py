"""Reading `sdmx1` MemberSelection values without assuming they are all MemberValues.

A cube-region member selection holds `SelectionValue` objects. Most providers use
`<common:Value>` (parsed as `MemberValue`, with a `.value` attribute), but SDMX 2.1 also
allows `<common:TimeRange>` / `<common:BeforePeriod>` / `<common:AfterPeriod>` (parsed as
`TimeRangeValue` subclasses, which have no `.value`). Reading `.value` unconditionally
raises `AttributeError` and - because the conversion is a single funnel for the whole cube
region - takes down every dimension of the dataset, not just the time dimension.

`read_member_selection` dispatches per value type in a single pass and skips anything
unrecognised with a warning, so an unsupported value type can never abort the whole
conversion again. It reports both kinds of value it understands and leaves the callers to
decide what to make of them: only the time dimension may legitimately carry a time range.
"""

import logging
from dataclasses import dataclass

from sdmx.model.common import BaseMemberSelection
from sdmx.model.v21 import AfterPeriod, BeforePeriod, MemberValue, RangePeriod

_log = logging.getLogger(__name__)

_DATE_FORMAT = "%Y-%m-%d"


@dataclass(frozen=True)
class MemberSelectionValues:
    """A member selection split into its coded values and its time-range envelope."""

    coded_values: set[str]
    time_range_start: str | None = None
    time_range_end: str | None = None

    @property
    def has_time_range(self) -> bool:
        return self.time_range_start is not None or self.time_range_end is not None


def read_member_selection(selection: BaseMemberSelection) -> MemberSelectionValues:
    """Read every value of the selection once, warning about the types we cannot interpret.

    Time ranges are reported as `YYYY-MM-DD` bounds. Several ranges collapse to the outer
    envelope (`min` of the starts, `max` of the ends); interior gaps of a non-contiguous union
    are lost. That is acceptable because the bounds only drive coarse "is the request entirely
    outside availability?" checks, and widening never wrongly rejects a query.

    `is_inclusive` is ignored for the same reason: treating an exclusive bound as inclusive
    is the safe direction.
    """

    coded_values: set[str] = set()
    start: str | None = None
    end: str | None = None
    unsupported: set[str] = set()

    for value in selection.values:
        if isinstance(value, MemberValue):
            coded_values.add(value.value)
        elif isinstance(value, RangePeriod):
            start = _min_bound(start, value.start.period.strftime(_DATE_FORMAT))
            end = _max_bound(end, value.end.period.strftime(_DATE_FORMAT))
        elif isinstance(value, AfterPeriod):
            start = _min_bound(start, value.period.strftime(_DATE_FORMAT))
        elif isinstance(value, BeforePeriod):
            end = _max_bound(end, value.period.strftime(_DATE_FORMAT))
        else:
            unsupported.add(type(value).__name__)

    if unsupported:
        _log.warning(
            f"Skipping unsupported SDMX selection value types {sorted(unsupported)}"
            f" for dimension {selection.values_for.id!r}"
        )

    return MemberSelectionValues(
        coded_values=coded_values, time_range_start=start, time_range_end=end
    )


def _min_bound(current: str | None, candidate: str) -> str:
    """Earlier of the two bounds, treating a missing `current` as "no bound yet"."""
    return candidate if current is None else min(current, candidate)


def _max_bound(current: str | None, candidate: str) -> str:
    """Later of the two bounds, treating a missing `current` as "no bound yet"."""
    return candidate if current is None else max(current, candidate)
