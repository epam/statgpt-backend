"""Reading `sdmx1` MemberSelection values without assuming they are all MemberValues.

A cube-region member selection holds `SelectionValue` objects. Most providers use
`<common:Value>` (parsed as `MemberValue`, with a `.value` attribute), but SDMX 2.1 also
allows `<common:TimeRange>` / `<common:BeforePeriod>` / `<common:AfterPeriod>` (parsed as
`TimeRangeValue` subclasses, which have no `.value`). Reading `.value` unconditionally
raises `AttributeError` and - because the conversion is a single funnel for the whole cube
region - takes down every dimension of the dataset, not just the time dimension.

The helpers here dispatch per value type and skip anything unrecognised with a warning,
so an unsupported value type can never abort the whole conversion again.
"""

import logging

from sdmx.model.common import BaseMemberSelection
from sdmx.model.v21 import AfterPeriod, BeforePeriod, MemberValue, RangePeriod

_log = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%d"


def member_value_ids(selection: BaseMemberSelection) -> set[str]:
    """Coded values in the selection. Time ranges are not coded values and are skipped."""

    result: set[str] = set()
    unsupported: set[str] = set()
    for value in selection.values:
        if isinstance(value, MemberValue):
            result.add(value.value)
        elif not isinstance(value, (RangePeriod, AfterPeriod, BeforePeriod)):
            unsupported.add(type(value).__name__)
    _log_unsupported(selection, unsupported)
    return result


def time_range_bounds(selection: BaseMemberSelection) -> tuple[str | None, str | None]:
    """Outer (start, end) bounds of the selection's time ranges, as `YYYY-MM-DD`.

    Several ranges collapse to the outer envelope (`min` of the starts, `max` of the ends);
    interior gaps of a non-contiguous union are lost. That is acceptable because the bounds
    only drive coarse "is the request entirely outside availability?" checks, and widening
    never wrongly rejects a query.

    `is_inclusive` is ignored for the same reason: treating an exclusive bound as inclusive
    is the safe direction.
    """

    start: str | None = None
    end: str | None = None
    unsupported: set[str] = set()

    for value in selection.values:
        if isinstance(value, RangePeriod):
            start = min_bound(start, value.start.period.strftime(DATE_FORMAT))
            end = max_bound(end, value.end.period.strftime(DATE_FORMAT))
        elif isinstance(value, AfterPeriod):
            start = min_bound(start, value.period.strftime(DATE_FORMAT))
        elif isinstance(value, BeforePeriod):
            end = max_bound(end, value.period.strftime(DATE_FORMAT))
        elif not isinstance(value, MemberValue):
            unsupported.add(type(value).__name__)

    _log_unsupported(selection, unsupported)
    return start, end


def min_bound(current: str | None, candidate: str) -> str:
    """Earlier of the two bounds, treating a missing `current` as "no bound yet"."""
    return candidate if current is None else min(current, candidate)


def max_bound(current: str | None, candidate: str) -> str:
    """Later of the two bounds, treating a missing `current` as "no bound yet"."""
    return candidate if current is None else max(current, candidate)


def _log_unsupported(selection: BaseMemberSelection, type_names: set[str]) -> None:
    if not type_names:
        return
    dimension_id = getattr(selection.values_for, "id", None)
    _log.warning(
        f"Skipping unsupported SDMX selection value types {sorted(type_names)}"
        f" for dimension {dimension_id!r}"
    )
