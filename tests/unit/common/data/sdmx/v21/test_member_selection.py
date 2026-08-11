"""Tests for reading SDMX 2.1 cube-region member selections.

OECD expresses `TIME_PERIOD` availability with `<common:TimeRange>`, which `sdmx1` parses into
a `RangePeriod` with no `.value` attribute. Reading `.value` unconditionally used to raise
`AttributeError` and take down the whole cube region with it.
"""

import logging
from datetime import datetime

from sdmx.model.common import Dimension, EndPeriod, StartPeriod
from sdmx.model.v21 import (
    AfterPeriod,
    BeforePeriod,
    MemberSelection,
    MemberValue,
    RangePeriod,
    SelectionValue,
)

from statgpt.common.data.sdmx.v21.member_selection import member_value_ids, time_range_bounds


def _selection(*values, dimension_id: str = "TIME_PERIOD") -> MemberSelection:
    return MemberSelection(values_for=Dimension(id=dimension_id), values=list(values))


def _range(start: str, end: str) -> RangePeriod:
    return RangePeriod(
        start=StartPeriod(is_inclusive=True, period=datetime.fromisoformat(start)),
        end=EndPeriod(is_inclusive=True, period=datetime.fromisoformat(end)),
    )


class TestMemberValueIds:
    def test_member_values_only(self):
        selection = _selection(
            MemberValue(value="G1"), MemberValue(value="GOY"), dimension_id="TRANSFORMATION"
        )

        assert member_value_ids(selection) == {"G1", "GOY"}
        assert time_range_bounds(selection) == (None, None)

    def test_empty_selection(self):
        assert member_value_ids(_selection()) == set()
        assert time_range_bounds(_selection()) == (None, None)

    def test_time_ranges_are_not_coded_values(self):
        selection = _selection(_range("1914-01-01T00:00:00", "2026-07-31T00:00:00"))

        assert member_value_ids(selection) == set()


class TestTimeRangeBounds:
    def test_range_period(self):
        selection = _selection(_range("1914-01-01T00:00:00", "2026-07-31T00:00:00"))

        assert time_range_bounds(selection) == ("1914-01-01", "2026-07-31")

    def test_after_period_sets_only_the_start(self):
        selection = _selection(
            AfterPeriod(is_inclusive=True, period=datetime(1914, 1, 1)),
        )

        assert time_range_bounds(selection) == ("1914-01-01", None)

    def test_before_period_sets_only_the_end(self):
        selection = _selection(
            BeforePeriod(is_inclusive=True, period=datetime(2026, 7, 31)),
        )

        assert time_range_bounds(selection) == (None, "2026-07-31")

    def test_several_ranges_collapse_to_the_outer_envelope(self):
        selection = _selection(
            _range("1960-01-01T00:00:00", "1979-12-31T00:00:00"),
            _range("1914-01-01T00:00:00", "1950-12-31T00:00:00"),
            _range("2000-01-01T00:00:00", "2026-07-31T00:00:00"),
        )

        assert time_range_bounds(selection) == ("1914-01-01", "2026-07-31")

    def test_is_inclusive_is_ignored(self):
        """Exclusive bounds are widened, never narrowed - widening cannot reject a valid query."""
        selection = _selection(
            RangePeriod(
                start=StartPeriod(is_inclusive=False, period=datetime(1914, 1, 1)),
                end=EndPeriod(is_inclusive=False, period=datetime(2026, 7, 31)),
            )
        )

        assert time_range_bounds(selection) == ("1914-01-01", "2026-07-31")

    def test_member_values_produce_no_bounds(self):
        selection = _selection(MemberValue(value="2020"), MemberValue(value="2021"))

        assert time_range_bounds(selection) == (None, None)


class TestUnsupportedSelectionValue:
    """The regression that mattered: one unknown value type must not abort the conversion."""

    def test_warns_and_skips_instead_of_raising(self, caplog):
        selection = _selection(
            MemberValue(value="G1"), SelectionValue(), dimension_id="TRANSFORMATION"
        )

        with caplog.at_level(logging.WARNING):
            ids = member_value_ids(selection)
            bounds = time_range_bounds(selection)

        assert ids == {"G1"}
        assert bounds == (None, None)
        assert "SelectionValue" in caplog.text
        assert "TRANSFORMATION" in caplog.text

    def test_warns_once_per_selection(self, caplog):
        selection = _selection(SelectionValue(), SelectionValue(), SelectionValue())

        with caplog.at_level(logging.WARNING):
            member_value_ids(selection)

        assert len(caplog.records) == 1
