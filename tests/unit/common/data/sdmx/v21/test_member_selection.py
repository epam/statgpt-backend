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

from statgpt.common.data.sdmx.v21.member_selection import read_member_selection


def _selection(*values: SelectionValue, dimension_id: str = "TIME_PERIOD") -> MemberSelection:
    return MemberSelection(values_for=Dimension(id=dimension_id), values=list(values))


def _range(start: str, end: str) -> RangePeriod:
    return RangePeriod(
        start=StartPeriod(is_inclusive=True, period=datetime.fromisoformat(start)),
        end=EndPeriod(is_inclusive=True, period=datetime.fromisoformat(end)),
    )


class TestCodedValues:
    def test_member_values_only(self) -> None:
        selection = _selection(
            MemberValue(value="G1"), MemberValue(value="GOY"), dimension_id="TRANSFORMATION"
        )

        values = read_member_selection(selection)

        assert values.coded_values == {"G1", "GOY"}
        assert not values.has_time_range

    def test_empty_selection(self) -> None:
        values = read_member_selection(_selection())

        assert values.coded_values == set()
        assert not values.has_time_range

    def test_time_ranges_are_not_coded_values(self) -> None:
        selection = _selection(_range("1914-01-01T00:00:00", "2026-07-31T00:00:00"))

        assert read_member_selection(selection).coded_values == set()

    def test_coded_values_and_a_time_range_coexist(self) -> None:
        selection = _selection(
            MemberValue(value="2020"), _range("1914-01-01T00:00:00", "2026-07-31T00:00:00")
        )

        values = read_member_selection(selection)

        assert values.coded_values == {"2020"}
        assert (values.time_range_start, values.time_range_end) == ("1914-01-01", "2026-07-31")


class TestTimeRangeBounds:
    def test_range_period(self) -> None:
        selection = _selection(_range("1914-01-01T00:00:00", "2026-07-31T00:00:00"))

        values = read_member_selection(selection)

        assert (values.time_range_start, values.time_range_end) == ("1914-01-01", "2026-07-31")
        assert values.has_time_range

    def test_after_period_sets_only_the_start(self) -> None:
        selection = _selection(
            AfterPeriod(is_inclusive=True, period=datetime(1914, 1, 1)),
        )

        values = read_member_selection(selection)

        assert (values.time_range_start, values.time_range_end) == ("1914-01-01", None)
        assert values.has_time_range

    def test_before_period_sets_only_the_end(self) -> None:
        selection = _selection(
            BeforePeriod(is_inclusive=True, period=datetime(2026, 7, 31)),
        )

        values = read_member_selection(selection)

        assert (values.time_range_start, values.time_range_end) == (None, "2026-07-31")
        assert values.has_time_range

    def test_several_ranges_collapse_to_the_outer_envelope(self) -> None:
        selection = _selection(
            _range("1960-01-01T00:00:00", "1979-12-31T00:00:00"),
            _range("1914-01-01T00:00:00", "1950-12-31T00:00:00"),
            _range("2000-01-01T00:00:00", "2026-07-31T00:00:00"),
        )

        values = read_member_selection(selection)

        assert (values.time_range_start, values.time_range_end) == ("1914-01-01", "2026-07-31")

    def test_is_inclusive_is_ignored(self) -> None:
        """Exclusive bounds are widened, never narrowed - widening cannot reject a valid query."""
        selection = _selection(
            RangePeriod(
                start=StartPeriod(is_inclusive=False, period=datetime(1914, 1, 1)),
                end=EndPeriod(is_inclusive=False, period=datetime(2026, 7, 31)),
            )
        )

        values = read_member_selection(selection)

        assert (values.time_range_start, values.time_range_end) == ("1914-01-01", "2026-07-31")

    def test_member_values_produce_no_bounds(self) -> None:
        selection = _selection(MemberValue(value="2020"), MemberValue(value="2021"))

        assert not read_member_selection(selection).has_time_range


class TestUnsupportedSelectionValue:
    """The regression that mattered: one unknown value type must not abort the conversion."""

    def test_warns_and_skips_instead_of_raising(self, caplog) -> None:
        selection = _selection(
            MemberValue(value="G1"), SelectionValue(), dimension_id="TRANSFORMATION"
        )

        with caplog.at_level(logging.WARNING):
            values = read_member_selection(selection)

        assert values.coded_values == {"G1"}
        assert not values.has_time_range
        assert "SelectionValue" in caplog.text
        assert "TRANSFORMATION" in caplog.text

    def test_warns_once_per_selection(self, caplog) -> None:
        selection = _selection(SelectionValue(), SelectionValue(), SelectionValue())

        with caplog.at_level(logging.WARNING):
            read_member_selection(selection)

        assert len(caplog.records) == 1
