"""Tests for converting an SDMX 2.1 availability response into a `DataSetAvailabilityQuery`."""

import os
from datetime import datetime

import sdmx
from sdmx.message import StructureMessage
from sdmx.model.common import Dimension, EndPeriod, StartPeriod
from sdmx.model.v21 import (
    Annotation,
    ConstraintRole,
    ConstraintRoleType,
    ContentConstraint,
    CubeRegion,
    MemberSelection,
    MemberValue,
    RangePeriod,
)

from statgpt.common.data.base import QueryOperator
from statgpt.common.data.quanthub.v21.dataset import QuanthubSdmx21DataSet
from statgpt.common.data.sdmx.v21.dataset import Sdmx21DataSet

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
OECD_FIXTURE = os.path.join(
    DATA_DIR, "availableconstraint_OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0.xml"
)


def _dataset(cls=Sdmx21DataSet):
    """A dataset stub carrying only what `_availability_result_to_query` reads."""
    dataset = cls.__new__(cls)
    dataset._virtual_dimensions = {}
    return dataset


def _availability_message(*, members: dict, annotations: list | None = None) -> StructureMessage:
    constraint = ContentConstraint(
        id="CC",
        role=ConstraintRole(role=ConstraintRoleType.actual),
        data_content_region=[
            CubeRegion(
                member={
                    Dimension(id=dim_id): MemberSelection(
                        values_for=Dimension(id=dim_id), values=list(values)
                    )
                    for dim_id, values in members.items()
                }
            )
        ],
        annotations=annotations or [],
    )
    message = StructureMessage()
    message.add(constraint)
    return message


def _range(start: str, end: str) -> RangePeriod:
    return RangePeriod(
        start=StartPeriod(is_inclusive=True, period=datetime.fromisoformat(start)),
        end=EndPeriod(is_inclusive=True, period=datetime.fromisoformat(end)),
    )


class TestOecdTimeRangeFixture:
    """The reported crash: OECD returns `<common:TimeRange>` for `TIME_PERIOD`."""

    def test_conversion_succeeds(self):
        message = sdmx.read_sdmx(OECD_FIXTURE)

        result = _dataset()._availability_result_to_query(message)

        assert result.time_period_start == "1914-01-01"
        assert result.time_period_end == "2026-07-31"

    def test_time_period_stays_out_of_the_dimension_queries(self):
        message = sdmx.read_sdmx(OECD_FIXTURE)

        result = _dataset()._availability_result_to_query(message)

        assert "TIME_PERIOD" not in result

    def test_other_dimensions_survive_the_time_range(self):
        """The core symptom: one unreadable member used to take down the whole cube region."""
        message = sdmx.read_sdmx(OECD_FIXTURE)

        result = _dataset()._availability_result_to_query(message)

        assert set(result.dimensions_queries_dict) == {
            "REF_AREA",
            "FREQ",
            "METHODOLOGY",
            "MEASURE",
            "UNIT_MEASURE",
            "EXPENDITURE",
            "ADJUSTMENT",
            "TRANSFORMATION",
        }
        transformation = result.dimensions_queries_dict["TRANSFORMATION"]
        assert transformation.operator == QueryOperator.IN
        assert transformation.values == ["G1", "GOY", "GY", "_Z"]
        assert len(result.dimensions_queries_dict["REF_AREA"].values) == 55


class TestCodedDimensions:
    def test_values_are_sorted(self):
        message = _availability_message(
            members={"FREQ": [MemberValue(value="Q"), MemberValue(value="A")]}
        )

        result = _dataset()._availability_result_to_query(message)

        assert result.dimensions_queries_dict["FREQ"].values == ["A", "Q"]

    def test_dimension_without_members_still_gets_an_empty_query(self):
        message = _availability_message(members={"FREQ": []})

        result = _dataset()._availability_result_to_query(message)

        assert result.dimensions_queries_dict["FREQ"].values == []
        assert result.dimensions_queries_dict["FREQ"].operator == QueryOperator.IN

    def test_no_constraints_yields_an_empty_query(self):
        result = _dataset()._availability_result_to_query(StructureMessage())

        assert result.dimensions_queries_dict == {}
        assert result.time_period_start is None
        assert result.time_period_end is None


class TestQuanthubAnnotationOverride:
    """The QuantHub subclass falls back to vendor annotations only when nothing was derived."""

    def test_derived_time_range_wins_over_annotations(self):
        message = _availability_message(
            members={
                "FREQ": [MemberValue(value="A")],
                "TIME_PERIOD": [_range("1914-01-01T00:00:00", "2026-07-31T00:00:00")],
            },
            annotations=[
                Annotation(id="time_period_start", title="1999-01-01"),
                Annotation(id="time_period_end", title="2001-12-31"),
            ],
        )

        result = _dataset(QuanthubSdmx21DataSet)._availability_result_to_query(message)

        assert result.time_period_start == "1914-01-01"
        assert result.time_period_end == "2026-07-31"

    def test_annotations_are_still_used_when_no_range_is_present(self):
        message = _availability_message(
            members={"FREQ": [MemberValue(value="A")]},
            annotations=[
                Annotation(id="time_period_start", title="1999-01-01"),
                Annotation(id="time_period_end", title="2001-12-31"),
            ],
        )

        result = _dataset(QuanthubSdmx21DataSet)._availability_result_to_query(message)

        assert result.time_period_start == "1999-01-01"
        assert result.time_period_end == "2001-12-31"
