"""Tests for folding grouped discovery record counts into the stats schema."""

from statgpt.common.schemas import DiscoveryIndexingStatus as Indexing
from statgpt.common.schemas import DiscoveryValidationStatus as Validation
from statgpt.common.services.discovery_dataset import fold_stats


def test_every_status_is_present_even_when_no_record_is_in_it() -> None:
    """A caller renders the whole breakdown, so an absent status has to read as zero."""
    stats = fold_stats([(Validation.VALID, Indexing.INDEXED, "OECD", 5)])

    assert set(stats.by_validation_status) == set(Validation)
    assert set(stats.by_indexing_status) == set(Indexing)
    assert stats.by_validation_status[Validation.INVALID] == 0
    assert stats.by_indexing_status[Indexing.FAILED] == 0


def test_counts_are_summed_across_the_groups_of_each_status() -> None:
    """Grouping is by both statuses at once, so each breakdown folds several rows."""
    stats = fold_stats(
        [
            (Validation.VALID, Indexing.INDEXED, "OECD", 10),
            (Validation.VALID, Indexing.OUTDATED, "OECD", 3),
            (Validation.INVALID, Indexing.NEW, "IMF", 2),
            (Validation.NOT_VALIDATED, Indexing.NEW, "IMF", 4),
        ]
    )

    assert stats.total == 19
    assert stats.by_validation_status[Validation.VALID] == 13
    assert stats.by_indexing_status[Indexing.NEW] == 6


def test_agency_counts_are_summed_across_the_status_groups() -> None:
    """An agency spans status groups, so its count is spread over several rows."""
    stats = fold_stats(
        [
            (Validation.VALID, Indexing.INDEXED, "OECD", 10),
            (Validation.INVALID, Indexing.NEW, "OECD", 3),
            (Validation.VALID, Indexing.INDEXED, "IMF", 2),
        ]
    )

    assert stats.by_agency == {"OECD": 13, "IMF": 2}


def test_agencies_are_ordered_by_count_then_name() -> None:
    """The map doubles as a picker's source, so its order cannot depend on the query plan."""
    stats = fold_stats(
        [
            (Validation.VALID, Indexing.INDEXED, "Eurostat", 1),
            (Validation.VALID, Indexing.INDEXED, "IMF", 7),
            (Validation.VALID, Indexing.INDEXED, "ABS", 1),
            (Validation.VALID, Indexing.INDEXED, "OECD", 7),
        ]
    )

    assert list(stats.by_agency) == ["IMF", "OECD", "ABS", "Eurostat"]


def test_only_agencies_present_are_reported() -> None:
    """Agencies are not an enum, so there is nothing to seed the map with."""
    stats = fold_stats([(Validation.VALID, Indexing.INDEXED, "OECD", 5)])

    assert stats.by_agency == {"OECD": 5}


def test_a_channel_with_no_records_reports_zeros() -> None:
    stats = fold_stats([])

    assert stats.total == 0
    assert set(stats.by_validation_status.values()) == {0}
    assert set(stats.by_indexing_status.values()) == {0}
    assert stats.by_agency == {}
