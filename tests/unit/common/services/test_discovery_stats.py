"""Tests for folding grouped discovery record counts into the stats schema."""

from statgpt.common.schemas import DiscoveryIndexingStatus as Indexing
from statgpt.common.schemas import DiscoveryValidationStatus as Validation
from statgpt.common.services.discovery_dataset import fold_stats


def test_every_status_is_present_even_when_no_record_is_in_it() -> None:
    """A caller renders the whole breakdown, so an absent status has to read as zero."""
    stats = fold_stats([(Validation.VALID, Indexing.INDEXED, 5)])

    assert set(stats.by_validation_status) == set(Validation)
    assert set(stats.by_indexing_status) == set(Indexing)
    assert stats.by_validation_status[Validation.INVALID] == 0
    assert stats.by_indexing_status[Indexing.FAILED] == 0


def test_counts_are_summed_across_the_groups_of_each_status() -> None:
    """Grouping is by both statuses at once, so each breakdown folds several rows."""
    stats = fold_stats(
        [
            (Validation.VALID, Indexing.INDEXED, 10),
            (Validation.VALID, Indexing.OUTDATED, 3),
            (Validation.INVALID, Indexing.NEW, 2),
            (Validation.NOT_VALIDATED, Indexing.NEW, 4),
        ]
    )

    assert stats.total == 19
    assert stats.by_validation_status[Validation.VALID] == 13
    assert stats.by_indexing_status[Indexing.NEW] == 6


def test_a_channel_with_no_records_reports_zeros() -> None:
    stats = fold_stats([])

    assert stats.total == 0
    assert set(stats.by_validation_status.values()) == {0}
    assert set(stats.by_indexing_status.values()) == {0}
