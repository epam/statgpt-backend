import pytest
from pydantic import ValidationError

from statgpt.common.schemas import DiscoveryDatasetBase


def _record(dataset_id: str) -> DiscoveryDatasetBase:
    return DiscoveryDatasetBase(agency="Bank Indonesia (BI)", dataset_id=dataset_id)


@pytest.mark.parametrize(
    "dataset_id, reason",
    [
        ("X" * 32, "at most 31 characters"),
        ("IMF/IFS", "must not contain /"),
        ("'TABEL1_1", "apostrophe"),
        ("TABEL1_1'", "apostrophe"),
        ("History", "Excel reserves"),
        (" history ", "Excel reserves"),
    ],
)
def test_an_id_no_sheet_can_be_named_after_is_refused(dataset_id: str, reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        _record(dataset_id)


@pytest.mark.parametrize("dataset_id", ["X" * 31, "TABEL1_1", "it's", "History of prices"])
def test_an_id_a_sheet_can_be_named_after_is_accepted(dataset_id: str) -> None:
    assert _record(dataset_id).dataset_id == dataset_id
