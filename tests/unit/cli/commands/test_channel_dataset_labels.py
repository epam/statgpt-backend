"""Tests that labelling a channel's datasets cannot abort the batch being labelled.

`channel reindex` builds a label for every dataset in the channel up front, so a single
dataset whose `details["urn"]` cannot be parsed must not stop the others from being submitted.
"""

import datetime
import uuid
from typing import Any

from statgpt.cli.commands.channel import _dataset_label, _get_urn_display
from statgpt.common.schemas import DataSet
from statgpt.common.schemas.dataset import Status

_NOW = datetime.datetime(2026, 1, 1)


def _dataset(title: str, details: dict[str, Any]) -> DataSet:
    return DataSet(
        id=1,
        id_=uuid.uuid5(uuid.NAMESPACE_URL, title),
        created_at=_NOW,
        updated_at=_NOW,
        title=title,
        description="",
        data_source_id=1,
        details=details,
        data_source=None,
        status=Status(status="online"),
    )


def test_a_valid_urn_is_labelled_by_its_short_form() -> None:
    dataset = _dataset(
        "Prices", {"urn": {"agency_id": "AG", "resource_id": "DF_A", "version": "1.0"}}
    )

    assert _dataset_label(dataset) == "AG:DF_A(1.0)"


def test_a_dataset_without_a_urn_falls_back_to_its_title() -> None:
    assert _dataset_label(_dataset("Prices", {})) == "Prices"


def test_an_unparseable_urn_does_not_raise(capsys) -> None:
    """A junk `urn` block used to raise ValidationError and abort the whole reindex."""
    dataset = _dataset("Prices", {"urn": {"nope": 1}})

    label = _dataset_label(dataset)

    assert label, "the dataset still needs a name in the summary"
    assert "validation error" in capsys.readouterr().out, "the malformed config is still reported"


def test_a_urn_that_is_not_a_mapping_does_not_raise() -> None:
    assert _get_urn_display(_dataset("Prices", {"urn": "AG:DF_A(1.0)"}))
