"""Tests for resolving `Sdmx21DataSet.updated_at` from the citation's free-text value."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from statgpt.common.data.sdmx.v21.dataset import Sdmx21DataSet


def _dataset(last_updated: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(citation=SimpleNamespace(last_updated=last_updated)),
        entity_id="cpi",
    )


async def test_the_citation_date_is_parsed():
    updated_at = await Sdmx21DataSet.updated_at(_dataset("2023-06-15"), SimpleNamespace())  # type: ignore[arg-type]

    assert updated_at == datetime(2023, 6, 15)


@pytest.mark.parametrize("last_updated", ["Quarterly, when ready", "99999999999999999999"])
async def test_an_unparsable_citation_date_resolves_to_none(last_updated: str):
    updated_at = await Sdmx21DataSet.updated_at(_dataset(last_updated), SimpleNamespace())  # type: ignore[arg-type]

    assert updated_at is None
