"""Tests for the links `Sdmx21DataSet` resolves: the explorer and citation links, and `dataset_url`."""

from types import SimpleNamespace

import pytest

from statgpt.common.data.base.config import DatasetCitation
from statgpt.common.data.sdmx.common import build_data_explorer_dataset_url
from statgpt.common.data.sdmx.v21.dataset import Sdmx21DataSet

_SHORT_URN = "IMF:CPI(1.0.0)"
_CITATION_URL = "https://data.example.org/cpi"
_DATASET_EXPLORER = "https://explorer.example.org/dataset"
_SOURCE_EXPLORER = "https://explorer.example.org/source"


def _dataset(
    *,
    dataset_explorer: str | None = None,
    source_explorer: str | None = None,
    citation_url: str | None = _CITATION_URL,
    use_data_explorer_for_dataset_url: bool = False,
) -> Sdmx21DataSet:
    """A dataset stub carrying only what the URL properties read."""
    dataset = Sdmx21DataSet.__new__(Sdmx21DataSet)
    dataset._config = SimpleNamespace(  # type: ignore[assignment]
        citation=DatasetCitation(url=citation_url),
        data_explorer_url_config=None,
        get_data_explorer_url=lambda: dataset_explorer,
    )
    dataset._datasource = SimpleNamespace(  # type: ignore[assignment]
        source_id="IMF",
        config=SimpleNamespace(
            use_data_explorer_for_dataset_url=use_data_explorer_for_dataset_url,
            data_explorer_url_config=None,
            get_data_explorer_url=lambda: source_explorer,
        ),
    )
    dataset._short_urn = _SHORT_URN
    return dataset


def _explorer_url(base: str) -> str:
    return build_data_explorer_dataset_url(base, _SHORT_URN, None)


class TestDataExplorerDatasetUrl:
    def test_dataset_explorer_overrides_the_data_source_one(self):
        dataset = _dataset(dataset_explorer=_DATASET_EXPLORER, source_explorer=_SOURCE_EXPLORER)

        assert dataset.data_explorer_dataset_url == _explorer_url(_DATASET_EXPLORER)

    def test_falls_back_to_the_data_source_explorer(self):
        dataset = _dataset(source_explorer=_SOURCE_EXPLORER)

        assert dataset.data_explorer_dataset_url == _explorer_url(_SOURCE_EXPLORER)

    def test_none_without_an_explorer(self):
        assert _dataset().data_explorer_dataset_url is None


class TestCitationUrl:
    def test_resolved_from_the_citation(self):
        assert _dataset().citation_url == _CITATION_URL

    def test_none_without_a_citation_url(self):
        assert _dataset(citation_url=None).citation_url is None


class TestDatasetUrl:
    @pytest.mark.parametrize("use_data_explorer_for_dataset_url", [True, False])
    def test_both_links_resolve_whichever_dataset_url_picks(
        self, use_data_explorer_for_dataset_url: bool
    ):
        dataset = _dataset(
            source_explorer=_SOURCE_EXPLORER,
            use_data_explorer_for_dataset_url=use_data_explorer_for_dataset_url,
        )

        assert dataset.data_explorer_dataset_url == _explorer_url(_SOURCE_EXPLORER)
        assert dataset.citation_url == _CITATION_URL

    def test_uses_the_explorer_when_configured(self):
        dataset = _dataset(source_explorer=_SOURCE_EXPLORER, use_data_explorer_for_dataset_url=True)

        assert dataset.dataset_url == _explorer_url(_SOURCE_EXPLORER)

    def test_uses_the_citation_by_default(self):
        dataset = _dataset(source_explorer=_SOURCE_EXPLORER)

        assert dataset.dataset_url == _CITATION_URL

    def test_falls_back_to_the_citation_without_an_explorer(self):
        dataset = _dataset(use_data_explorer_for_dataset_url=True)

        assert dataset.dataset_url == _CITATION_URL
