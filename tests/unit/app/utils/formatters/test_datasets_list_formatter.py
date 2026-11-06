from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from statgpt.app.chains.datasets_meta._utils import _create_formatter_config
from statgpt.app.utils.formatters import DatasetsListFormatter
from statgpt.common.schemas.enums import (
    AvailableDatasetsHeaderFormat,
    AvailableDatasetsVersion,
    LocaleEnum,
)

_NOTE_START = "Note: some of the counted datasets are not fully onboarded"


def _dataset(name: str, provider: str) -> SimpleNamespace:
    return SimpleNamespace(
        entity_id=name.lower(),
        name=name,
        config=SimpleNamespace(
            is_official=True,
            citation=SimpleNamespace(
                provider=provider,
                provider_agency_names_with_fallback_to_provider=[provider],
            ),
        ),
    )


def _formatter(
    header: AvailableDatasetsHeaderFormat = AvailableDatasetsHeaderFormat.totals,
) -> DatasetsListFormatter:
    config = _create_formatter_config(
        version=AvailableDatasetsVersion.short,
        locale=LocaleEnum.EN,
        stats_header_format=header,
    )
    formatter = DatasetsListFormatter(config, auth_context=SimpleNamespace())  # type: ignore[arg-type]
    # The per-dataset rendering is not under test here: one line per dataset is enough.
    formatter._formatter = SimpleNamespace(  # type: ignore[assignment]
        format=AsyncMock(side_effect=lambda ds, indicator_count=None: f"- {ds.name}")
    )
    return formatter


async def _format(formatter: DatasetsListFormatter, **kwargs) -> str:
    return await formatter.format(
        [_dataset("CPI", "IMF"), _dataset("GDP", "IMF")],
        sort_by_name=True,
        add_stats=True,
        group_by_provider=True,
        indicator_counts={"cpi": 10, "gdp": 5},
        **kwargs,
    )


async def test_without_metadata_only_datasets_nothing_changes():
    text = await _format(_formatter(), metadata_only_counts={})

    assert _NOTE_START not in text
    assert "Total datasets: 2\nTotal providers: 1\nTotal number of indicators: 15" in text
    assert "Total datasets from this provider: 2" in text
    assert "Other providers" not in text


async def test_metadata_only_datasets_are_counted_not_listed():
    text = await _format(_formatter(), metadata_only_counts={"IMF": 3, "BdE": 4, "ECB": 4})

    assert text.startswith(_NOTE_START)
    # Merged into the dataset and provider counts; the indicators stay the queryable ones.
    assert "Total datasets: 13\nTotal providers: 3\nTotal number of indicators: 15" in text
    assert "### Provider: IMF\n\nTotal datasets from this provider: 5" in text
    assert "### Other providers\n\n- BdE: 4 datasets\n- ECB: 4 datasets" in text
    assert text.count("\n- ") == 4  # CPI, GDP and the two other providers


@pytest.mark.parametrize("with_indicators", [True, False])
async def test_agencies_header_merges_metadata_only_agencies(with_indicators: bool):
    formatter = _formatter(AvailableDatasetsHeaderFormat.agencies)

    text = await formatter.format(
        [_dataset("CPI", "IMF")],
        add_stats=True,
        indicator_counts={"cpi": 10} if with_indicators else None,
        metadata_only_counts={"BdE": 4},
    )

    assert text.startswith(_NOTE_START)
    assert "I have access to 5 datasets" in text
    assert "provided by 2 agencies, including: BdE, IMF." in text
