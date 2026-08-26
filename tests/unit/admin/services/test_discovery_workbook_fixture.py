"""End-to-end tests of the discovery write path over a committed workbook.

`test_discovery_upload.py` builds workbooks in memory to exercise parsing edge cases. This runs
the whole write path over one realistic file instead - real template headers, a plausible mix of
records - and asserts what each record becomes by the time it is publishable: parsed, validated,
and carrying the country filter axis a search will match on.

The fixture is the input and this file holds the expectations, so editing the workbook without
meaning to shows up here as a failure rather than as a silently weaker test. Regenerate the
workbook with `data/build_discovery_fixture.py`.
"""

from pathlib import Path

import pytest

from statgpt.admin.services.discovery_publisher import build_metadata, render_document_body
from statgpt.admin.services.discovery_upload import parse_discovery_file
from statgpt.admin.services.discovery_validation import DiscoveryValidator
from statgpt.common import schemas
from statgpt.common.schemas.discovery_dataset import DiscoveryDatasetBase
from statgpt.common.services.discovery_reference_area import SENTINEL, ground_reference_areas

FIXTURE = Path(__file__).parent / "data" / "discovery_index_multi_country.xlsx"

_CHANNEL = "statgpt-test-channel"
_EXPECTED_RECORD_COUNT = 16


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fixtures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture(scope="module")
def records() -> dict[str, DiscoveryDatasetBase]:
    """The fixture parsed through the real upload path, keyed by dataset id."""
    parsed = parse_discovery_file(FIXTURE.read_bytes(), FIXTURE.name, max_rows=1000)
    return {row.values["dataset_id"]: DiscoveryDatasetBase(**row.values) for row in parsed.rows}


@pytest.fixture(scope="module")
def areas(records: dict[str, DiscoveryDatasetBase]) -> dict[str, list[str]]:
    """The country filter axis each record publishes."""
    return {
        dataset_id: build_metadata(record, _CHANNEL, schemas.DiscoveryGrade.C).reference_area_values
        for dataset_id, record in records.items()
    }


@pytest.fixture(scope="module")
def channel_areas(areas: dict[str, list[str]]) -> list[str]:
    """Every country value the channel would hold once the fixture is indexed.

    This is what search grounds against, and the service accepts nothing outside it.
    """
    values: list[str] = []
    for record_areas in areas.values():
        values.extend(record_areas)
    return list(dict.fromkeys(values))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the file parses ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_the_fixture_parses_through_the_real_upload_path(records):
    assert len(records) == _EXPECTED_RECORD_COUNT


def test_the_template_headers_resolve_rather_than_the_positional_fallback(records):
    """The fixture uses the workbook's own column labels, so a header alias that stops resolving
    shows up here instead of being masked by the positional fallback."""
    record = records["JP_MONETARY_BASE"]

    assert record.agency == "Bank of Japan (BOJ)"
    assert record.missing_indicators.startswith("consumer prices")
    assert record.url == "https://example.com/jp/monetary-base"


def test_every_record_is_publishable(records):
    """A fixture record that cannot pass the indexing gate would test nothing downstream."""
    validator = DiscoveryValidator()

    failures = {
        dataset_id: validator.validate(record)
        for dataset_id, record in records.items()
        if validator.validate(record)
    }

    assert failures == {}


def test_every_record_renders_a_document_body(records):
    """The body is what retrieval searches over, so an empty one is a silent hole in the index."""
    for dataset_id, record in records.items():
        body = render_document_body(record)

        assert record.name in body, dataset_id
        assert "## Indicators coverage" in body, dataset_id


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the country axis, per case ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize(
    ("dataset_id", "expected"),
    [
        ("JP_MONETARY_BASE", ["Japan (JPN)"]),
        ("CA_LABOUR_FORCE", ["Canada (CAN)"]),
        ("BR_CPI", ["Brazil (BRA)"]),
        ("IN_MONEY", ["India (IND)"]),
        ("MY_MONETARY", ["Malaysia (MYS)"]),
        ("ID_BROAD_MONEY", ["Indonesia (IDN)"]),
        ("DE_NATIONAL_ACCOUNTS", ["Germany (DEU)"]),
    ],
)
def test_a_single_country_record_publishes_exactly_that_country(areas, dataset_id, expected):
    """No sentinel: a record scoped to one country must not match every other country's question."""
    assert areas[dataset_id] == expected


def test_a_multi_country_record_publishes_every_member(areas):
    """The case a whole-cell filter cannot match, and the reason the axis exists."""
    assert areas["ASEAN_MACRO"] == [
        "Indonesia (IDN)",
        "Malaysia (MYS)",
        "Thailand (THA)",
        "Philippines (PHL)",
        "Viet Nam (VNM)",
        "Singapore (SGP)",
    ]


@pytest.mark.parametrize("dataset_id", ["EA_HICP", "WORLD_AGGREGATES", "BIS_CREDIT"])
def test_a_record_with_no_country_scope_carries_the_sentinel(areas, dataset_id):
    """A group label, a world aggregate, and an empty cell. Each has to survive a country filter,
    or a question about Germany would silently lose the euro-area dataset."""
    assert SENTINEL in areas[dataset_id]


def test_a_group_label_is_also_published_as_its_own_value(areas):
    """So a question naming the euro area can filter to it directly."""
    assert areas["EA_HICP"] == ["Euro area", SENTINEL]


def test_an_empty_cell_publishes_only_the_sentinel(areas):
    assert areas["BIS_CREDIT"] == [SENTINEL]


def test_a_mixed_scope_record_publishes_its_country_and_the_sentinel(areas):
    """Part of the scope is enumerable and part is not, so both behaviors apply at once."""
    assert areas["ID_MIXED_SCOPE"] == ["Indonesia (IDN)", "ASEAN member states", SENTINEL]


def test_partner_countries_never_become_filter_values(areas):
    """A question about Japan's exports is for a record that *reports* Japan."""
    assert areas["TRADE_BILATERAL"] == ["Japan (JPN)", "Germany (DEU)"]


def test_the_verbatim_cell_survives_alongside_the_axis(records):
    """The parse is additive: nothing a submitter wrote is rewritten."""
    metadata = build_metadata(records["ASEAN_MACRO"], _CHANNEL, schemas.DiscoveryGrade.C)

    assert metadata.reference_area.startswith("Indonesia (IDN); Malaysia (MYS)")
    assert len(metadata.reference_area_values) == 6


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ what a search would narrow to ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _matching(areas: dict[str, list[str]], filter_values: list[str]) -> set[str]:
    """The records the service would return for a filter, matching an array field by containment."""
    return {
        dataset_id
        for dataset_id, record_areas in areas.items()
        if any(value in record_areas for value in filter_values)
    }


def test_a_question_about_one_country_reaches_its_records_and_the_unscoped_ones(
    areas, channel_areas
):
    """The whole point of the axis, asserted end to end over the fixture."""
    grounded = ground_reference_areas(["Malaysia"], channel_areas)

    assert grounded.matched_any
    matched = _matching(areas, grounded.values)

    # The Malaysian record, the multi-country record that includes Malaysia, and the records with
    # no country scope.
    assert {"MY_MONETARY", "ASEAN_MACRO"} <= matched
    assert {"EA_HICP", "WORLD_AGGREGATES", "BIS_CREDIT"} <= matched
    # Nothing scoped to another country.
    assert not {"JP_MONETARY_BASE", "BR_CPI", "CA_LABOUR_FORCE"} & matched


def test_a_country_no_record_covers_grounds_to_nothing(channel_areas):
    """The search then runs unfiltered and leaves the judge to reject what comes back, rather
    than filtering to the unscoped records alone."""
    grounded = ground_reference_areas(["Argentina"], channel_areas)

    assert grounded.values == []
    assert grounded.unmatched == ["Argentina"]


def test_an_iso_code_grounds_the_same_way_as_a_name(areas, channel_areas):
    """The published value carries both, which is what makes it self-grounding."""
    by_name = ground_reference_areas(["Germany"], channel_areas).values
    by_code = ground_reference_areas(["DEU"], channel_areas).values

    assert by_name == by_code
    assert "DE_NATIONAL_ACCOUNTS" in _matching(areas, by_code)


def test_a_bilateral_record_is_reached_by_its_reporter_not_its_partner(areas, channel_areas):
    """China appears in the record's text but must not be a filter value."""
    by_reporter = ground_reference_areas(["Japan"], channel_areas)
    by_partner = ground_reference_areas(["China"], channel_areas)

    assert "TRADE_BILATERAL" in _matching(areas, by_reporter.values)
    assert by_partner.unmatched == ["China"]


def test_two_countries_ground_to_a_union(areas, channel_areas):
    grounded = ground_reference_areas(["Japan", "Brazil"], channel_areas)
    matched = _matching(areas, grounded.values)

    assert {"JP_MONETARY_BASE", "JP_CPI", "JP_FX", "BR_CPI"} <= matched
    assert "CA_LABOUR_FORCE" not in matched


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the negative fields ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_the_fixture_carries_the_two_correct_rejection_cases(records):
    """Filtering cannot handle either: both records are in the right country, and retrieval ranks
    them *higher* for the very thing they exclude, because the exclusion names it. Only the judge
    can rule them out - so the fixture has to keep supplying them.
    """
    fx = records["JP_FX"]
    retail = records["CA_RETAIL"]

    assert "gross domestic product, GDP" in fx.missing_indicators
    assert "Quebec" in retail.excluded_regional_values
    # And both are otherwise perfectly reachable by a country filter.
    assert "Japan" in fx.reference_area
    assert "Canada" in retail.reference_area


def test_excluded_regions_reach_the_document_metadata(records):
    """The judge reads this field, so it has to travel with the document rather than only in the
    body, where a chunk that matched elsewhere would not carry it."""
    metadata = build_metadata(records["CA_RETAIL"], _CHANNEL, schemas.DiscoveryGrade.C)

    assert metadata.excluded_regional_values == "Quebec and Ontario are absent"
    assert metadata.missing_indicators.startswith("e-commerce sales")


def test_the_fixture_keeps_one_indicator_family_across_several_countries(records):
    """So a test can tell a country filter apart from text matching: 'broad money' alone cannot
    distinguish these four."""
    sharing = {
        dataset_id
        for dataset_id, record in records.items()
        if "broad money" in record.indicators_coverage.casefold()
    }

    assert {"IN_MONEY", "MY_MONETARY", "ID_BROAD_MONEY"} <= sharing
