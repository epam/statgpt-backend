"""Tests for the reference-area filter axis shared by the discovery write and read paths."""

import pytest

from statgpt.common.services.discovery_reference_area import (
    SENTINEL,
    ground_reference_areas,
    parse_reference_area,
    value_aliases,
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ parsing column A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_a_single_country_becomes_one_value_and_no_sentinel():
    """The common case. A record scoped to one country must not match every other country."""
    assert parse_reference_area("Indonesia (IDN)") == ["Indonesia (IDN)"]


def test_every_country_of_a_multi_country_cell_becomes_its_own_value():
    """The case a whole-cell filter cannot match, and the reason this axis exists."""
    cell = "Indonesia (IDN); Malaysia (MYS); Thailand (THA)"

    assert parse_reference_area(cell) == [
        "Indonesia (IDN)",
        "Malaysia (MYS)",
        "Thailand (THA)",
    ]


def test_a_group_label_is_kept_as_a_value_and_also_gets_the_sentinel():
    """'Euro area' is what a question about the euro area filters to, and the sentinel is what
    keeps the record reachable from a question about one of its member countries."""
    assert parse_reference_area("Euro area") == ["Euro area", SENTINEL]


def test_an_empty_cell_yields_only_the_sentinel():
    """A record with no stated scope must survive every country filter, not none of them."""
    assert parse_reference_area("") == [SENTINEL]
    assert parse_reference_area("   ") == [SENTINEL]


def test_a_mixed_cell_keeps_its_countries_and_still_gets_the_sentinel():
    """Part of the scope is enumerable and part is not, so both behaviors are needed."""
    assert parse_reference_area("Indonesia (IDN); Euro area") == [
        "Indonesia (IDN)",
        "Euro area",
        SENTINEL,
    ]


def test_partner_countries_are_not_filter_values():
    """A question about Japan's exports is for a record that *reports* Japan.

    Every entry from the marker onwards is the counterparty list, including the entry the marker
    itself sits in.
    """
    cell = "Japan (JPN); Germany (DEU); partner countries: China; United States; European Union"

    assert parse_reference_area(cell) == ["Japan (JPN)", "Germany (DEU)"]


def test_the_partner_marker_is_matched_regardless_of_case_and_spacing():
    assert parse_reference_area("Japan (JPN);  PARTNER COUNTRIES : China") == ["Japan (JPN)"]


def test_whitespace_is_normalized_and_empty_entries_are_dropped():
    """The published value has to equal the one the read path grounds to, so both sides fold the
    same way. A trailing ';' is a normal thing to find in a hand-filled cell."""
    assert parse_reference_area("  Indonesia   (IDN) ;; ; ") == ["Indonesia (IDN)"]


def test_duplicates_collapse_so_an_unchanged_cell_renders_one_stable_list():
    """A record must not be republished because a value repeated."""
    assert parse_reference_area("Japan (JPN); Japan (JPN)") == ["Japan (JPN)"]


@pytest.mark.parametrize(
    "cell",
    [
        "World",
        "Advanced economies",
        "Sub-Saharan Africa",
    ],
)
def test_an_aggregate_label_always_carries_the_sentinel(cell: str):
    assert SENTINEL in parse_reference_area(cell)


def test_a_lowercase_code_still_counts_as_a_country():
    """The workbook asks for an uppercase ISO code, but a submitter's casing must not silently
    turn a country record into a sentinel-only one."""
    assert parse_reference_area("indonesia (idn)") == ["indonesia (idn)"]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ aliases ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_a_value_is_matched_by_its_name_its_code_and_its_whole_text():
    """Why the publisher keeps the name and the code together: the value grounds itself, with no
    country-name table - and this repository has none."""
    aliases = value_aliases("Indonesia (IDN)")

    assert "indonesia" in aliases
    assert "idn" in aliases
    assert "indonesia (idn)" in aliases


def test_a_group_label_is_matched_by_its_own_text():
    assert value_aliases("Euro area") == {"euro area"}


def test_the_sentinel_has_no_aliases():
    """It is never something a user names; it is unioned in unconditionally."""
    assert value_aliases(SENTINEL) == set()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ grounding ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_AVAILABLE = ["Indonesia (IDN)", "Japan (JPN)", "Euro area", SENTINEL]


def test_a_country_name_grounds_to_the_published_value():
    grounded = ground_reference_areas(["Japan"], _AVAILABLE)

    assert "Japan (JPN)" in grounded.values
    assert grounded.unmatched == []
    assert grounded.matched_any


def test_an_iso_code_grounds_to_the_same_value():
    assert "Japan (JPN)" in ground_reference_areas(["JPN"], _AVAILABLE).values


def test_matching_ignores_case_and_surrounding_whitespace():
    assert "Indonesia (IDN)" in ground_reference_areas(["  indonesia "], _AVAILABLE).values


def test_the_sentinel_joins_the_union_whenever_anything_matched():
    """So a country filter never excludes a record whose scope could not be pinned to countries."""
    grounded = ground_reference_areas(["Japan"], _AVAILABLE)

    assert grounded.values == ["Japan (JPN)", SENTINEL]


def test_several_countries_all_ground_and_are_unioned():
    grounded = ground_reference_areas(["Japan", "Indonesia"], _AVAILABLE)

    assert grounded.values == ["Japan (JPN)", "Indonesia (IDN)", SENTINEL]


def test_a_country_the_channel_has_no_records_for_grounds_to_nothing():
    """The result is empty rather than a filter of the sentinel alone.

    Filtering on the sentinel alone would narrow the search to the records with no country scope,
    which is the opposite of what a request about Brazil asked for. An empty result tells the
    caller to search unfiltered and let the judge reject what comes back.
    """
    grounded = ground_reference_areas(["Brazil"], _AVAILABLE)

    assert grounded.values == []
    assert grounded.unmatched == ["Brazil"]
    assert not grounded.matched_any


def test_a_partly_grounded_request_keeps_what_matched_and_reports_what_did_not():
    grounded = ground_reference_areas(["Japan", "Brazil"], _AVAILABLE)

    assert grounded.values == ["Japan (JPN)", SENTINEL]
    assert grounded.unmatched == ["Brazil"]


def test_only_values_the_channel_holds_are_ever_returned():
    """The service types a filterable field as a `Literal` of the values present, so an
    ungrounded value fails the whole retrieval request rather than matching nothing."""
    grounded = ground_reference_areas(["Japan"], ["Indonesia (IDN)"])

    assert grounded.values == []


def test_the_sentinel_is_omitted_when_the_channel_does_not_hold_it():
    """Every record resolves to a country, so no document carries the sentinel - and sending it
    would break the request."""
    grounded = ground_reference_areas(["Japan"], ["Japan (JPN)", "Indonesia (IDN)"])

    assert grounded.values == ["Japan (JPN)"]


def test_no_countries_at_all_grounds_to_an_unfiltered_search():
    grounded = ground_reference_areas([], _AVAILABLE)

    assert grounded.values == []
    assert grounded.unmatched == []


def test_blank_entities_are_neither_matched_nor_reported():
    grounded = ground_reference_areas(["", "   "], _AVAILABLE)

    assert grounded.values == []
    assert grounded.unmatched == []


def test_duplicate_entities_ground_once():
    grounded = ground_reference_areas(["Japan", "JPN", "japan"], _AVAILABLE)

    assert grounded.values == ["Japan (JPN)", SENTINEL]


def test_a_group_label_named_directly_grounds_to_that_label():
    """The label survives parsing as a value precisely so this works."""
    grounded = ground_reference_areas(["Euro area"], _AVAILABLE)

    assert grounded.values == ["Euro area", SENTINEL]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ round trip ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize(
    ("cell", "question_country"),
    [
        ("Indonesia (IDN)", "Indonesia"),
        ("Indonesia (IDN); Malaysia (MYS); Thailand (THA)", "Malaysia"),
        ("Japan (JPN); Germany (DEU); partner countries: China", "Germany"),
        ("indonesia (idn)", "IDN"),
    ],
)
def test_what_the_publisher_writes_is_what_the_search_grounds_to(cell: str, question_country: str):
    """The one invariant that holds the two halves together.

    If a published value and a grounded value ever disagree, the filter matches nothing and the
    feature fails silently - no error, just no results.
    """
    published = parse_reference_area(cell)

    grounded = ground_reference_areas([question_country], published)

    assert grounded.matched_any
    assert set(grounded.values) <= set(published)
