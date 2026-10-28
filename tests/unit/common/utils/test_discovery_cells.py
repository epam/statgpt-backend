"""Tests for splitting the ';'-separated cells of a discovery record.

These arrays are what a retrieval request filters on, so a token that comes out wrong here is a
dataset that either cannot be found or is found for the wrong query.
"""

import pytest

from statgpt.common.utils import (
    FREQUENCY_VOCABULARY,
    compact_units,
    is_known_frequency,
    parse_frequencies,
    parse_reference_areas,
    split_cell,
)


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("France", ["France"]),
        ("France; Germany", ["France", "Germany"]),
        # Formatting, not values: a trailing separator and a doubled one carry no token.
        ("France;;Germany;", ["France", "Germany"]),
        ("  France  ;\tGermany\n", ["France", "Germany"]),
        # Internal runs collapse, so two spellings of one label are one value.
        ("Euro  area", ["Euro area"]),
        ("", []),
        ("   ", []),
        (";", []),
        # A value repeated in one cell would show up twice in the channel's dimensions.
        ("France; france; FRANCE", ["France"]),
    ],
)
def test_a_cell_splits_into_its_values(cell: str, expected: list[str]) -> None:
    assert split_cell(cell) == expected


@pytest.mark.parametrize(
    ("cell", "areas", "partners"),
    [
        ("France; Germany", ["France", "Germany"], []),
        # The label applies to its own token and to every token after it, not just the next one.
        (
            "France; partner countries: China; India; Japan",
            ["France"],
            ["China", "India", "Japan"],
        ),
        # The label can also be a token of its own.
        ("France; partner countries:; China", ["France"], ["China"]),
        ("Partner Countries: China", [], ["China"]),
        # A second label is a partner value, not another divider: the roles do not nest.
        (
            "France; partner countries: China; partner countries: India",
            ["France"],
            ["China", "partner countries: India"],
        ),
        ("", [], []),
    ],
)
def test_partner_countries_are_split_off(cell: str, areas: list[str], partners: list[str]) -> None:
    assert parse_reference_areas(cell) == (areas, partners)


@pytest.mark.parametrize("area", ["World", "Euro area", "European Union", "Advanced economies"])
def test_group_labels_are_values_in_their_own_right(area: str) -> None:
    """Nothing is expanded: a dataset about a group is not a dataset about a member of it."""
    assert parse_reference_areas(area) == ([area], [])


def test_frequencies_are_folded_onto_the_vocabularys_spelling() -> None:
    """Otherwise `annual` and `Annual` would be two filter values covering the same datasets."""
    assert parse_frequencies("annual; MONTHLY ; semi-ANNUAL") == [
        "Annual",
        "Monthly",
        "Semi-annual",
    ]


def test_a_frequency_outside_the_vocabulary_is_kept_as_submitted() -> None:
    """Such a record fails validation and is never published, so nothing may vanish silently."""
    assert parse_frequencies("Annual; Fortnightly") == ["Annual", "Fortnightly"]


@pytest.mark.parametrize("frequency", FREQUENCY_VOCABULARY)
def test_every_vocabulary_frequency_is_recognized_whatever_its_casing(frequency: str) -> None:
    assert is_known_frequency(frequency)
    assert is_known_frequency(frequency.upper())
    assert not is_known_frequency(f"{frequency} ish")


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        (["M1 (Rp billions)", "M2 (Rp billions)"], ("Rp billions", ["M1", "M2"])),
        # Nested parentheses belong to the unit: it is the whole trailing group.
        (["Rate (% (p.a.))", "Yield (% (p.a.))"], ("% (p.a.)", ["Rate", "Yield"])),
        # Parentheses inside a name stay with the name.
        (
            ["Broad money (M2) (Rp billions)", "M1 (Rp billions)"],
            ("Rp billions", ["Broad money (M2)", "M1"]),
        ),
        # Units that differ only in whitespace are one unit.
        (["M1 ( Rp  billions )", "M2 (Rp billions)"], ("Rp billions", ["M1", "M2"])),
    ],
)
def test_a_unit_every_entry_shares_is_factored_out(
    entries: list[str], expected: tuple[str, list[str]]
) -> None:
    assert compact_units(entries) == expected


@pytest.mark.parametrize(
    "entries",
    [
        ["M1 (Rp billions)", "M2 (USD millions)"],
        # Case-sensitive: `USD` and `usd` are kept apart, as submitted.
        ["M1 (USD)", "M2 (usd)"],
        ["M1 (Rp billions)", "M2"],
        ["M1 (Rp billions)", "M2 (Rp billions"],
        ["(Rp billions)", "M2 (Rp billions)"],
        ["M1 ()", "M2 ()"],
    ],
)
def test_a_list_without_one_shared_unit_is_left_as_it_is(entries: list[str]) -> None:
    assert compact_units(entries) == (None, entries)
