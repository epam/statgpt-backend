import pytest
from pydantic import ValidationError

from statgpt.common.schemas import (
    DiscoveryIndicator,
    dump_indicators_json,
    parse_indicators_json,
    read_indicators,
    render_flat_indicators,
)


def test_the_json_form_round_trips() -> None:
    """What the flat text could not keep apart: a name in parentheses, units holding them."""
    indicators = [
        DiscoveryIndicator(name="broad money (M2)", units=""),
        DiscoveryIndicator(name="CPI", units="Index (2015=100)"),
        DiscoveryIndicator(name="Exports; goods", units="USD"),
    ]

    assert parse_indicators_json(dump_indicators_json(indicators)) == indicators


@pytest.mark.parametrize("cell", ["", "   "])
def test_a_blank_cell_is_an_empty_list(cell: str) -> None:
    assert parse_indicators_json(cell) == []


def test_the_flat_text_of_older_exports_is_refused() -> None:
    with pytest.raises(ValidationError):
        parse_indicators_json("broad money (M2) (billions of Rp); policy rate (percent)")


def test_a_semicolon_inside_a_value_is_rendered_as_a_comma() -> None:
    """The flat text joins indicators with ';', so one inside a value would read as two."""
    indicators = [DiscoveryIndicator(name="Exports; goods", units="USD; current")]

    assert render_flat_indicators(indicators) == "Exports, goods (USD, current)"


def test_a_stored_rows_json_is_read_into_indicators() -> None:
    assert read_indicators([{"name": "M1", "units": "USD"}, {"name": "M2"}]) == [
        DiscoveryIndicator(name="M1", units="USD"),
        DiscoveryIndicator(name="M2", units=""),
    ]


@pytest.mark.parametrize("name", ["", "   "])
def test_an_indicator_needs_a_name(name: str) -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        DiscoveryIndicator(name=name, units="USD")
