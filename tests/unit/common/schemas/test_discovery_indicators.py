import pytest
from pydantic import ValidationError

from statgpt.common.schemas import (
    DiscoveryIndicator,
    parse_flat_indicators,
    read_indicators,
    render_flat_indicators,
)


def test_the_flat_form_round_trips() -> None:
    flat = "broad money (M2) (billions of Rp); policy rate (percent per annum); headcount"

    assert render_flat_indicators(parse_flat_indicators(flat)) == flat


def test_parsing_drops_empty_entries_and_keeps_repeated_ones() -> None:
    """The list says what was submitted; folding repeats is a consumer's business."""
    assert parse_flat_indicators(" M1 (USD);; ; M1 (USD) ") == [
        DiscoveryIndicator(name="M1", units="USD"),
        DiscoveryIndicator(name="M1", units="USD"),
    ]


def test_a_semicolon_inside_a_value_is_rendered_as_a_comma() -> None:
    """Otherwise the indicator would come back as two once the flat form is re-imported."""
    indicators = [DiscoveryIndicator(name="Exports; goods", units="USD; current")]

    flat = render_flat_indicators(indicators)

    assert flat == "Exports, goods (USD, current)"
    assert len(parse_flat_indicators(flat)) == 1


def test_a_stored_rows_json_is_read_into_indicators() -> None:
    assert read_indicators([{"name": "M1", "units": "USD"}, {"name": "M2"}]) == [
        DiscoveryIndicator(name="M1", units="USD"),
        DiscoveryIndicator(name="M2", units=""),
    ]


@pytest.mark.parametrize("name", ["", "   "])
def test_an_indicator_needs_a_name(name: str) -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        DiscoveryIndicator(name=name, units="USD")
