import pytest

from statgpt.common.schemas.query import (
    JsonComponentQuery,
    JsonQueryMetadata,
    JsonQueryOperator,
    JsonQueryWithMetadata,
)
from statgpt.common.schemas.record_id import (
    RecordId,
    build_sdmx_series_key,
    compose_record_id,
    record_id_of,
)

_KEY_ORDER = ["FREQ", "REF_AREA", "INDICATOR"]


def _filter(component_code: str, *values: str) -> JsonComponentQuery:
    return JsonComponentQuery(
        component_code=component_code, operator=JsonQueryOperator.IN, values=list(values)
    )


def _query(
    urn: str,
    filters: list[JsonComponentQuery],
    key_order: list[str] | None = _KEY_ORDER,
) -> JsonQueryWithMetadata:
    return JsonQueryWithMetadata(
        urn=urn,
        filters=filters,
        metadata=JsonQueryMetadata(
            country_dimension="REF_AREA",
            indicator_dimensions=["INDICATOR"],
            time_period_dimension="TIME_PERIOD",
            key_dimension_ids_in_dsd_order=key_order,
        ),
    )


def test_series_key_follows_dsd_order_not_filter_order():
    # Filters listed out of DSD order still produce the DSD-ordered key: this is what makes
    # the id stable regardless of how the filters happen to be listed.
    key = build_sdmx_series_key(
        [_filter("INDICATOR", "CPI"), _filter("FREQ", "A")],
        time_component="TIME_PERIOD",
        key_dimension_ids_in_dsd_order=_KEY_ORDER,
    )
    assert key == "A..CPI"


def test_series_key_joins_multiple_values_with_plus():
    key = build_sdmx_series_key(
        [_filter("REF_AREA", "FR", "DE")],
        time_component="TIME_PERIOD",
        key_dimension_ids_in_dsd_order=_KEY_ORDER,
    )
    assert key == ".FR+DE."


def test_series_key_skips_the_time_dimension():
    key = build_sdmx_series_key(
        [_filter("FREQ", "A"), _filter("TIME_PERIOD", "2020")],
        time_component="TIME_PERIOD",
        key_dimension_ids_in_dsd_order=_KEY_ORDER,
    )
    assert key == "A.."


def test_series_key_leaves_empty_slots_for_unfiltered_dimensions():
    key = build_sdmx_series_key(
        [_filter("D", "d1"), _filter("A", "a1")],
        time_component="TIME_PERIOD",
        key_dimension_ids_in_dsd_order=["A", "B", "C", "D"],
    )
    assert key == "a1...d1"


def test_series_key_falls_back_to_filter_order_without_hint():
    key = build_sdmx_series_key(
        [_filter("D", "d1"), _filter("A", "a1")],
        time_component="TIME_PERIOD",
        key_dimension_ids_in_dsd_order=None,
    )
    assert key == "d1.a1"


def test_compose_and_parse_are_inverse():
    record_id = RecordId(
        agency_id="IMF.RES", resource_id="ED", version="1.0.0", series_key="A.FR+DE.CPI"
    )
    assert record_id.compose() == "IMF.RES:ED(1.0.0)/A.FR+DE.CPI"
    assert RecordId.parse(record_id.compose()) == record_id


def test_parse_preserves_empty_series_key():
    parsed = RecordId.parse("IMF:CPI(1.0.0)/")
    assert parsed == RecordId(agency_id="IMF", resource_id="CPI", version="1.0.0", series_key="")


@pytest.mark.parametrize(
    "value",
    [
        "IMF:CPI(1.0.0)",  # no series-key separator
        "not-a-urn/A.B",  # dataflow ref not in AGENCY:RESOURCE(VERSION) form
        "IMF:CPI/A.B",  # missing version
    ],
)
def test_parse_rejects_malformed_ids(value: str):
    with pytest.raises(ValueError):
        RecordId.parse(value)


def test_compose_record_id_from_query():
    query = _query(
        "IMF.RES:ED(1.0.0)",
        [_filter("FREQ", "A"), _filter("REF_AREA", "FR", "DE")],
    )
    assert compose_record_id(query) == "IMF.RES:ED(1.0.0)/A.FR+DE."


def test_round_trip_search_id_back_to_the_same_record():
    """Definition of done: an id returned by a search reconstructs the same record.

    A search builds a query; its record id is fed back (as a follow-up call would) and parsed
    into the same dataflow reference and the same categorical filters that select the same
    series.
    """
    query = _query(
        "IMF.RES:ED(1.0.0)",
        [_filter("FREQ", "A"), _filter("INDICATOR", "CPI", "PPI")],
    )

    record_id = record_id_of(query)

    # The id names the exact dataflow the search returned.
    assert record_id.dataflow_ref == query.urn
    assert (record_id.agency_id, record_id.resource_id, record_id.version) == (
        query.agency_id,
        query.resource_id,
        query.version,
    )

    # Parsing the opaque string reconstructs the same components...
    reparsed = RecordId.parse(record_id.compose())
    assert reparsed == record_id

    # ...and the same categorical filters that select the same series.
    key_order = query.metadata.key_dimension_ids_in_dsd_order
    assert key_order is not None
    assert reparsed.to_component_filters(key_order) == query.filters


def test_id_is_stable_across_filter_ordering():
    # Same logical record described two ways yields the same id.
    a = _query("IMF:CPI(1.0.0)", [_filter("FREQ", "A"), _filter("INDICATOR", "CPI")])
    b = _query("IMF:CPI(1.0.0)", [_filter("INDICATOR", "CPI"), _filter("FREQ", "A")])
    assert compose_record_id(a) == compose_record_id(b)
