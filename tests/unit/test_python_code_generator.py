import pytest

from statgpt.app.services.python_code_generator import (
    PYTHON_SDMX1_HEADER,
    _build_key_from_filters,
    _build_params_from_filters,
    generate_merged_python_code,
    generate_python_code_from_query,
)
from statgpt.common.schemas.query import (
    JsonComponentQuery,
    JsonQueryMetadata,
    JsonQueryOperator,
    JsonQueryWithMetadata,
)

_VALID_URN = "ESTAT:DF_COVID(1.0)"


def test_json_query_urn_derived_ids() -> None:
    query = JsonQueryWithMetadata(
        urn=_VALID_URN,
        filters=[],
        metadata=JsonQueryMetadata(
            country_dimension="A",
            indicator_dimensions=["B"],
            time_period_dimension="TIME_PERIOD",
        ),
    )
    assert query.agency_id == "ESTAT"
    assert query.resource_id == "DF_COVID"
    assert query.version == "1.0"


def test_json_query_with_metadata_legacy_body_parses() -> None:
    body = {
        "urn": "IMF.RES:WEO(9.0.0)",
        "sdmx1Source": "IMF_DATA",
        "metadata": {
            "countryDimension": "COUNTRY",
            "indicatorDimensions": ["INDICATOR"],
            "timePeriodDimension": "TIME_PERIOD",
        },
        "filters": [
            {
                "componentCode": "COUNTRY",
                "operator": "in",
                "values": ["USA"],
            },
            {
                "componentCode": "INDICATOR",
                "operator": "in",
                "values": ["NGDP_RPCH"],
            },
            {
                "componentCode": "TIME_PERIOD",
                "operator": "between",
                "values": ["2026-01-01", "2028-12-31"],
            },
        ],
    }
    q = JsonQueryWithMetadata.model_validate(body)
    assert q.metadata.time_period_dimension == "TIME_PERIOD"
    assert q.sdmx1_source == "IMF_DATA"


def test_build_key_from_filters_uses_dsd_order_and_empty_slots() -> None:
    filters = [
        JsonComponentQuery(component_code="D", operator=JsonQueryOperator.IN, values=["d1"]),
        JsonComponentQuery(component_code="A", operator=JsonQueryOperator.IN, values=["a1"]),
    ]
    order = ["A", "B", "C", "D"]
    key = _build_key_from_filters(filters, "TIME_PERIOD", order)
    assert key == "a1...d1"


def test_build_key_from_filters_skips_time_in_order_hint() -> None:
    filters = [
        JsonComponentQuery(component_code="A", operator=JsonQueryOperator.IN, values=["a1"]),
    ]
    order = ["A", "TIME_PERIOD", "B"]
    key = _build_key_from_filters(filters, "TIME_PERIOD", order)
    assert key == "a1."


def test_build_key_from_filters_legacy_filter_order_when_no_hint() -> None:
    filters = [
        JsonComponentQuery(component_code="D", operator=JsonQueryOperator.IN, values=["d1"]),
        JsonComponentQuery(component_code="A", operator=JsonQueryOperator.IN, values=["a1"]),
    ]
    key = _build_key_from_filters(filters, "TIME_PERIOD", None)
    assert key == "d1.a1"


def test_generate_python_code_uses_rest_key_dimension_codes() -> None:
    query = JsonQueryWithMetadata(
        urn=_VALID_URN,
        filters=[
            JsonComponentQuery(component_code="B", operator=JsonQueryOperator.IN, values=["b1"]),
            JsonComponentQuery(
                component_code="TIME_PERIOD",
                operator=JsonQueryOperator.BETWEEN,
                values=["2020", "2021"],
            ),
        ],
        metadata=JsonQueryMetadata(
            country_dimension="A",
            indicator_dimensions=["B"],
            time_period_dimension="TIME_PERIOD",
            rest_key_dimension_codes=["A", "B", "C"],
        ),
        sdmx1_source=None,
    )
    code = generate_python_code_from_query(query)
    assert 'key=".b1."' in code


def test_build_params_rejects_gt_lt_and_malformed_between() -> None:
    t = "TIME_PERIOD"
    with pytest.raises(ValueError, match="Exclusive time bound"):
        _build_params_from_filters(
            [JsonComponentQuery(component_code=t, operator=JsonQueryOperator.GT, values=["2020"])]
        )
    with pytest.raises(ValueError, match="Exclusive time bound"):
        _build_params_from_filters(
            [JsonComponentQuery(component_code=t, operator=JsonQueryOperator.LT, values=["2020"])]
        )
    with pytest.raises(ValueError, match="BETWEEN requires exactly two"):
        _build_params_from_filters(
            [
                JsonComponentQuery(
                    component_code=t,
                    operator=JsonQueryOperator.BETWEEN,
                    values=["2020", "2021", "2022"],
                )
            ]
        )
    with pytest.raises(ValueError, match="BETWEEN requires exactly two"):
        _build_params_from_filters(
            [
                JsonComponentQuery(
                    component_code=t, operator=JsonQueryOperator.BETWEEN, values=["2020"]
                )
            ]
        )


def test_build_params_in_single_period_maps_to_point_range() -> None:
    t = "TIME_PERIOD"
    params = _build_params_from_filters(
        [JsonComponentQuery(component_code=t, operator=JsonQueryOperator.IN, values=["2020"])]
    )
    assert params["startPeriod"] == "2020"
    assert params["endPeriod"] == "2020"
    assert params["detail"] == "full"


def test_build_params_in_multiple_values_rejected() -> None:
    with pytest.raises(ValueError, match="multiple values"):
        _build_params_from_filters(
            [
                JsonComponentQuery(
                    component_code="TIME_PERIOD",
                    operator=JsonQueryOperator.IN,
                    values=["2020", "2021"],
                )
            ]
        )


def _make_query(urn: str) -> JsonQueryWithMetadata:
    return JsonQueryWithMetadata(
        urn=urn,
        filters=[
            JsonComponentQuery(component_code="A", operator=JsonQueryOperator.IN, values=["a1"]),
        ],
        metadata=JsonQueryMetadata(
            country_dimension="A",
            indicator_dimensions=["B"],
            time_period_dimension="TIME_PERIOD",
        ),
    )


def test_generate_merged_python_code_single_query_has_header_and_no_suffix() -> None:
    code = generate_merged_python_code([_make_query(_VALID_URN)])

    assert code.startswith(PYTHON_SDMX1_HEADER + "\n\n")
    assert "# Dataset:" not in code
    assert "data_msg_1" not in code


def test_generate_merged_python_code_multi_query_separates_sections() -> None:
    urn_a = "ESTAT:DF_A(1.0)"
    urn_b = "ESTAT:DF_B(1.0)"
    code = generate_merged_python_code([_make_query(urn_a), _make_query(urn_b)])

    assert code.count(PYTHON_SDMX1_HEADER) == 1
    assert code.startswith(PYTHON_SDMX1_HEADER + "\n\n")
    assert f"# Dataset: {urn_a}" in code
    assert f"# Dataset: {urn_b}" in code
    assert "provider_1" in code and "provider_2" in code
