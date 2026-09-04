import pytest
from pydantic import ValidationError

from statgpt.app.schemas.query import AppJsonQuery, AppJsonQueryWithMetadata
from statgpt.app.schemas.service import GeneratePythonCodeRequest
from statgpt.app.services.python_code_generator import (
    PYTHON_SDMX1_HEADER,
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

_VALID_URN = "IMF:WEO(1.0)"


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
    assert query.agency_id == "IMF"
    assert query.resource_id == "WEO"
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


def test_generate_python_code_uses_key_dimension_ids_in_dsd_order() -> None:
    query = AppJsonQueryWithMetadata(
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
            key_dimension_ids_in_dsd_order=["A", "B", "C"],
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


def _make_query(urn: str, disabled: bool = False) -> AppJsonQueryWithMetadata:
    return AppJsonQueryWithMetadata(
        urn=urn,
        filters=[
            JsonComponentQuery(component_code="A", operator=JsonQueryOperator.IN, values=["a1"]),
        ],
        metadata=JsonQueryMetadata(
            country_dimension="A",
            indicator_dimensions=["B"],
            time_period_dimension="TIME_PERIOD",
        ),
        disabled=disabled,
    )


def test_generate_merged_python_code_single_query_has_header_and_no_suffix() -> None:
    code = generate_merged_python_code([_make_query(_VALID_URN)])

    assert code.startswith(PYTHON_SDMX1_HEADER + "\n\n")
    assert "# Dataset:" not in code
    assert "data_msg_1" not in code


def test_generate_merged_python_code_multi_query_separates_sections() -> None:
    urn_a = "IMF:DF_A(1.0)"
    urn_b = "IMF:DF_B(1.0)"
    code = generate_merged_python_code([_make_query(urn_a), _make_query(urn_b)])

    assert code.count(PYTHON_SDMX1_HEADER) == 1
    assert code.startswith(PYTHON_SDMX1_HEADER + "\n\n")
    assert f"# Dataset: {urn_a}" in code
    assert f"# Dataset: {urn_b}" in code
    assert "provider_1" in code and "provider_2" in code


def _make_query_with_unrepresentable_time(urn: str) -> AppJsonQueryWithMetadata:
    """A query with an exclusive `gt` time bound, which has no SDMX REST representation."""
    return AppJsonQueryWithMetadata(
        urn=urn,
        filters=[
            JsonComponentQuery(component_code="A", operator=JsonQueryOperator.IN, values=["a1"]),
            JsonComponentQuery(
                component_code="TIME_PERIOD", operator=JsonQueryOperator.GT, values=["2020"]
            ),
        ],
        metadata=JsonQueryMetadata(
            country_dimension="A",
            indicator_dimensions=["B"],
            time_period_dimension="TIME_PERIOD",
        ),
    )


def test_generate_merged_python_code_single_unrepresentable_query_yields_placeholder() -> None:
    urn = "IMF:DF_BAD(1.0)"
    code = generate_merged_python_code([_make_query_with_unrepresentable_time(urn)])

    assert code.startswith(PYTHON_SDMX1_HEADER + "\n\n")
    assert f"# Unable to generate a reproducible sdmx1 snippet for {urn}." in code
    # The failure is isolated: no partial/broken request body is emitted.
    assert "sdmx.Client" not in code


def test_generate_merged_python_code_skips_only_the_unrepresentable_query() -> None:
    good_urn = "IMF:DF_GOOD(1.0)"
    bad_urn = "IMF:DF_BAD(1.0)"
    code = generate_merged_python_code(
        [_make_query(good_urn), _make_query_with_unrepresentable_time(bad_urn)]
    )

    # The good dataset still renders a complete snippet...
    assert f"# Dataset: {good_urn}" in code
    assert 'sdmx.Client("IMF")' in code
    # ...while the bad one degrades to a placeholder instead of dropping the whole snippet.
    assert f"# Dataset: {bad_urn}" in code
    assert f"# Unable to generate a reproducible sdmx1 snippet for {bad_urn}." in code


def test_generate_merged_python_code_drops_dataset_with_any_excluded_filter() -> None:
    excluded_query = AppJsonQueryWithMetadata(
        urn="IMF:DF_X(1.0)",
        filters=[
            JsonComponentQuery(component_code="A", operator=JsonQueryOperator.IN, values=["a1"]),
            JsonComponentQuery(
                component_code="B",
                operator=JsonQueryOperator.EXCLUDED,
                values=["b_unavailable"],
            ),
        ],
        metadata=JsonQueryMetadata(
            country_dimension="A",
            indicator_dimensions=["B"],
            time_period_dimension="TIME_PERIOD",
        ),
    )
    code = generate_merged_python_code([_make_query(_VALID_URN), excluded_query])
    assert "DF_X" not in code
    assert "b_unavailable" not in code
    # Only one query survives, so the single-query branch is taken (no `# Dataset:` headers).
    assert "# Dataset:" not in code
    assert "provider_1" not in code
    # The surviving query renders as a complete sdmx1 snippet.
    assert 'provider = sdmx.Client("IMF")' in code
    assert '"IMF,WEO,1.0"' in code
    assert 'key="a1"' in code
    assert "'detail': 'full'" in code


def test_generate_merged_python_code_returns_header_only_when_all_excluded() -> None:
    excluded_query = AppJsonQueryWithMetadata(
        urn="IMF:DF_X(1.0)",
        filters=[
            JsonComponentQuery(
                component_code="B",
                operator=JsonQueryOperator.EXCLUDED,
                values=["b1"],
            ),
        ],
        metadata=JsonQueryMetadata(
            country_dimension="A",
            indicator_dimensions=["B"],
            time_period_dimension="TIME_PERIOD",
        ),
    )
    code = generate_merged_python_code([excluded_query])
    assert code == PYTHON_SDMX1_HEADER


def test_generate_merged_python_code_drops_disabled_query() -> None:
    disabled_urn = "IMF:DF_DISABLED(1.0)"
    code = generate_merged_python_code(
        [_make_query(_VALID_URN), _make_query(disabled_urn, disabled=True)]
    )
    assert "DF_DISABLED" not in code
    # Only one query survives, so the single-query branch is taken (no `# Dataset:` headers).
    assert "# Dataset:" not in code
    assert '"IMF,WEO,1.0"' in code


def test_generate_merged_python_code_returns_header_only_when_all_disabled() -> None:
    code = generate_merged_python_code([_make_query(_VALID_URN, disabled=True)])
    assert code == PYTHON_SDMX1_HEADER


def test_generate_python_code_request_rejects_all_disabled_queries() -> None:
    with pytest.raises(ValidationError, match="All queries are disabled"):
        GeneratePythonCodeRequest(queries=[_make_query(_VALID_URN, disabled=True)])


def test_generate_python_code_request_accepts_partially_disabled_queries() -> None:
    request = GeneratePythonCodeRequest(
        queries=[_make_query(_VALID_URN), _make_query("IMF:DF_DISABLED(1.0)", disabled=True)]
    )
    assert len(request.queries) == 2


def test_app_json_query_disabled_defaults_to_false_and_is_serialized() -> None:
    query = _make_query(_VALID_URN)
    assert query.disabled is False
    assert query.model_dump(by_alias=True)["disabled"] is False


def test_app_json_query_with_metadata_from_query_serializes_disabled() -> None:
    common_query = JsonQueryWithMetadata(
        urn=_VALID_URN,
        filters=[],
        metadata=JsonQueryMetadata(
            country_dimension="A",
            indicator_dimensions=["B"],
            time_period_dimension="TIME_PERIOD",
        ),
    )
    app_query = AppJsonQueryWithMetadata.from_query(
        query=common_query, metadata=common_query.metadata
    )
    assert app_query.model_dump(by_alias=True)["disabled"] is False


def test_app_json_query_with_metadata_from_common_round_trips() -> None:
    common_query = JsonQueryWithMetadata(
        urn=_VALID_URN,
        filters=[
            JsonComponentQuery(component_code="A", operator=JsonQueryOperator.IN, values=["a1"]),
        ],
        metadata=JsonQueryMetadata(
            country_dimension="A",
            indicator_dimensions=["B"],
            time_period_dimension="TIME_PERIOD",
            key_dimension_ids_in_dsd_order=["A", "B"],
        ),
        sdmx1_source="IMF_DATA",
    )
    app_query = AppJsonQueryWithMetadata.from_common(common_query)
    assert app_query.disabled is False
    # `from_common` preserves the common fields, adds `disabled`, and derives the record id.
    assert app_query.model_dump() == {
        **common_query.model_dump(),
        "disabled": False,
        "record_id": "IMF:WEO(1.0)/a1.",
    }

    dump = app_query.model_dump(by_alias=True)
    assert dump["disabled"] is False
    assert dump["sdmx1Source"] == "IMF_DATA"
    assert dump["metadata"]["keyDimensionIdsInDsdOrder"] == ["A", "B"]
    assert dump["recordId"] == "IMF:WEO(1.0)/a1."


def test_app_json_query_rejects_non_bool_disabled() -> None:
    with pytest.raises(ValidationError):
        AppJsonQuery.model_validate({"urn": _VALID_URN, "filters": [], "disabled": "banana"})
