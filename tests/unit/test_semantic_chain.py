import pytest

from statgpt.chains.data_query.v2.query.indicator_selection.semantic.packed_indicators_selection import (
    PackedIndicatorsSelectionV2ChainFactory,
)
from statgpt.schemas.query_builder import DatasetDimQueries


@pytest.mark.parametrize(
    "query, target",
    (
        # invalid, single dataset
        ({'1': {}}, False),
        ({'1': {'indicator': []}}, False),
        ({'1': {'indicator': [], 'unit': []}}, False),
        # invalid, multiple datasets
        (
            {
                '1': {},
                '2': {},
            },
            False,
        ),
        (
            {
                '1': {},
                '2': {'indicator': []},
            },
            False,
        ),
        (
            {
                '1': {'indicator': []},
                '2': {'indicator': []},
            },
            False,
        ),
        (
            {
                '1': {'indicator': [], 'unit': []},
                '2': {'indicator': []},
            },
            False,
        ),
        # valid, single dataset
        ({'1': {'indicator': ['a']}}, True),
        ({'1': {'indicator': ['a', 'b']}}, True),
        ({'1': {'indicator': ['a'], 'unit': []}}, True),
        ({'1': {'indicator': ['a', 'b'], 'unit': []}}, True),
        ({'1': {'indicator': ['a'], 'unit': ['c']}}, True),
        ({'1': {'indicator': ['a', 'b'], 'unit': ['c', 'd']}}, True),
        # valid, multiple datasets
        (
            {
                '1': {'indicator': []},
                '2': {'indicator': ['b']},
            },
            True,
        ),
        (
            {
                '1': {'indicator': ['a']},
                '2': {'indicator': []},
            },
            True,
        ),
        (
            {
                '1': {'indicator': ['a']},
                '2': {'indicator': ['b']},
            },
            True,
        ),
        (
            {
                '1': {'indicator': ['a'], 'unit': ['c']},
                '2': {'indicator': ['d']},
            },
            True,
        ),
        (
            {
                '1': {'indicator': ['a'], 'unit': ['c']},
                '2': {'indicator': ['d'], 'unit': ['e']},
            },
            True,
        ),
        (
            {
                '1': {'indicator': []},
                '2': {'indicator': [], 'unit': ['e']},
            },
            True,
        ),
    ),
)
def test_is_dataset_dim_query_valid(query, target):
    query_parsed = DatasetDimQueries(queries=query)
    is_valid = query_parsed.is_valid()
    assert is_valid == target


@pytest.mark.parametrize(
    "queries_raw, target_queries",
    (
        # both qureies are valid, one query is not present
        (
            {
                "queries": {
                    'exact': {'1': {'indicator': ['e1', 'e2']}},
                    'child': {'1': {'indicator': ['c1']}},
                },
            },
            {'1': {'indicator': ['e1', 'e2']}},
        ),
        # both qureies are valid, one query is not present
        (
            {
                "queries": {
                    'exact': {'1': {'indicator': ['e1', 'e2'], 'unit': []}},
                    'child': {'1': {'indicator': ['c1']}},
                },
            },
            {'1': {'indicator': ['e1', 'e2'], 'unit': []}},
        ),
        # exact query is invalid
        (
            {
                "queries": {
                    'exact': {},
                    'child': {'1': {'indicator': ['c1']}},
                },
            },
            {'1': {'indicator': ['c1']}},
        ),
        (
            {
                "queries": {
                    'exact': {'1': {'indicator': []}},
                    'child': {'1': {'indicator': ['c1']}},
                },
            },
            {'1': {'indicator': ['c1']}},
        ),
        # child query is invalid
        (
            {
                "queries": {
                    'exact': {'1': {'indicator': ['e1', 'e2']}},
                    'child': {},
                },
            },
            {'1': {'indicator': ['e1', 'e2']}},
        ),
        (
            {
                "queries": {
                    'exact': {'1': {'indicator': ['e1', 'e2']}},
                    'child': {'1': {'indicator': []}},
                },
            },
            {'1': {'indicator': ['e1', 'e2']}},
        ),
        # both queries are invalid
        (
            {
                "queries": {
                    'exact': {},
                    'child': {},
                },
            },
            {},
        ),
        (
            {
                "queries": {
                    'exact': {},
                    'child': {'1': {}},
                },
            },
            {'1': {}},
        ),
    ),
)
def test_query_construction_llm_response_v2(queries_raw, target_queries):
    data = PackedIndicatorsSelectionV2ChainFactory.LLMResponse.model_validate(queries_raw)
    combined_queries = data.get_queries()
    assert combined_queries == target_queries
