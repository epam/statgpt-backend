import pytest

from common.data.base.query import DataSetAvailabilityQuery, Query, QueryOperator


@pytest.mark.parametrize(
    "query1, query2, target",
    (
        (
            {'indicator': ['a']},
            {'indicator': []},
            {'indicator': []},
        ),
        (
            {'indicator': []},
            {'indicator': ['a']},
            {'indicator': []},
        ),
        (
            {'indicator': ['a', 'b']},
            {'indicator': ['b', 'c']},
            {'indicator': ['b']},
        ),
        (
            {'indicator': ['a', 'b'], 'currency': ['USD', 'EUR']},
            {'indicator': ['b', 'c']},
            {'indicator': ['b']},
        ),
        (
            {'indicator': ['b', 'c']},
            {'indicator': ['a', 'b'], 'currency': ['USD', 'EUR']},
            {'indicator': ['b']},
        ),
        (
            {'indicator': ['a', 'b'], 'currency': ['USD', 'EUR']},
            {'indicator': ['b', 'c'], 'currency': ['USD', 'BYN']},
            {'indicator': ['b'], 'currency': ['USD']},
        ),
        (
            {'indicator': ['a', 'b'], 'currency': ['USD', 'EUR']},
            {'indicator': ['b', 'c'], 'currency': ['BYN']},
            {'indicator': ['b'], 'currency': []},
        ),
        (
            {'indicator': ['b', 'c'], 'currency': ['BYN']},
            {'indicator': ['a', 'b'], 'currency': ['USD', 'EUR']},
            {'indicator': ['b'], 'currency': []},
        ),
        (
            {'indicator': ['b', 'c'], 'currency': ['BYN']},
            {'indicator': ['a', 'b'], 'currency': []},
            {'indicator': ['b'], 'currency': []},
        ),
        (
            {'indicator': ['a', 'b'], 'currency': []},
            {'indicator': ['b', 'c'], 'currency': ['BYN']},
            {'indicator': ['b'], 'currency': []},
        ),
    ),
)
def test_query_filtration(query1, query2, target):

    def _input2query(input_dict):
        return DataSetAvailabilityQuery(
            dimensions_queries_dict={
                k: Query(values=v, operator=QueryOperator.IN) for k, v in input_dict.items()
            }
        )

    query1 = _input2query(query1)
    query2 = _input2query(query2)
    target = _input2query(target)

    out = query1.filter(query2)

    assert out == target
