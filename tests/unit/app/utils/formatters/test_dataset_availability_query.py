import pytest

from statgpt.app.utils.formatters import (
    DatasetAvailabilityQueryFormatter,
    DatasetAvailabilityQueryFormatterConfig,
)
from statgpt.common.data.base import DimensionQuery, QueryOperator


class _FakeCategoricalDimension:
    """The slice of `CategoricalDimension` that `_format_dimension_query` reads."""

    name = 'Country'

    @staticmethod
    def name_by_query_id(query_id: str) -> str:
        return {'FR': 'France'}[query_id]


def _render(dim_query: DimensionQuery, format_values_as_list: bool) -> str:
    formatter = DatasetAvailabilityQueryFormatter(
        config=DatasetAvailabilityQueryFormatterConfig(
            format_values_as_list=format_values_as_list, add_value_ids=True
        ),
        auth_context=None,  # type: ignore[arg-type]
    )
    return formatter._format_dimension_query(
        dimension_id='REF_AREA',
        dim_query=dim_query,
        categorical_dimensions={'REF_AREA': _FakeCategoricalDimension()},  # type: ignore[dict-item]
        datetime_dimensions={},
        indicators=set(),
    )


@pytest.mark.parametrize('format_values_as_list', [True, False])
def test_all_values_query_is_rendered_as_wildcard(format_values_as_list: bool):
    dim_query = DimensionQuery(dimension_id='REF_AREA', values=[], operator=QueryOperator.ALL)

    assert _render(dim_query, format_values_as_list) == '\t* _Country_: **\\***'


def test_explicit_values_are_rendered_as_list():
    dim_query = DimensionQuery(dimension_id='REF_AREA', values=['FR'], operator=QueryOperator.IN)

    assert _render(dim_query, format_values_as_list=True) == '\t* _Country_:\n\t\t* **[FR] France**'
