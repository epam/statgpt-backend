import pytest
from eval.dataset.rag.target_gathering_agent.schemas import DateFilter, Filter, FilterItem


@pytest.mark.parametrize(
    "current, target, expected_result",
    [
        (
            Filter(document_ids=[1, 2], filters_disjunction=[], last_n=None),
            Filter(document_ids=[2, 3], filters_disjunction=[], last_n=None),
            Filter(document_ids=[2], filters_disjunction=[], last_n=None),
        ),
        (
            Filter(document_ids=None, filters_disjunction=[], last_n=5),
            Filter(document_ids=None, filters_disjunction=[], last_n=5),
            Filter(document_ids=None, filters_disjunction=[], last_n=5),
        ),
        (
            Filter(document_ids=[1, 2], filters_disjunction=[], last_n=5),
            Filter(document_ids=None, filters_disjunction=[], last_n=5),
            Filter(document_ids=None, filters_disjunction=[], last_n=5),
        ),
        (
            Filter(document_ids=None, filters_disjunction=[], last_n=5),
            Filter(document_ids=[1, 2], filters_disjunction=[], last_n=5),
            Filter(document_ids=[1, 2], filters_disjunction=[], last_n=5),
        ),
        (
            Filter(document_ids=[1, 2], filters_disjunction=[], last_n=5),
            Filter(document_ids=[2, 3], filters_disjunction=[], last_n=None),
            Filter(document_ids=[2], filters_disjunction=[], last_n=5),
        ),
        (
            Filter(document_ids=[1, 2], filters_disjunction=[], last_n=None),
            Filter(document_ids=[2, 3], filters_disjunction=[], last_n=5),
            Filter(document_ids=[2, 3], filters_disjunction=[], last_n=5),
        ),
        (
            Filter(document_ids=[1, 2], filters_disjunction=[], last_n=5),
            Filter(document_ids=[2, 3], filters_disjunction=[], last_n=3),
            Filter(document_ids=[2, 3], filters_disjunction=[], last_n=3),
        ),
        (
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_date=DateFilter(start='2025-08-01', end='2025-09-01'))
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=None,
                last_n=2,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=None,
                last_n=2,
            ),
        ),
        (
            # 2 global date filters
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_date=DateFilter(start='2025-08-01', end='2025-09-01'))
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_date=DateFilter(start='2025-08-10', end='2025-09-10'))
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_date=DateFilter(start='2025-08-10', end='2025-09-01'))
                ],
                last_n=None,
            ),
        ),
        (
            # global date filter and list of publication date filters
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_date=DateFilter(start='2025-08-01', end='2025-09-01'))
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[FilterItem(publication_type='A')],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-01', end='2025-09-01'),
                        publication_type='A',
                    )
                ],
                last_n=None,
            ),
        ),
        (
            # global date filter and list of publication date filters
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_date=DateFilter(start='2025-08-01', end='2025-09-01'))
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_type='A'),
                    FilterItem(publication_type='B'),
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-01', end='2025-09-01'),
                        publication_type='A',
                    ),
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-01', end='2025-09-01'),
                        publication_type='B',
                    ),
                ],
                last_n=None,
            ),
        ),
        (
            # global date filter and list of publication date filters, change order of arguments
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_type='A'),
                    FilterItem(publication_type='B'),
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_date=DateFilter(start='2025-08-01', end='2025-09-01'))
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-01', end='2025-09-01'),
                        publication_type='A',
                    ),
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-01', end='2025-09-01'),
                        publication_type='B',
                    ),
                ],
                last_n=None,
            ),
        ),
        (
            # global date filter and list of publication date filters
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_date=DateFilter(start='2025-08-01', end='2025-09-01'))
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(
                        publication_type='A',
                        publication_date=DateFilter(start='2025-08-10', end='2025-09-10'),
                    ),
                    FilterItem(
                        publication_type='B',
                        publication_date=DateFilter(start='2025-07-01', end='2025-10-01'),
                    ),
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-10', end='2025-09-01'),
                        publication_type='A',
                    ),
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-01', end='2025-09-01'),
                        publication_type='B',
                    ),
                ],
                last_n=None,
            ),
        ),
        (
            # change order of arguments
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(
                        publication_type='A',
                        publication_date=DateFilter(start='2025-08-10', end='2025-09-10'),
                    ),
                    FilterItem(
                        publication_type='B',
                        publication_date=DateFilter(start='2025-07-01', end='2025-10-01'),
                    ),
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_date=DateFilter(start='2025-08-01', end='2025-09-01'))
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-10', end='2025-09-01'),
                        publication_type='A',
                    ),
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-01', end='2025-09-01'),
                        publication_type='B',
                    ),
                ],
                last_n=None,
            ),
        ),
        (
            # A and B are both publication date filters
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(
                        publication_type='A',
                        publication_date=DateFilter(start='2025-08-01', end='2025-09-01'),
                    ),
                    FilterItem(
                        publication_type='B',
                        publication_date=DateFilter(start='2025-07-01', end='2025-10-01'),
                    ),
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-01', end='2025-11-01'),
                        publication_type='B',
                    ),
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-10', end='2025-09-10'),
                        publication_type='C',
                    ),
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-01', end='2025-10-01'),
                        publication_type='B',
                    ),
                ],
                last_n=None,
            ),
        ),
    ],
)
def test_intersecting_filters(current: Filter, target: Filter, expected_result: Filter):
    result = current.bound_by_target_prefilter(target)
    assert result == expected_result


@pytest.mark.parametrize(
    "current, target",
    [
        (
            # no common date range
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_date=DateFilter(start='2025-08-01', end='2025-09-01'))
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(publication_date=DateFilter(start='2025-03-10', end='2025-04-10'))
                ],
                last_n=None,
            ),
        ),
        (
            # A and B are both publication date filters with no common publication types
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(
                        publication_type='A',
                        publication_date=DateFilter(start='2025-08-01', end='2025-09-01'),
                    ),
                ],
                last_n=None,
            ),
            Filter(
                document_ids=None,
                filters_disjunction=[
                    FilterItem(
                        publication_date=DateFilter(start='2025-08-01', end='2025-11-01'),
                        publication_type='B',
                    ),
                ],
                last_n=None,
            ),
        ),
    ],
)
def test_incorrect_intersecting_filters(current: Filter, target: Filter):
    with pytest.raises(ValueError):
        current.bound_by_target_prefilter(target)
