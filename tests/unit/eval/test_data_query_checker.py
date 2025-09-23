import numpy as np
import pytest

from eval.checkers.indicator_selection_checker import DataQueryChecker
from eval.schemas.test_case import DataQueryTerms
from statgpt.schemas.query_builder import DatasetDimensionTermNameType


@pytest.mark.parametrize(
    "target, received, macro_recall, macro_precision",
    [
        # ---------------- No Target ---------------- #
        (
            {},
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name'}},
            },
            np.nan,
            np.nan,
        ),
        # ---------------- Single Dimensions ---------------- #
        (
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name'}},
            },
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name'}},
            },
            1.0,
            1.0,
        ),
        (
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name'}},
            },
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name_different'}},
            },
            0.0,
            0.0,
        ),
        (
            # did not receive dataset_1:dim_1 dimension -> hence nan in precision
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name'}},
            },
            {
                'dataset_2': {'dim_1': {'term_1': 'term_1_name'}},
            },
            0.0,
            np.nan,
        ),
        (
            # did not receive dataset_1:dim_1 dimension -> hence nan in precision
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name'}},
            },
            {
                'dataset_1': {'dim_2': {'term_1': 'term_1_name'}},
            },
            0.0,
            np.nan,
        ),
        (
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name', 'term_2': 'term_2_name'}},
            },
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name'}},
            },
            0.5,
            1.0,
        ),
        (
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name', 'term_2': 'term_2_name'}},
            },
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name', 'term_2': 'term_2_name'}},
            },
            1.0,
            1.0,
        ),
        (
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name', 'term_2': 'term_2_name'}},
            },
            {
                'dataset_1': {'dim_1': {'term_1': 'term_1_name', 'term_3': 'term_3_name'}},
            },
            0.5,
            0.5,
        ),
        # ---------------- Multiple Dimensions ---------------- #
        (
            {
                'dataset_1': {
                    'dim_1': {'DIM1.T1': 'DIM1.T1.name'},
                    'dim_2': {'DIM2.T1': 'DIM2.T1.name'},
                },
            },
            {
                'dataset_1': {
                    'dim_1': {'DIM1.T1': 'DIM1.T1.name'},
                    'dim_2': {'DIM2.T1': 'DIM2.T1.name'},
                },
            },
            1.0,
            1.0,
        ),
        (
            # one target dimension is missing
            {
                'dataset_1': {
                    'dim_1': {'DIM1.T1': 'DIM1.T1.name'},
                    'dim_2': {'DIM2.T1': 'DIM2.T1.name'},
                },
            },
            {
                'dataset_1': {
                    'dim_1': {'DIM1.T1': 'DIM1.T1.name'},
                },
            },
            0.5,
            1.0,
        ),
        (
            {
                'dataset_1': {
                    'dim_1': {'DIM1.T1': 'DIM1.T1.name'},
                    'dim_2': {'DIM2.T1': 'DIM2.T1.name'},
                },
            },
            {
                'dataset_1': {
                    'dim_1': {'DIM1.T1': 'DIM1.T1.name'},
                    'dim_2': {},
                },
            },
            0.5,
            1.0,
        ),
        # TODO: add more tests
    ],
)
def test_data_query_checker(
    target: DatasetDimensionTermNameType,
    received: DatasetDimensionTermNameType,
    macro_recall: float,
    macro_precision: float,
):
    dqt_target = DataQueryTerms(raw=target)
    dqt_received = DataQueryTerms(raw=received)
    checker = DataQueryChecker()
    res = checker._compute(dqt_target, dqt_received)

    assert res.agg_metrics.agg_recall == pytest.approx(macro_recall, nan_ok=True)
    assert res.agg_metrics.agg_precision == pytest.approx(macro_precision, nan_ok=True)
