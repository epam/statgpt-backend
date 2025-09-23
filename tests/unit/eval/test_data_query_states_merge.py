import pytest

from eval.schemas.test_case import DataQueryStateMerger
from statgpt.schemas.query_builder import DatasetDimensionTermNameType, QueryBuilderAgentState


@pytest.mark.parametrize(
    "input_list_of_dataset_2_dim_2_term_id_2_name, expected_dataset_2_dim_2_term_id_2_name, correct",
    [
        (
            [
                {"dataset_1": {"COUNTRY": {"USA": "United States"}}},
                # add new dataset
                {"dataset_2": {"COUNTRY": {"CAN": "Canada"}}},
            ],
            {
                "dataset_1": {"COUNTRY": {"USA": "United States"}},
                "dataset_2": {"COUNTRY": {"CAN": "Canada"}},
            },
            True,
        ),
        (
            [
                {
                    "dataset_1": {
                        "SERIES": {"GDP_MP": "Gross domestic product at market prices"},
                        "REF_AREA": {"CA": "Canada", "FR": "France"},
                        "ADJUSTMENT": {
                            "Y": "Calendar and seasonally adjusted data",
                            "N": "Not adjusted",
                        },
                        "FREQ": {"Q": "Quarterly"},
                    },
                    "dataset_2": {
                        "INDICATOR": {"GDP_PC_CUR_US": "GDP per capita (current US$)"},
                        "COUNTRY": {"CAN": "Canada"},
                        "FREQ": {"A": "Annual"},
                    },
                },
                {
                    # add new indicator
                    "dataset_1": {
                        "SERIES": {"GDP_PC_GROWTH_PERC": "GDP per capita growth (annual %)"},
                    },
                    # add existing indicator and new country
                    "dataset_2": {
                        "INDICATOR": {"GDP_PC_CUR_US": "GDP per capita (current US$)"},
                        "COUNTRY": {"FRA": "France"},
                    },
                },
                {
                    # add new dimension
                    "dataset_2": {
                        "INDICATOR": {"GDP_PC_CUR_US": "GDP per capita (current US$)"},
                        "COUNTRY": {"CAN": "Canada"},
                        "FREQ": {"A": "Annual"},
                        "BREAKDOWN": {"TOTAL": "Total Economy"},
                    },
                },
            ],
            {
                "dataset_1": {
                    "SERIES": {
                        "GDP_MP": "Gross domestic product at market prices",
                        "GDP_PC_GROWTH_PERC": "GDP per capita growth (annual %)",
                    },
                    "REF_AREA": {"CA": "Canada", "FR": "France"},
                    "ADJUSTMENT": {
                        "Y": "Calendar and seasonally adjusted data",
                        "N": "Not adjusted",
                    },
                    "FREQ": {"Q": "Quarterly"},
                },
                "dataset_2": {
                    "INDICATOR": {"GDP_PC_CUR_US": "GDP per capita (current US$)"},
                    "COUNTRY": {"CAN": "Canada", "FRA": "France"},
                    "FREQ": {"A": "Annual"},
                    "BREAKDOWN": {"TOTAL": "Total Economy"},
                },
            },
            True,
        ),
        (
            [
                {"dataset_1": {"COUNTRY": {"CAN": "Canada"}}},
                # another state with conflicting term name
                {"dataset_1": {"COUNTRY": {"CAN": "The Canada"}}},
            ],
            None,
            False,
        ),
    ],
)
def test_data_query_states_merge(
    input_list_of_dataset_2_dim_2_term_id_2_name: list[DatasetDimensionTermNameType],
    expected_dataset_2_dim_2_term_id_2_name: DatasetDimensionTermNameType,
    correct: bool,
):
    data_query_states = [
        QueryBuilderAgentState(dimension_id_to_name=x)
        for x in input_list_of_dataset_2_dim_2_term_id_2_name
    ]
    merger = DataQueryStateMerger(data_query_states)

    if correct:
        dataset_2_dim_2_term_id_2_name = merger.merge_dimension_id_to_name()
        assert dataset_2_dim_2_term_id_2_name == expected_dataset_2_dim_2_term_id_2_name
    else:
        with pytest.raises(ValueError):
            _ = merger.merge_dimension_id_to_name()
