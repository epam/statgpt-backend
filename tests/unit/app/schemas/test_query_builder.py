from sdmx.model import common as sdmx_common

from statgpt.app.schemas.query_builder import LLMSelectionDimensionCandidate
from statgpt.common.data.base import VirtualDimensionValue
from statgpt.common.data.sdmx.common import DimensionCodeCategory, DimensionVirtualCodeCategory


def _code_candidate(
    index: int, dataset_id: str, code_id: str, name: str, dimension_alias: str
) -> LLMSelectionDimensionCandidate:
    return LLMSelectionDimensionCandidate(
        index=index,
        score=0.9,
        dataset_id=dataset_id,
        dimension_category=DimensionCodeCategory(
            code=sdmx_common.Code(id=code_id, name=name),
            locale='en',
            dimension_id='REF_AREA',
            dimension_name='Reference area',
            dimension_alias=dimension_alias,
        ),
    )


def _all_values_candidate(
    index: int,
    dataset_id: str,
    name: str,
    dimension_alias: str,
    code_id: str = 'ALL_COUNTRIES',
    dimension_id: str = 'REF_AREA',
    is_all_countries: bool = True,
) -> LLMSelectionDimensionCandidate:
    return LLMSelectionDimensionCandidate(
        index=index,
        score=1.0,
        dataset_id=dataset_id,
        dimension_category=DimensionVirtualCodeCategory(
            fixed_item=VirtualDimensionValue(id=code_id, name=name, description=None),
            dimension_id=dimension_id,
            dimension_name=dimension_alias,
            dimension_alias=dimension_alias,
        ),
        is_all_values=True,
        is_all_countries=is_all_countries,
    )


def _counterpart_all_values_candidate(
    index: int, dataset_id: str, name: str, dimension_alias: str
) -> LLMSelectionDimensionCandidate:
    return _all_values_candidate(
        index,
        dataset_id,
        name,
        dimension_alias,
        code_id='ALL_COUNTERPARTS',
        dimension_id='COUNTERPART_AREA',
        is_all_countries=False,
    )


def _candidates() -> list[LLMSelectionDimensionCandidate]:
    return [
        _code_candidate(0, 'ds_1', 'FR', 'France', 'Country/Reference area'),
        _code_candidate(1, 'ds_2', 'FR', 'France', 'Country/Reference Area'),
        _all_values_candidate(2, 'ds_1', 'All countries', 'Country/Reference area'),
        _all_values_candidate(3, 'ds_2', 'All countries (wildcard)', 'Country/Reference area'),
        _all_values_candidate(4, 'ds_3', 'All countries', 'Country/Reference Area'),
        _all_values_candidate(5, 'ds_4', 'All countries', 'Reporting country'),
        # same label as the country terms, but not a country dimension
        _counterpart_all_values_candidate(6, 'ds_1', 'All countries', 'Counterpart area'),
        _counterpart_all_values_candidate(7, 'ds_2', 'All counterparts', 'Counterpart Area'),
    ]


class TestCandidatesToLlmString:
    def test_dimension_aliases_differing_by_case_are_grouped(self):
        text = LLMSelectionDimensionCandidate.candidates_to_llm_string(_candidates())

        assert text.lower().count('## dimension: "country/reference area"') == 1
        assert text.lower().count('## dimension: "counterpart area"') == 1
        assert text.count('- France') == 1

    def test_all_countries_terms_are_shown_once_whatever_the_dimension_alias(self):
        text = LLMSelectionDimensionCandidate.candidates_to_llm_string(_candidates())

        assert text.count('system code: ALL_COUNTRIES') == 1
        # the only term of this dimension was merged into the first dataset's term
        assert '## dimension: "Reporting country"' not in text

    def test_other_all_values_terms_are_shown_once_per_dimension(self):
        text = LLMSelectionDimensionCandidate.candidates_to_llm_string(_candidates())

        assert text.count('system code: ALL_COUNTERPARTS') == 1

    def test_regular_code_is_not_merged_with_all_values_term(self):
        candidates = [
            _code_candidate(0, 'ds_1', 'ALL', 'All countries', 'Reference area'),
            _all_values_candidate(1, 'ds_2', 'All countries', 'Reference area'),
        ]

        text = LLMSelectionDimensionCandidate.candidates_to_llm_string(candidates)

        assert '(id: 0, system code: ALL)' in text
        assert '(id: 1, system code: ALL_COUNTRIES)' in text


class TestPropagateSelectionStatusToDuplicates:
    def test_all_countries_selection_is_propagated_regardless_of_label_and_alias(self):
        selected = LLMSelectionDimensionCandidate.propagate_selection_status_to_duplicates(
            candidates=_candidates(), selected_ids={'3'}
        )

        assert set(selected) == {'2', '3', '4', '5'}

    def test_all_values_selection_is_propagated_regardless_of_label_and_alias_case(self):
        selected = LLMSelectionDimensionCandidate.propagate_selection_status_to_duplicates(
            candidates=_candidates(), selected_ids={'7'}
        )

        assert set(selected) == {'6', '7'}

    def test_all_countries_and_other_all_values_selections_are_not_mixed(self):
        selected_countries = (
            LLMSelectionDimensionCandidate.propagate_selection_status_to_duplicates(
                candidates=_candidates(), selected_ids={'2'}
            )
        )
        selected_counterparts = (
            LLMSelectionDimensionCandidate.propagate_selection_status_to_duplicates(
                candidates=_candidates(), selected_ids={'6'}
            )
        )

        assert set(selected_countries) == {'2', '3', '4', '5'}
        assert set(selected_counterparts) == {'6', '7'}

    def test_regular_code_selection_is_propagated_across_alias_case(self):
        selected = LLMSelectionDimensionCandidate.propagate_selection_status_to_duplicates(
            candidates=_candidates(), selected_ids={'0'}
        )

        assert set(selected) == {'0', '1'}
