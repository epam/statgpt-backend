from sdmx.model import common as sdmx_common

from statgpt.app.schemas.query_builder import LLMSelectionDimensionCandidate
from statgpt.common.data.base import VirtualDimensionValue
from statgpt.common.data.sdmx.common import DimensionCodeCategory, DimensionVirtualCodeCategory

_ALL_COUNTRIES_LABEL = LLMSelectionDimensionCandidate.all_countries_label


def _code_candidate(
    index: int,
    dataset_id: str,
    code_id: str,
    name: str,
    dimension_alias: str,
    is_country_dimension: bool = True,
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
        is_country_dimension=is_country_dimension,
    )


def _all_values_candidate(
    index: int,
    dataset_id: str,
    name: str,
    dimension_alias: str,
    code_id: str = 'ALL_COUNTRIES',
    dimension_id: str = 'REF_AREA',
    is_country_dimension: bool = True,
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
        is_country_dimension=is_country_dimension,
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
        is_country_dimension=False,
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


def _issue_705_candidates() -> list[LLMSelectionDimensionCandidate]:
    """The first 'all countries' term is of the dataset with a rare country dimension alias."""
    return [
        _code_candidate(0, 'bis', 'US', 'United States', 'Reference area'),
        _code_candidate(1, 'imf_1', 'USA', 'United States', 'Country/Reference area'),
        _code_candidate(2, 'imf_2', 'USA', 'United States', 'Country/Reference area'),
        _code_candidate(3, 'imf_2', 'FRA', 'France', 'Country/Reference Area'),
        _all_values_candidate(
            4,
            'bis',
            'All countries individually (wildcard) - select for per-country breakdown',
            'Reference area',
        ),
        _all_values_candidate(
            5,
            'imf_1',
            'All countries - must be selected when query explicitly asks for all countries',
            'Country/Reference area',
        ),
        _all_values_candidate(6, 'imf_2', 'All countries', 'Country/Reference Area'),
    ]


def _terms_by_dimension(text: str) -> dict[str, list[str]]:
    """Map the dimensions shown to the LLM to their terms."""
    res = {}
    for block in text.split('## dimension: ')[1:]:
        header, *lines = block.strip().splitlines()
        res[header.strip('"')] = [line for line in lines if line]
    return res


class TestCandidatesToLlmString:
    def test_dimension_aliases_differing_by_case_are_grouped(self):
        text = LLMSelectionDimensionCandidate.candidates_to_llm_string(_candidates())

        assert text.lower().count('## dimension: "country/reference area"') == 1
        assert text.lower().count('## dimension: "counterpart area"') == 1
        assert text.count('- France') == 1

    def test_all_countries_terms_are_shown_once_whatever_the_dimension_alias(self):
        text = LLMSelectionDimensionCandidate.candidates_to_llm_string(_candidates())

        assert text.count('system code: ALL_COUNTRIES') == 1
        # the only term of this dimension was merged into the main country dimension term
        assert '## dimension: "Reporting country"' not in text

    def test_all_countries_term_is_shown_with_common_label(self):
        text = LLMSelectionDimensionCandidate.candidates_to_llm_string(_issue_705_candidates())

        assert text.count(_ALL_COUNTRIES_LABEL) == 1
        assert 'per-country breakdown' not in text
        assert 'explicitly asks' not in text

    def test_all_countries_term_is_shown_in_the_main_country_dimension(self):
        text = LLMSelectionDimensionCandidate.candidates_to_llm_string(_issue_705_candidates())

        terms = _terms_by_dimension(text)
        # the dimension of the first dataset's term has the fewest country terms
        assert terms['Reference area'] == ['- United States (id: 0, system code: US)']
        assert f'- {_ALL_COUNTRIES_LABEL} (id: 4, system code: ALL_COUNTRIES)' in (
            terms['Country/Reference area']
        )

    def test_all_countries_term_without_country_codes_is_shown_in_the_most_common_dimension(
        self,
    ):
        candidates = [c for c in _issue_705_candidates() if c.is_all_values]

        text = LLMSelectionDimensionCandidate.candidates_to_llm_string(candidates)

        assert _terms_by_dimension(text) == {
            'Country/Reference area': [
                f'- {_ALL_COUNTRIES_LABEL} (id: 4, system code: ALL_COUNTRIES)'
            ],
        }

    def test_all_values_terms_of_other_dimensions_keep_their_label(self):
        candidates = [
            _counterpart_all_values_candidate(0, 'ds_1', 'All counterparts', 'Counterpart area'),
        ]

        text = LLMSelectionDimensionCandidate.candidates_to_llm_string(candidates)

        assert text == (
            '## dimension: "Counterpart area"\n'
            '- All counterparts (id: 0, system code: ALL_COUNTERPARTS)'
        )

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
