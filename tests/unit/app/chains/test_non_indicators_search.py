"""Unit tests for the 'all values' terms in the non-indicator dimensions search chain."""

import re
from types import SimpleNamespace
from unittest.mock import Mock

from sdmx.model import common as sdmx_common

from statgpt.app.chains.data_query.query_builder.dimensions.non_indicators import (
    NonIndicatorsSearchChainFactory,
)
from statgpt.app.schemas.query_builder import LLMSelectionDimensionCandidate
from statgpt.app.schemas.selection_candidates import SelectedCandidates
from statgpt.app.services.chat_facade import (
    ChannelServiceFacade,
    ScoredDimensionCandidate,
    VersionedDataSet,
)
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import VirtualDimensionValue
from statgpt.common.data.sdmx.common import DimensionCodeCategory
from statgpt.common.data.sdmx.v21.dataset import Sdmx21DataSet
from statgpt.common.schemas.enums import TimePeriodStrategy


def _make_dataset(
    entity_id: str, country_dim_id: str, dimensions: dict[str, tuple[str, str, str]]
) -> VersionedDataSet:
    """`dimensions` maps dimension id to its alias, 'all values' term id and label."""
    data = Mock(spec=Sdmx21DataSet)
    data.entity_id = entity_id
    data.source_id = entity_id
    data.non_indicator_dimensions.return_value = [
        SimpleNamespace(entity_id=dim_id, name=alias, alias=alias)
        for dim_id, (alias, _, _) in dimensions.items()
    ]
    data.country_dimension.return_value = SimpleNamespace(entity_id=country_dim_id)
    data.config = SimpleNamespace(
        country_dimension=country_dim_id,
        dimension_all_values={
            dim_id: VirtualDimensionValue(id=code_id, name=label, description=None)
            for dim_id, (_, code_id, label) in dimensions.items()
        },
    )
    return VersionedDataSet(version=Mock(), data=data)


# the country dimensions have different aliases and 'all values' labels
_COUNTRY_DIMENSIONS = {'ds_a': 'REF_AREA', 'ds_b': 'REF_AREA', 'ds_c': 'REP_COUNTRY'}


def _make_inputs() -> dict:
    datasets = [
        _make_dataset(
            'ds_a',
            country_dim_id='REF_AREA',
            dimensions={
                'REF_AREA': ('Country/Reference area', 'ALL_COUNTRIES', 'All countries'),
                'COUNTERPART_AREA': ('Counterpart area', 'ALL_COUNTRIES', 'All countries'),
            },
        ),
        _make_dataset(
            'ds_b',
            country_dim_id='REF_AREA',
            dimensions={
                'REF_AREA': (
                    'Country/Reference Area',
                    'ALL_COUNTRIES',
                    'All countries individually (wildcard)',
                ),
            },
        ),
        _make_dataset(
            'ds_c',
            country_dim_id='REP_COUNTRY',
            dimensions={'REP_COUNTRY': ('Reporting country', '_ALL', 'All countries')},
        ),
    ]
    return {
        'auth_context': Mock(spec=AuthContext),
        'data_service': Mock(spec=ChannelServiceFacade),
        'datasets_dict': {ds.data.entity_id: ds for ds in datasets},
    }


def _code_candidate(
    dataset_id: str, dimension_id: str, dimension_alias: str, code_id: str
) -> ScoredDimensionCandidate:
    return ScoredDimensionCandidate(
        score=0.9,
        dataset_id=dataset_id,
        dimension_category=DimensionCodeCategory(
            code=sdmx_common.Code(id=code_id, name=code_id),
            locale='en',
            dimension_id=dimension_id,
            dimension_name=dimension_alias,
            dimension_alias=dimension_alias,
        ),
    )


def _make_factory() -> NonIndicatorsSearchChainFactory:
    factory = NonIndicatorsSearchChainFactory.__new__(NonIndicatorsSearchChainFactory)
    factory._config = SimpleNamespace(  # type: ignore[assignment]
        time_period_strategy=TimePeriodStrategy.AFTER,
        filter_by_country_entities=True,
    )
    return factory


def _visible_country_terms_ids(candidates: list[LLMSelectionDimensionCandidate]) -> list[str]:
    """Ids of the country dimensions 'all values' terms the LLM sees in the prompt."""
    text = LLMSelectionDimensionCandidate.candidates_to_llm_string(candidates)
    visible_ids = set(re.findall(r'\(id: (\d+),', text))
    country_ids = {
        c._id for c in candidates if (c.dataset_id, c.dimension_id) in _COUNTRY_DIMENSIONS.items()
    }
    return sorted(visible_ids & country_ids, key=int)


def test_all_countries_terms_are_shown_to_llm_once():
    candidates = _make_factory()._add_all_values_to_nonindicator_candidates(_make_inputs())

    assert all(c.is_all_values for c in candidates)
    assert {(c.dataset_id, c.dimension_id) for c in candidates if c.is_all_countries} == set(
        _COUNTRY_DIMENSIONS.items()
    )

    text = LLMSelectionDimensionCandidate.candidates_to_llm_string(candidates)
    # one term for the country dimensions, one for the counterpart area dimension
    assert len(re.findall(r'\(id: (\d+),', text)) == 2
    assert len(_visible_country_terms_ids(candidates)) == 1
    assert '## dimension: "Counterpart area"' in text


def test_country_dimension_terms_are_marked():
    inputs = _make_inputs()
    inputs['dimension_candidates'] = [
        _code_candidate('ds_a', 'REF_AREA', 'Country/Reference area', 'FRA'),
        _code_candidate('ds_a', 'COUNTERPART_AREA', 'Counterpart area', 'FRA'),
        _code_candidate('ds_c', 'REP_COUNTRY', 'Reporting country', 'FR'),
    ]

    candidates = _make_factory()._prepare_dimension_candidates_for_llm(inputs)

    assert [c.is_country_dimension for c in candidates] == [True, False, True]
    assert not any(c.is_all_countries for c in candidates)


def test_all_countries_term_is_shown_next_to_most_country_terms():
    factory = _make_factory()
    inputs = _make_inputs()
    inputs['dimension_candidates'] = [
        _code_candidate('ds_a', 'REF_AREA', 'Country/Reference area', 'FRA'),
        *(
            _code_candidate('ds_c', 'REP_COUNTRY', 'Reporting country', code_id)
            for code_id in ('FR', 'DE', 'IT')
        ),
    ]
    inputs['dimension_candidates_for_llm_selection'] = (
        factory._prepare_dimension_candidates_for_llm(inputs)
    )

    candidates = factory._add_all_values_to_nonindicator_candidates(inputs)

    text = LLMSelectionDimensionCandidate.candidates_to_llm_string(candidates)
    # 'ds_a' term is the first one, but 'ds_c' dimension has the most country terms
    reporting_country_terms = text.split('## dimension: "Reporting country"')[1].split('##')[0]
    assert LLMSelectionDimensionCandidate.all_countries_label in reporting_country_terms


def test_all_countries_selection_keeps_every_dataset_with_all_countries_query():
    factory = _make_factory()
    inputs = _make_inputs()

    candidates = factory._add_all_values_to_nonindicator_candidates(inputs)
    inputs['dimension_candidates_for_llm_selection'] = candidates
    # the LLM selects only one of the country terms it sees (see issue #705)
    selected_ids = _visible_country_terms_ids(candidates)[:1]
    inputs['dimension_values_llm_selection_output'] = SelectedCandidates(ids=selected_ids)

    inputs['dimension_candidates'] = factory._filter_dimension_candidates_by_llm_response(inputs)
    inputs['strong_queries'] = factory._candidates_to_queries(inputs)
    queries = factory._filter_strong_queries_by_countries(inputs)

    assert set(queries) == set(_COUNTRY_DIMENSIONS)
    for ds_id, country_dim_id in _COUNTRY_DIMENSIONS.items():
        dim_queries = queries[ds_id].dimensions_queries_dict
        assert dim_queries[country_dim_id].is_all_selected
    # 'All countries' of the counterpart area dimension is a different term
    assert 'COUNTERPART_AREA' not in queries['ds_a'].dimensions_queries_dict
