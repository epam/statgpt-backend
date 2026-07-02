"""Unit tests for the Stage-1 (search preparation) chain composition."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from langchain_core.runnables import Runnable, RunnableLambda

from statgpt.app.chains.data_query.query_builder.misc.search_preparation import (
    SearchPreparationChainFactory,
)
from statgpt.app.schemas.query_builder import (
    DataSetsSelectionChainResponse,
    DateTimeQueryResponse,
    NamedEntitiesResponse,
    NamedEntity,
)
from statgpt.app.services.chat_facade import ChannelServiceFacade, VersionedDataSet
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import DataSet
from statgpt.common.schemas.data_query_tool import DataQueryStageNames
from statgpt.common.schemas.tool_details import StagesConfig

NORMALIZED_QUERY = 'normalized query'
REWRITTEN_QUERY = 'rewritten query'


class FakeChoice:
    """Minimal ChoiceI implementation for ChainState validation."""

    def create_stage(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('no stage expected in this test')

    def append_content(self, content: str) -> Any:
        pass

    def add_attachment(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def set_state(self, state: dict) -> Any:
        pass


class StubChain:
    """LLM sub-chain stub: records the inputs it was created with, returns a fixed output."""

    def __init__(self, output: Any):
        self._output = output
        self.captured_inputs: dict | None = None

    def create_chain(self, inputs: dict) -> Runnable:
        self.captured_inputs = dict(inputs)
        return RunnableLambda(lambda _: self._output)


def _make_versioned_dataset(entity_id: str) -> VersionedDataSet:
    data = Mock(spec=DataSet)
    data.entity_id = entity_id
    return VersionedDataSet(version=Mock(), data=data)


def _make_factory() -> tuple[SearchPreparationChainFactory, dict[str, StubChain]]:
    factory = SearchPreparationChainFactory.__new__(SearchPreparationChainFactory)
    factory._config = SimpleNamespace(  # type: ignore[assignment]
        pipeline_stage_names=DataQueryStageNames(),
        stages_config=StagesConfig(),
    )

    stubs = {
        'normalization': StubChain(NORMALIZED_QUERY),
        'datetime': StubChain(
            DateTimeQueryResponse(start='2020-01-01', end='2020-12-31', time_period_specified=True)
        ),
        'named_entities': StubChain(
            NamedEntitiesResponse(entities=[NamedEntity(entity='France', entity_type='country')])
        ),
        'datasets_selection': StubChain(
            DataSetsSelectionChainResponse(dataset_ids=['ds2'], rewritten_query=REWRITTEN_QUERY)
        ),
    }
    factory._normalization_chain = stubs['normalization']  # type: ignore[assignment]
    factory._datetime_chain = stubs['datetime']  # type: ignore[assignment]
    factory._named_entities_chain = stubs['named_entities']  # type: ignore[assignment]
    factory._datasets_selection_chain = stubs['datasets_selection']  # type: ignore[assignment]
    return factory, stubs


def _make_inputs() -> dict:
    data_service = Mock(spec=ChannelServiceFacade)
    data_service.list_available_datasets.return_value = [
        _make_versioned_dataset('ds1'),
        _make_versioned_dataset('ds2'),
    ]
    data_service.get_country_named_entity_type.return_value = 'Country'

    return {
        'auth_context': Mock(spec=AuthContext),
        'choice': FakeChoice(),
        'target': None,
        'state': {},
        'data_service': data_service,
        'query': 'GDP of France in 2020',
    }


async def test_search_preparation_chain_produces_all_stage1_keys():
    factory, stubs = _make_factory()
    inputs = _make_inputs()

    result = await factory.create().ainvoke(inputs)

    expected_keys = {
        'versioned_datasets_dict',
        'normalized_query',
        'normalized_query_raw',
        'date_time_query_response',
        'datasets_selection_response',
        'datasets_dict',
        'named_entities_response',
        'country_named_entities',
    }
    assert expected_keys <= result.keys()

    assert set(result['versioned_datasets_dict']) == {'ds1', 'ds2'}
    # datasets filter from the selection response is applied
    assert set(result['datasets_dict']) == {'ds2'}
    assert result['date_time_query_response'] == DateTimeQueryResponse(
        start='2020-01-01', end='2020-12-31', time_period_specified=True
    )
    assert result['country_named_entities'] == [NamedEntity(entity='France', entity_type='country')]


async def test_normalized_query_after_selection_equals_rewritten_query():
    factory, stubs = _make_factory()
    inputs = _make_inputs()

    result = await factory.create().ainvoke(inputs)

    assert result['normalized_query'] == REWRITTEN_QUERY
    # raw normalization output is preserved before the rewrite
    assert result['normalized_query_raw'] == NORMALIZED_QUERY


async def test_stage1_step_inputs_preserve_data_dependencies():
    factory, stubs = _make_factory()
    inputs = _make_inputs()

    await factory.create().ainvoke(inputs)

    # the merged head runs on the raw tool inputs: neither normalization nor datetime
    # may observe outputs of other head branches
    for stub_name in ('normalization', 'datetime'):
        captured = stubs[stub_name].captured_inputs
        assert captured is not None
        assert 'normalized_query' not in captured
        assert 'versioned_datasets_dict' not in captured
        assert 'date_time_query_response' not in captured

    # dataset selection consumes the pre-rewrite normalized query
    selection_inputs = stubs['datasets_selection'].captured_inputs
    assert selection_inputs is not None
    assert selection_inputs['normalized_query'] == NORMALIZED_QUERY

    # NER consumes the post-rewrite normalized query
    ner_inputs = stubs['named_entities'].captured_inputs
    assert ner_inputs is not None
    assert ner_inputs['normalized_query'] == REWRITTEN_QUERY
