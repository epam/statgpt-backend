"""The stage and the tool response each carry the explorer link on their own terms."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pandas as pd

from statgpt.app.chains.data_query.parameters import DataQueryParameters
from statgpt.app.chains.data_query.query_builder.query.execute_query import ExecuteQueryChain
from statgpt.app.config import ChainParametersConfig
from statgpt.app.services.chat_facade import ChannelServiceFacade, VersionedDataSet
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import DataSet, DataSetQuery
from statgpt.common.data.base.dataset import DataResponseStatus
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas.data_query_tool import DataQueryExplorerLink, DataQueryMessages
from statgpt.common.schemas.enums import (
    DataParsingStatus,
    DataRequestStatus,
    ExplorerLinkPolicy,
    InvocationSource,
    LocaleEnum,
)

_URL = "https://explorer.example/view?q=1"
_DATASET_ID = "ds1"


class _FakeTarget:
    def __init__(self) -> None:
        self.content = ""

    def append_content(self, content: str) -> None:
        self.content += content


class _FakeDataResponse:
    """The slice of `DataResponse` the query formatter reads."""

    def __init__(self, has_data: bool) -> None:
        self.status = DataResponseStatus(
            request_status=DataRequestStatus.SUCCESS, parsing_status=DataParsingStatus.SUCCESS
        )
        self.visual_dataframe = pd.DataFrame({"value": [1]}) if has_data else pd.DataFrame()
        self.url_query = _URL
        self.time_period = None
        self.is_empty = not has_data

    def get_display_series_count(self) -> int:
        return 1


def _versioned_dataset() -> VersionedDataSet:
    data = Mock(spec=DataSet)
    data.entity_id = _DATASET_ID
    data.source_id = _DATASET_ID
    data.name = "Test dataset"
    data.config = SimpleNamespace(is_official=False, citation=None)
    data.dimensions.return_value = []
    data.indicator_dimensions.return_value = []
    return VersionedDataSet(version=Mock(), data=data)


def _chain(explorer_link: DataQueryExplorerLink) -> ExecuteQueryChain:
    chain = ExecuteQueryChain.__new__(ExecuteQueryChain)
    chain._messages = DataQueryMessages()  # type: ignore[attr-defined]
    chain._explorer_link = explorer_link  # type: ignore[attr-defined]
    return chain


def _inputs(source: InvocationSource, has_data: bool = True) -> dict:
    data_service = Mock(spec=ChannelServiceFacade)
    data_service.channel_config = Mock(spec=ChannelConfig)
    data_service.channel_config.locale = LocaleEnum.EN

    return {
        "auth_context": Mock(spec=AuthContext),
        "choice": Mock(),
        "target": _FakeTarget(),
        "state": {},
        "data_service": data_service,
        "datasets_dict": {_DATASET_ID: _versioned_dataset()},
        "dataset_queries": {_DATASET_ID: DataSetQuery(dimensions_queries=[])},
        ChainParametersConfig.DATA_RESPONSES: {_DATASET_ID: _FakeDataResponse(has_data)},
        ChainParametersConfig.CONFIGURATION: SimpleNamespace(
            get_current_timestamp=lambda: "2026-01-01T00:00:00+00:00"
        ),
        ChainParametersConfig.INVOCATION_SOURCE: source,
        DataQueryParameters.STATE: {},
    }


async def _run(explorer_link: DataQueryExplorerLink, source: InvocationSource, **kwargs: Any):
    inputs = _inputs(source, **kwargs)
    result = await _chain(explorer_link).summarize_dataset_queries(inputs)
    return inputs["target"].content, result[DataQueryParameters.RESPONSE_FIELD]


async def test_default_config_links_on_both_surfaces():
    """The out-of-the-box behavior must be unchanged."""
    stage, response = await _run(DataQueryExplorerLink(), InvocationSource.AGENT)
    assert _URL in stage
    assert _URL in response


async def test_stage_keeps_the_link_while_the_agent_response_drops_it():
    stage, response = await _run(
        DataQueryExplorerLink(
            stage=ExplorerLinkPolicy.always, agent=ExplorerLinkPolicy.only_when_no_data
        ),
        InvocationSource.AGENT,
    )
    assert _URL in stage
    assert _URL not in response


async def test_only_when_no_data_still_links_where_nothing_was_delivered():
    stage, response = await _run(
        DataQueryExplorerLink(
            stage=ExplorerLinkPolicy.always, agent=ExplorerLinkPolicy.only_when_no_data
        ),
        InvocationSource.AGENT,
        has_data=False,
    )
    assert _URL in stage
    assert _URL in response


async def test_mcp_policy_applies_only_on_the_mcp_path():
    config = DataQueryExplorerLink(agent=ExplorerLinkPolicy.always, mcp=ExplorerLinkPolicy.never)

    _, agent_response = await _run(config, InvocationSource.AGENT)
    _, mcp_response = await _run(config, InvocationSource.MCP)

    assert _URL in agent_response
    assert _URL not in mcp_response
