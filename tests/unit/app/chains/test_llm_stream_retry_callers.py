"""The streaming LLM calls outside the Supreme Agent retry a stream that stalls before any content."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableGenerator

from statgpt.app.chains.data_query.parameters import DataQueryParameters
from statgpt.app.chains.data_query.query_builder.query import incomplete_queries
from statgpt.app.chains.data_query.query_builder.query.incomplete_queries import (
    IncompleteQueriesChain,
)
from statgpt.app.chains.datasets_meta import metadata_tool
from statgpt.app.chains.out_of_scope_checker import OutOfScopeChecker, OutOfScopeCheckerResponse
from statgpt.app.chains.tools import StatGptTool
from statgpt.app.config import ChainParametersConfig
from statgpt.common.schemas import LLMModelConfig
from statgpt.common.schemas.channel import ChannelConfig, OutOfScopeConfig, SupremeAgentConfig
from statgpt.common.schemas.tools import DatasetsMetadataTool as DatasetsMetadataToolConfig
from statgpt.common.utils import llm_stream_retry

_ANSWER = [AIMessageChunk(content=""), AIMessageChunk(content="Hello"), AIMessageChunk(content="!")]


class _StallingThenAnsweringModel:
    """A chat model whose first stream stalls before any chunk; the next one answers."""

    def __init__(self) -> None:
        self.calls = 0
        self.runnable = RunnableGenerator(self._astream)

    async def _astream(self, inputs: AsyncIterator[Any]) -> AsyncIterator[AIMessageChunk]:
        async for _ in inputs:
            pass
        self.calls += 1
        if self.calls == 1:
            raise httpx.ReadTimeout("")
        for chunk in _ANSWER:
            yield chunk


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(llm_stream_retry.asyncio, "sleep", no_sleep)


@pytest.fixture
def model() -> _StallingThenAnsweringModel:
    return _StallingThenAnsweringModel()


@pytest.fixture
def channel_config() -> ChannelConfig:
    return ChannelConfig(
        supreme_agent=SupremeAgentConfig(
            name="StatGPT",
            domain="official statistics",
            terminology_domain="official statistics",
        ),
        out_of_scope=OutOfScopeConfig(
            domain="official statistics", start_new_conversation_messages_threshold=-1
        ),
    )


def _appended(stage: MagicMock) -> str:
    return "".join(call.args[0] for call in stage.append_content.call_args_list)


async def test_out_of_scope_response_retries_a_stream_that_stalls_before_any_content(
    monkeypatch: pytest.MonkeyPatch,
    model: _StallingThenAnsweringModel,
    channel_config: ChannelConfig,
):
    checker = OutOfScopeChecker(channel_config)
    verdict = OutOfScopeCheckerResponse(reasoning="Not about statistics.", out_of_scope=True)
    checker_chain = SimpleNamespace(ainvoke=AsyncMock(return_value=verdict))
    monkeypatch.setattr(checker, "build_checker_chain", lambda *args: checker_chain)
    monkeypatch.setattr(checker, "build_response_chain", lambda *args: model.runnable)
    choice = MagicMock()
    inputs = {
        ChainParametersConfig.STATE: {},
        ChainParametersConfig.SKIP_OUT_OF_SCOPE_CHECK: False,
        ChainParametersConfig.AUTH_CONTEXT: MagicMock(),
        ChainParametersConfig.CHOICE: choice,
        ChainParametersConfig.HISTORY: MagicMock(),
    }

    await checker._stream_response(inputs)

    assert model.calls == 2
    assert _appended(choice) == "Hello!"


async def test_incomplete_queries_retries_a_stream_that_stalls_before_any_content(
    monkeypatch: pytest.MonkeyPatch, model: _StallingThenAnsweringModel
):
    monkeypatch.setattr(incomplete_queries, "get_chat_model", lambda **kwargs: model.runnable)
    target = MagicMock()
    inputs = {
        "query": "GDP of Ukraine",
        ChainParametersConfig.TARGET: target,
        ChainParametersConfig.DATASET_QUERIES: {},
    }
    chain = IncompleteQueriesChain(LLMModelConfig(), system_prompt="Explain what is missing.")

    result = await chain.create_chain(inputs, api_key="key")

    assert model.calls == 2
    assert _appended(target) == "Hello!"
    assert result.invoke({})[DataQueryParameters.RESPONSE_FIELD] == "Hello!"


async def test_datasets_metadata_retries_a_stream_that_stalls_before_any_content(
    monkeypatch: pytest.MonkeyPatch,
    model: _StallingThenAnsweringModel,
    channel_config: ChannelConfig,
):
    monkeypatch.setattr(metadata_tool, "get_chat_model", lambda **kwargs: model.runnable)
    formatter = SimpleNamespace(format=AsyncMock(return_value="No datasets."))
    monkeypatch.setattr(metadata_tool, "DatasetsListFormatter", lambda *args, **kwargs: formatter)
    tool_config = DatasetsMetadataToolConfig(name="datasets_metadata", description="Metadata.")
    tool = StatGptTool.from_config(tool_config, channel_config)
    target = MagicMock()
    data_service = SimpleNamespace(list_available_datasets=AsyncMock(return_value=[]))
    inputs = {
        ChainParametersConfig.DATA_SERVICE: data_service,
        ChainParametersConfig.AUTH_CONTEXT: MagicMock(),
        ChainParametersConfig.TARGET: target,
    }

    response, _ = await tool._arun(inputs, query="Which datasets cover GDP?")

    assert model.calls == 2
    assert response == "Hello!"
    assert _appended(target) == "Hello!"
