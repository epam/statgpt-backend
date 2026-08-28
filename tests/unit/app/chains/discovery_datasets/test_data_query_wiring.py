"""Tests for how the discovery datasets lookup is wired into the data query tool.

The lookup rides along with a data query, so the wiring has two obligations: run it beside the
pipeline rather than after it, and never let it change what the user would otherwise have got.
"""

import asyncio
from typing import Any

import pytest

from statgpt.app.chains.data_query import data_query_tool as tool_module
from statgpt.app.chains.data_query.data_query_tool import DataQueryTool
from statgpt.app.chains.data_query.parameters import DataQueryParameters
from statgpt.app.config import ChainParametersConfig
from statgpt.app.schemas.discovery_datasets import (
    DiscoveryDatasetsEvalAttachment,
    DiscoveryDatasetsOutcome,
)
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas import DataQueryTool as DataQueryToolConfig
from statgpt.common.schemas import SupremeAgentConfig
from statgpt.common.schemas.enums import InvocationSource

_SUPREME_AGENT = SupremeAgentConfig(
    name="T", domain="D", terminology_domain="T", language_instructions=["i"]
)

_DISCOVERY_BLOCK: dict[str, Any] = {
    "type": "DISCOVERY_DATASETS",
    "name": "discovery_datasets",
    "description": "d",
    "details": {
        "applicationId": "generic-rag-app",
        "templates": {"wrapper": "### Datasets\n\n{items}", "item": "- {name}"},
    },
}


def _tool(*, with_discovery: bool) -> DataQueryTool:
    details: dict[str, Any] = {"supremeAgent": _SUPREME_AGENT}
    if with_discovery:
        details["discoveryDatasets"] = _DISCOVERY_BLOCK
    channel_config = ChannelConfig.model_validate(details)
    tool_config = DataQueryToolConfig(name="data_query", description="d")
    return DataQueryTool.from_config(tool_config, channel_config)  # type: ignore[return-value]


class _Chain:
    """Stands in for the query builder chain, reporting when it ran."""

    def __init__(
        self,
        response: str = "Here is your data.",
        *,
        error: Exception | None = None,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self._started = started
        self._release = release

    async def ainvoke(self, inputs: dict) -> dict:
        if self._started is not None:
            self._started.set()
        if self._release is not None:
            await self._release.wait()
        if self._error is not None:
            raise self._error
        return {DataQueryParameters.RESPONSE_FIELD: self._response}


class _Runner:
    """Stands in for the lookup, reporting when it ran and whether it was cancelled."""

    def __init__(
        self,
        rendered: str | None = "### Datasets\n\n- Alpha",
        *,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self._rendered = rendered
        self._started = started
        self._release = release
        self.cancelled = False

    async def run(self, query: str, inputs: dict) -> DiscoveryDatasetsOutcome:
        if self._started is not None:
            self._started.set()
        try:
            if self._release is not None:
                await self._release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return DiscoveryDatasetsOutcome(
            rendered=self._rendered,
            eval_attachment=DiscoveryDatasetsEvalAttachment(query=query, rendered=self._rendered),
        )


def _install(monkeypatch: pytest.MonkeyPatch, chain: _Chain, runner: _Runner | None) -> None:
    monkeypatch.setattr(
        tool_module,
        "QueryBuilderFactory",
        lambda _config: _FactoryStub(chain),
    )
    monkeypatch.setattr(
        tool_module.DiscoveryDatasetsRunner,
        "from_channel_config",
        classmethod(lambda cls, channel_config: runner),
    )


class _FactoryStub:
    def __init__(self, chain: _Chain) -> None:
        self._chain = chain

    async def create_chain(self, inputs: dict) -> _Chain:
        return self._chain


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the response ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_the_rendered_block_is_appended_to_the_tool_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _Chain("Here is your data."), _Runner())

    response, artifact = await _tool(with_discovery=True)._arun({}, "gdp")

    assert response == "Here is your data.\n\n### Datasets\n\n- Alpha"
    assert artifact.discovery_datasets_block == "### Datasets\n\n- Alpha"
    assert artifact.discovery_datasets_eval_attachment is not None
    assert artifact.discovery_datasets_eval_attachment.query == "gdp"


async def test_an_mcp_call_carries_the_block_on_the_artifact_instead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An MCP client gets it as a content block of its own, so folding it in would duplicate it.

    It also keeps markdown out of the response the provider reports as `message`, which a
    client parses rather than reads.
    """
    _install(monkeypatch, _Chain("Here is your data."), _Runner())
    inputs = {ChainParametersConfig.INVOCATION_SOURCE: InvocationSource.MCP}

    response, artifact = await _tool(with_discovery=True)._arun(inputs, "gdp")

    assert response == "Here is your data."
    assert artifact.discovery_datasets_block == "### Datasets\n\n- Alpha"


async def test_a_lookup_that_found_nothing_leaves_the_response_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eval attachment is still carried: a run that found nothing is worth seeing."""
    _install(monkeypatch, _Chain("Here is your data."), _Runner(rendered=None))

    response, artifact = await _tool(with_discovery=True)._arun({}, "gdp")

    assert response == "Here is your data."
    assert artifact.discovery_datasets_block is None
    assert artifact.discovery_datasets_eval_attachment is not None


async def test_an_unconfigured_channel_runs_the_pipeline_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _Chain("Here is your data."), None)

    response, artifact = await _tool(with_discovery=False)._arun({}, "gdp")

    assert response == "Here is your data."
    assert artifact.discovery_datasets_eval_attachment is None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ concurrency ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_the_lookup_starts_before_the_pipeline_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise the feature costs the user its whole latency."""
    lookup_started = asyncio.Event()
    release_pipeline = asyncio.Event()
    chain = _Chain(started=asyncio.Event(), release=release_pipeline)
    runner = _Runner(started=lookup_started)
    _install(monkeypatch, chain, runner)

    task = asyncio.create_task(_tool(with_discovery=True)._arun({}, "gdp"))
    await asyncio.wait_for(lookup_started.wait(), timeout=1)
    release_pipeline.set()

    response, _ = await asyncio.wait_for(task, timeout=1)
    assert response.endswith("- Alpha")


async def test_a_failed_pipeline_raises_unwrapped_and_cancels_the_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tool's caller must see the pipeline's own error, not an ExceptionGroup.

    The pipeline fails only once the lookup is in flight, which is the case that matters: a
    lookup left running past the turn would write its debug stage into someone else's answer.
    """
    lookup_started = asyncio.Event()
    runner = _Runner(started=lookup_started, release=asyncio.Event())  # never released
    _install(
        monkeypatch,
        _Chain(error=RuntimeError("pipeline exploded"), release=lookup_started),
        runner,
    )

    with pytest.raises(RuntimeError, match="pipeline exploded"):
        await _tool(with_discovery=True)._arun({}, "gdp")

    # No yielding to the loop first: the tool awaits the cancellation before it re-raises, so
    # the lookup is already finished by the time its caller sees the error.
    assert runner.cancelled is True
