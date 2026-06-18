from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from statgpt.app.chains.data_query.data_query_tool import DataQueryTool
from statgpt.app.chains.datasets_meta.metadata_tool import DatasetsMetadataTool
from statgpt.app.chains.file_rags.file_rag_tool import FileRagTool
from statgpt.app.chains.out_of_scope_checker import OutOfScopeChecker, OutOfScopeCheckerResponse
from statgpt.app.chains.tools import StatGptTool
from statgpt.app.chains.web_search.web_agent import WebSearchAgentTool
from statgpt.app.chains.web_search.web_rag import WebSearchTool


def _channel_config():
    return SimpleNamespace(
        out_of_scope=SimpleNamespace(
            domain="official statistics",
            use_general_topics_blacklist=False,
            custom_blacklist=None,
            llm_model_config=SimpleNamespace(),
        ),
        supreme_agent=SimpleNamespace(language_instructions=["Answer in English"]),
        tools=[SimpleNamespace(name="data_query", out_of_scope_description="Query data")],
    )


@pytest.mark.parametrize("out_of_scope", [True, False])
async def test_classify_returns_model_decision(monkeypatch, out_of_scope):
    decision = OutOfScopeCheckerResponse(reasoning="because reasons", out_of_scope=out_of_scope)

    class _FakeModel:
        def with_structured_output(self, schema, method):
            return RunnableLambda(lambda _: decision)

    monkeypatch.setattr(
        "statgpt.app.chains.out_of_scope_checker.get_chat_model",
        lambda api_key, model_config: _FakeModel(),
    )

    checker = OutOfScopeChecker(_channel_config())  # type: ignore[arg-type]
    result = await checker.classify([HumanMessage(content="GDP of France")], "key")

    assert isinstance(result, OutOfScopeCheckerResponse)
    assert result.out_of_scope is out_of_scope


@pytest.mark.parametrize(
    "tool_cls",
    [DataQueryTool, DatasetsMetadataTool, FileRagTool, WebSearchTool, WebSearchAgentTool],
)
def test_free_text_tools_expose_query_as_guardrail_input(tool_cls):
    # get_guardrail_input ignores instance state, so it can be called without
    # constructing the (config-heavy) tool instance.
    assert tool_cls.get_guardrail_input(object(), {"query": "GDP"}) == "GDP"
    assert tool_cls.get_guardrail_input(object(), {}) is None


def test_base_tool_has_no_guardrail_input():
    assert StatGptTool.get_guardrail_input(object(), {"query": "GDP"}) is None
