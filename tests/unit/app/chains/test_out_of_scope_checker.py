from types import SimpleNamespace
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import Field, create_model

from statgpt.app.chains.data_query.data_query_tool import DataQueryArgs
from statgpt.app.chains.datasets_meta.metadata_tool import DatasetsMetadataArgs
from statgpt.app.chains.file_rags.file_rag_tool import FileRagArgs
from statgpt.app.chains.out_of_scope_checker import OutOfScopeChecker, OutOfScopeCheckerResponse
from statgpt.app.chains.tools import GuardrailInput, ToolArgs
from statgpt.app.chains.web_search.web_agent import WebSearchArgs
from statgpt.app.chains.web_search.web_rag import BaseWebSearchArgs


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
    result = await checker.classify(
        [HumanMessage(content="GDP of France")],
        SimpleNamespace(api_key="key"),  # type: ignore[arg-type]
    )

    assert isinstance(result, OutOfScopeCheckerResponse)
    assert result.out_of_scope is out_of_scope


async def test_generate_response_returns_model_message(monkeypatch):
    monkeypatch.setattr(
        "statgpt.app.chains.out_of_scope_checker.get_chat_model",
        lambda api_key, model_config: RunnableLambda(
            lambda _: AIMessage(content="I can only help with official statistics.")
        ),
    )

    checker = OutOfScopeChecker(_channel_config())  # type: ignore[arg-type]
    result = await checker.generate_response(
        [HumanMessage(content="weather in London")],
        "off-domain weather request",
        SimpleNamespace(api_key="key"),  # type: ignore[arg-type]
    )

    assert result == "I can only help with official statistics."


@pytest.mark.parametrize(
    "args_cls",
    [DataQueryArgs, DatasetsMetadataArgs, FileRagArgs, BaseWebSearchArgs, WebSearchArgs],
)
def test_free_text_args_expose_query_as_guardrail_input(args_cls):
    assert args_cls.get_guardrail_input({"query": "GDP"}) == "GDP"
    assert args_cls.get_guardrail_input({}) is None


def test_base_args_have_no_guardrail_input():
    assert ToolArgs.get_guardrail_input({"query": "GDP"}) is None


def test_guardrail_input_follows_renamed_field():
    # The marker travels with the field declaration: renaming the field keeps the
    # guardrail wired up without touching any extraction logic.
    class RenamedArgs(ToolArgs):
        question: Annotated[str, GuardrailInput] = Field()

    assert RenamedArgs.get_guardrail_input({"question": "GDP"}) == "GDP"
    assert RenamedArgs.get_guardrail_input({"query": "GDP"}) is None


def test_guardrail_input_rejects_multiple_marked_fields():
    class AmbiguousArgs(ToolArgs):
        first: Annotated[str, GuardrailInput] = Field()
        second: Annotated[str, GuardrailInput] = Field()

    with pytest.raises(ValueError, match="multiple guardrail fields"):
        AmbiguousArgs.get_guardrail_input({"first": "a", "second": "b"})


def test_guardrail_marker_survives_create_model_inheritance():
    # WebSearchTool builds its schema dynamically via create_model(__base__=...);
    # the inherited marked field must still be screened.
    dynamic = create_model("DynamicWebSearchArgs", __base__=BaseWebSearchArgs)
    assert dynamic.get_guardrail_input({"query": "GDP"}) == "GDP"
