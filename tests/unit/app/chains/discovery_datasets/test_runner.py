"""Tests for the chat-time discovery datasets lookup.

The runner decorates someone else's answer, so most of what matters here is what it does when
something goes wrong: it must never raise, and it must never put a dataset in front of a user
that the search did not return.
"""

from typing import Any

import pytest

from statgpt.app.chains.discovery_datasets import DiscoveryDatasetsRunner
from statgpt.app.chains.discovery_datasets import runner as runner_module
from statgpt.app.schemas.discovery_datasets import (
    DiscoveryRelevanceItem,
    DiscoveryRelevanceResponse,
)
from statgpt.common.schemas import (
    ChannelConfig,
    GenericRagDocument,
    SupremeAgentConfig,
    SystemUserPrompt,
)
from statgpt.common.schemas.discovery_datasets_tool import (
    DiscoveryDatasetsDetails,
    DiscoveryDatasetsPrompts,
    DiscoveryDatasetsTemplates,
)
from statgpt.common.services.generic_rag import GenericRagChannelError

_APPLICATION = "generic-rag-app"

_SUPREME_AGENT = SupremeAgentConfig(
    name="T", domain="D", terminology_domain="T", language_instructions=["i"]
)


def _channel_config(**overrides: Any) -> ChannelConfig:
    return ChannelConfig.model_validate({"supremeAgent": _SUPREME_AGENT, **overrides})


def _details(**overrides: Any) -> DiscoveryDatasetsDetails:
    fields: dict[str, Any] = {
        "application_id": _APPLICATION,
        "top_n": 5,
        "templates": DiscoveryDatasetsTemplates(wrapper="### Datasets\n\n{items}", item="- {name}"),
    }
    fields.update(overrides)
    return DiscoveryDatasetsDetails(**fields)


def _document(document_id: int, name: str, agency: str = "IMF") -> GenericRagDocument:
    return GenericRagDocument(
        id=document_id,
        url=f"files/doc{document_id}.txt",
        display_name=f"doc{document_id}.txt",
        mime_type="text/plain",
        size=10,
        metadata={
            "grade": "C",
            "statgpt_channel": "statgpt-gtdc",
            "agency": agency,
            "name": name,
        },
        status="ready",
    )


class _FakeClient:
    """Stands in for `GenericRagSearchClient`, recording what was asked of it."""

    def __init__(
        self,
        documents: list[GenericRagDocument] | None = None,
        *,
        search_error: Exception | None = None,
        download_error: Exception | None = None,
    ) -> None:
        self._documents = documents or []
        self._search_error = search_error
        self._download_error = download_error
        self.search_calls: list[tuple[str, int, list[str] | None]] = []
        self.downloaded: list[int] = []

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def search_documents(
        self, query: str, limit: int, indexes: list[str] | None = None
    ) -> list[GenericRagDocument]:
        self.search_calls.append((query, limit, indexes))
        if self._search_error is not None:
            raise self._search_error
        return self._documents

    async def download_document(self, document_id: int) -> str:
        self.downloaded.append(document_id)
        if self._download_error is not None:
            raise self._download_error
        return f"Description of {document_id}."


class _Judge:
    """Stands in for the relevance chain, recording whether it was reached at all."""

    def __init__(
        self, response: DiscoveryRelevanceResponse | None = None, error: Exception | None = None
    ) -> None:
        self._response = response or DiscoveryRelevanceResponse()
        self._error = error
        self.calls: list[tuple[str, list[int]]] = []

    async def __call__(self, query: str, candidates: list[Any], auth_context: Any) -> Any:
        self.calls.append((query, [candidate.document_id for candidate in candidates]))
        if self._error is not None:
            raise self._error
        return self._response


@pytest.fixture
def inputs() -> dict[str, Any]:
    """The chain inputs the runner reads: an auth context, no choice, no debug stages."""
    return {"auth_context": _AuthContext(), "state": {}, "choice": None}


class _AuthContext:
    api_key = "user-key"


def _install(
    monkeypatch: pytest.MonkeyPatch, client: _FakeClient, judge: _Judge | None = None
) -> _Judge:
    judge = judge or _Judge()
    monkeypatch.setattr(
        runner_module.GenericRagSearchClient,
        "for_application",
        classmethod(lambda cls, application_id, auth_context: client),
    )
    monkeypatch.setattr(DiscoveryDatasetsRunner, "_judge", judge)
    return judge


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ availability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_no_runner_without_the_config_block() -> None:
    config = _channel_config()

    assert DiscoveryDatasetsRunner.from_channel_config(config) is None
    assert config.discovery_application_id is None


def test_no_runner_when_the_block_is_disabled() -> None:
    config = _channel_config(
        discoveryDatasets={
            "type": "DISCOVERY_DATASETS",
            "name": "discovery_datasets",
            "description": "d",
            "enabled": False,
            "details": {
                "applicationId": _APPLICATION,
                "templates": {"wrapper": "{items}", "item": "- {name}"},
            },
        }
    )

    assert DiscoveryDatasetsRunner.from_channel_config(config) is None
    # Indexing is unaffected by `enabled`: the publish target is still readable.
    assert config.discovery_application_id == _APPLICATION


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the happy path ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_relevant_datasets_are_rendered_in_rank_order(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    """The user reads the search's own ordering, not the order the model answered in."""
    client = _FakeClient([_document(1, "Alpha"), _document(2, "Beta"), _document(3, "Gamma")])
    _install(
        monkeypatch,
        client,
        _Judge(
            DiscoveryRelevanceResponse(
                items=[
                    DiscoveryRelevanceItem(document_id=3, relevant=True, reason="third"),
                    DiscoveryRelevanceItem(document_id=2, relevant=False, reason="no"),
                    DiscoveryRelevanceItem(document_id=1, relevant=True, reason="first"),
                ]
            )
        ),
    )

    outcome = await DiscoveryDatasetsRunner(_details()).run("gdp", inputs)

    assert outcome.rendered == "### Datasets\n\n- Alpha\n- Gamma"
    assert outcome.eval_attachment.selected_document_ids == [1, 3]
    assert client.search_calls == [("gdp", 5, None)]


async def test_the_eval_attachment_carries_the_whole_run(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    client = _FakeClient([_document(1, "Alpha")])
    _install(
        monkeypatch,
        client,
        _Judge(
            DiscoveryRelevanceResponse(
                items=[DiscoveryRelevanceItem(document_id=1, relevant=True, reason="covers it")]
            )
        ),
    )

    attachment = (await DiscoveryDatasetsRunner(_details()).run("gdp", inputs)).eval_attachment

    assert attachment.query == "gdp"
    assert [candidate.rank for candidate in attachment.candidates] == [1]
    assert attachment.candidates[0].description == "Description of 1."
    assert attachment.llm_response is not None
    assert attachment.rendered == "### Datasets\n\n- Alpha"
    assert attachment.error is None


async def test_configured_top_n_and_indexes_reach_the_search(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    client = _FakeClient([])
    _install(monkeypatch, client)

    await DiscoveryDatasetsRunner(_details(top_n=12, indexes=["semantic"])).run("q", inputs)

    assert client.search_calls == [("q", 12, ["semantic"])]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ nothing to show ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_no_search_results_skips_the_judge_entirely(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    judge = _install(monkeypatch, _FakeClient([]))

    outcome = await DiscoveryDatasetsRunner(_details()).run("q", inputs)

    assert outcome.rendered is None
    assert judge.calls == []
    assert outcome.eval_attachment.llm_response is None
    assert outcome.eval_attachment.error is None


async def test_nothing_judged_relevant_renders_nothing(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    _install(
        monkeypatch,
        _FakeClient([_document(1, "Alpha")]),
        _Judge(
            DiscoveryRelevanceResponse(
                items=[DiscoveryRelevanceItem(document_id=1, relevant=False, reason="off topic")]
            )
        ),
    )

    outcome = await DiscoveryDatasetsRunner(_details()).run("q", inputs)

    assert outcome.rendered is None
    assert outcome.eval_attachment.selected_document_ids == []


async def test_a_hit_whose_metadata_is_not_a_discovery_record_is_skipped(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    """One RAG channel can hold documents this channel never published."""
    foreign = GenericRagDocument(id=2, display_name="other.pdf", metadata={"unrelated": "yes"})
    _install(
        monkeypatch,
        _FakeClient([_document(1, "Alpha"), foreign]),
        _Judge(
            DiscoveryRelevanceResponse(
                items=[DiscoveryRelevanceItem(document_id=1, relevant=True, reason="ok")]
            )
        ),
    )

    outcome = await DiscoveryDatasetsRunner(_details()).run("q", inputs)

    assert outcome.rendered == "### Datasets\n\n- Alpha"
    assert [candidate.document_id for candidate in outcome.eval_attachment.candidates] == [1]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hallucinations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_verdict_on_an_id_that_was_never_offered_is_dropped(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    """An invented id must not be able to put a dataset in front of a user."""
    _install(
        monkeypatch,
        _FakeClient([_document(1, "Alpha")]),
        _Judge(
            DiscoveryRelevanceResponse(
                items=[
                    DiscoveryRelevanceItem(document_id=999, relevant=True, reason="invented"),
                    DiscoveryRelevanceItem(document_id=1, relevant=True, reason="real"),
                ]
            )
        ),
    )

    outcome = await DiscoveryDatasetsRunner(_details()).run("q", inputs)

    assert outcome.rendered == "### Datasets\n\n- Alpha"
    assert outcome.eval_attachment.selected_document_ids == [1]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ failures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_failed_search_yields_no_block_and_records_the_error(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    """A user without access to the RAG deployment must still get their data."""
    _install(
        monkeypatch,
        _FakeClient(search_error=GenericRagChannelError("document search", "forbidden", 403)),
    )

    outcome = await DiscoveryDatasetsRunner(_details()).run("q", inputs)

    assert outcome.rendered is None
    assert outcome.eval_attachment.error is not None
    assert "403" in outcome.eval_attachment.error


async def test_a_failed_download_still_judges_the_candidate(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    """Most of a discovery record is metadata, so one unreadable body is not fatal."""
    _install(
        monkeypatch,
        _FakeClient([_document(1, "Alpha")], download_error=RuntimeError("gone")),
        _Judge(
            DiscoveryRelevanceResponse(
                items=[DiscoveryRelevanceItem(document_id=1, relevant=True, reason="ok")]
            )
        ),
    )

    outcome = await DiscoveryDatasetsRunner(_details()).run("q", inputs)

    assert outcome.rendered == "### Datasets\n\n- Alpha"
    assert outcome.eval_attachment.candidates[0].description == ""
    assert outcome.eval_attachment.error is None


async def test_a_failed_judge_yields_no_block_and_records_the_error(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    _install(
        monkeypatch,
        _FakeClient([_document(1, "Alpha")]),
        _Judge(error=RuntimeError("llm exploded")),
    )

    outcome = await DiscoveryDatasetsRunner(_details()).run("q", inputs)

    assert outcome.rendered is None
    assert outcome.eval_attachment.error == "RuntimeError: llm exploded"
    assert outcome.eval_attachment.candidates != []


async def test_a_missing_auth_context_is_reported_rather_than_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _FakeClient([]))

    outcome = await DiscoveryDatasetsRunner(_details()).run("q", {"state": {}})

    assert outcome.rendered is None
    assert outcome.eval_attachment.error is not None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the relevance prompt ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class _FakeLlm:
    """Stands in for the chat model, capturing the messages the prompt template produced."""

    def __init__(self, sink: list[Any]) -> None:
        self._sink = sink

    def with_structured_output(self, schema: Any, method: str) -> Any:
        from langchain_core.runnables import RunnableLambda

        assert schema is DiscoveryRelevanceResponse
        assert method == "json_schema"

        def respond(value: Any) -> DiscoveryRelevanceResponse:
            self._sink.append(value)
            return DiscoveryRelevanceResponse(
                items=[DiscoveryRelevanceItem(document_id=1, relevant=True, reason="ok")]
            )

        return RunnableLambda(respond)


async def test_the_default_relevance_prompt_renders_the_query_and_candidates(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    """Exercises the real prompt template: a stray brace in it would only fail here."""
    captured: list[Any] = []
    monkeypatch.setattr(
        runner_module.GenericRagSearchClient,
        "for_application",
        classmethod(lambda cls, application_id, auth_context: _FakeClient([_document(1, "Alpha")])),
    )
    monkeypatch.setattr(runner_module, "get_chat_model", lambda **_: _FakeLlm(captured))

    outcome = await DiscoveryDatasetsRunner(_details()).run("gdp of france", inputs)

    assert outcome.rendered == "### Datasets\n\n- Alpha"
    (prompt_value,) = captured
    system, user = prompt_value.to_messages()
    assert "official statistics" in system.content
    assert "gdp of france" in user.content
    # The candidates reach the model as YAML, description included.
    assert "document_id: 1" in user.content
    assert "Alpha" in user.content
    assert "Description of 1." in user.content


async def test_a_configured_relevance_prompt_overrides_the_default(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    captured: list[Any] = []
    monkeypatch.setattr(
        runner_module.GenericRagSearchClient,
        "for_application",
        classmethod(lambda cls, application_id, auth_context: _FakeClient([_document(1, "Alpha")])),
    )
    monkeypatch.setattr(runner_module, "get_chat_model", lambda **_: _FakeLlm(captured))
    details = _details(
        prompts=DiscoveryDatasetsPrompts(
            relevance_prompt=SystemUserPrompt(
                system_message="Custom judge.", user_message="Q: {query}\nC: {candidates}"
            )
        )
    )

    await DiscoveryDatasetsRunner(details).run("inflation", inputs)

    (prompt_value,) = captured
    system, user = prompt_value.to_messages()
    assert system.content == "Custom judge."
    assert user.content.startswith("Q: inflation")
