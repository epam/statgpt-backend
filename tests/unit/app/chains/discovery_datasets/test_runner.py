"""Tests for the chat-time discovery datasets lookup.

The runner decorates someone else's answer, so most of what matters here is what it does when
something goes wrong: it must never raise, and it must never put a dataset in front of a user
that the search did not return.
"""

from typing import Any, Self

import pytest
from langchain_core.runnables import RunnableLambda

from statgpt.app.chains.discovery_datasets import DiscoveryDatasetsRunner
from statgpt.app.chains.discovery_datasets import runner as runner_module
from statgpt.app.config import StateVarsConfig
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
_CHANNEL = "statgpt-gtdc"

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


def _document(
    document_id: int, name: str, agency: str = "IMF", channel: str = _CHANNEL
) -> GenericRagDocument:
    return GenericRagDocument(
        id=document_id,
        url=f"files/doc{document_id}.txt",
        display_name=f"doc{document_id}.txt",
        mime_type="text/plain",
        size=10,
        metadata={
            "grade": "C",
            "statgpt_channel": channel,
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

    async def __aenter__(self) -> Self:
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


class _AuthContext:
    api_key: str = "user-key"


class _DataService:
    """Stands in for `ChannelServiceFacade`, which the runner reads the channel id from."""

    def __init__(self, deployment_id: str = _CHANNEL) -> None:
        self.deployment_id = deployment_id


@pytest.fixture
def inputs() -> dict[str, Any]:
    """The chain inputs the runner reads: an auth context, a channel, no debug stages."""
    return {
        "auth_context": _AuthContext(),
        "data_service": _DataService(),
        "state": {},
        "choice": None,
    }


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
    """One RAG channel can hold documents that are not discovery records at all."""
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ channel scoping ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_another_channels_document_is_never_surfaced(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    """Several StatGPT channels can share one RAG channel; each sees only its own records."""
    client = _FakeClient(
        [_document(1, "Alpha"), _document(2, "Beta", channel="statgpt-other-tenant")]
    )
    judge = _install(
        monkeypatch,
        client,
        _Judge(
            DiscoveryRelevanceResponse(
                items=[
                    DiscoveryRelevanceItem(document_id=1, relevant=True, reason="ok"),
                    DiscoveryRelevanceItem(document_id=2, relevant=True, reason="ok"),
                ]
            )
        ),
    )

    outcome = await DiscoveryDatasetsRunner(_details()).run("q", inputs)

    assert outcome.rendered == "### Datasets\n\n- Alpha"
    assert outcome.eval_attachment.selected_document_ids == [1]
    # The judge is never even offered the foreign document.
    assert judge.calls == [("q", [1])]
    # Nor is its body downloaded: the filter reads the metadata the search already returned.
    assert client.downloaded == [1]


async def test_only_foreign_documents_skips_the_judge_entirely(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    judge = _install(
        monkeypatch, _FakeClient([_document(1, "Alpha", channel="statgpt-other-tenant")])
    )

    outcome = await DiscoveryDatasetsRunner(_details()).run("q", inputs)

    assert outcome.rendered is None
    assert judge.calls == []
    assert outcome.eval_attachment.candidates == []
    assert outcome.eval_attachment.error is None


async def test_ranks_have_no_gaps_after_filtering(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    """Rank is the position a user reads, so it counts the documents that survived."""
    _install(
        monkeypatch,
        _FakeClient(
            [
                _document(1, "Alpha"),
                _document(2, "Beta", channel="statgpt-other-tenant"),
                _document(3, "Gamma"),
            ]
        ),
    )

    outcome = await DiscoveryDatasetsRunner(_details()).run("q", inputs)

    candidates = outcome.eval_attachment.candidates
    assert [(item.document_id, item.rank) for item in candidates] == [(1, 1), (3, 2)]


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


@pytest.mark.parametrize("missing", ["auth_context", "data_service"])
async def test_a_missing_chain_input_is_reported_rather_than_raised(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any], missing: str
) -> None:
    _install(monkeypatch, _FakeClient([]))
    del inputs[missing]

    outcome = await DiscoveryDatasetsRunner(_details()).run("q", inputs)

    assert outcome.rendered is None
    assert outcome.eval_attachment.error is not None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the debug stage ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class _Stage:
    """Records what a stage was told, and when it was opened and closed."""

    def __init__(self) -> None:
        self.content: list[str] = []
        self.closed = False

    def append_content(self, content: str) -> None:
        self.content.append(content)

    def append_name(self, name: str) -> None:
        return None

    def close(self, *_: Any, **__: Any) -> None:
        self.closed = True

    def open(self) -> None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True


class _Choice:
    """A choice that hands out one recording stage, as `create_stage` is used here."""

    def __init__(self) -> None:
        self.stage = _Stage()
        self.stage_names: list[str] = []

    def create_stage(self, name: str = "", **_: Any) -> _Stage:
        self.stage_names.append(name)
        return self.stage


@pytest.fixture
def debug_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    inputs["state"] = {StateVarsConfig.SHOW_DEBUG_STAGES: True}
    inputs["choice"] = _Choice()
    return inputs


async def test_each_step_reports_to_the_debug_stage_as_it_finishes(
    monkeypatch: pytest.MonkeyPatch, debug_inputs: dict[str, Any]
) -> None:
    """The stage is the lookup's own, and it fills up as the run goes rather than at the end."""
    _install(
        monkeypatch,
        _FakeClient([_document(1, "Alpha")]),
        _Judge(
            DiscoveryRelevanceResponse(
                items=[DiscoveryRelevanceItem(document_id=1, relevant=True, reason="covers it")]
            )
        ),
    )

    await DiscoveryDatasetsRunner(_details()).run("gdp", debug_inputs)

    choice: _Choice = debug_inputs["choice"]
    assert choice.stage_names == ["[DEBUG] Discovery Datasets Lookup"]
    sections = "".join(choice.stage.content)
    assert "## Retrieved documents" in sections
    assert "## Relevance verdicts" in sections
    assert "## Rendered block" in sections
    assert "covers it" in sections
    assert choice.stage.closed is True


async def test_a_failure_is_reported_on_the_debug_stage_too(
    monkeypatch: pytest.MonkeyPatch, debug_inputs: dict[str, Any]
) -> None:
    _install(
        monkeypatch,
        _FakeClient(search_error=GenericRagChannelError("document search", "forbidden", 403)),
    )

    await DiscoveryDatasetsRunner(_details()).run("gdp", debug_inputs)

    sections = "".join(debug_inputs["choice"].stage.content)
    assert "## Error" in sections
    assert "403" in sections


async def test_no_stage_is_opened_without_debug_stages(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Any]
) -> None:
    """The lookup has no user-facing progress to report, so it is debug-only."""
    choice = _Choice()
    inputs["choice"] = choice
    _install(monkeypatch, _FakeClient([_document(1, "Alpha")]))

    await DiscoveryDatasetsRunner(_details()).run("gdp", inputs)

    assert choice.stage_names == []
    assert choice.stage.content == []


async def test_a_broken_stage_does_not_cost_the_lookup_its_block(
    monkeypatch: pytest.MonkeyPatch, debug_inputs: dict[str, Any]
) -> None:
    """Reporting is the least important thing the lookup does."""
    _install(
        monkeypatch,
        _FakeClient([_document(1, "Alpha")]),
        _Judge(
            DiscoveryRelevanceResponse(
                items=[DiscoveryRelevanceItem(document_id=1, relevant=True, reason="ok")]
            )
        ),
    )
    stage = debug_inputs["choice"].stage
    monkeypatch.setattr(
        stage, "append_content", lambda _: (_ for _ in ()).throw(RuntimeError("stream closed"))
    )

    outcome = await DiscoveryDatasetsRunner(_details()).run("gdp", debug_inputs)

    assert outcome.rendered == "### Datasets\n\n- Alpha"
    assert outcome.eval_attachment.error is None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the relevance prompt ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class _FakeLlm:
    """Stands in for the chat model, capturing the messages the prompt template produced."""

    def __init__(self, sink: list[Any]) -> None:
        self._sink = sink

    def with_structured_output(self, schema: Any, method: str) -> Any:
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
