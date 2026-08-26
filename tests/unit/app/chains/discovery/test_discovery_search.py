"""Tests for the discovery search component: filtering, folding, and judging."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.runnables import RunnableLambda

from statgpt.app.chains.discovery import search as search_module
from statgpt.app.chains.discovery.search import DiscoverySearchService
from statgpt.app.schemas.discovery import DiscoveryJudgement, DiscoverySelection
from statgpt.common.schemas import DiscoveryFallbackConfig
from statgpt.common.schemas.generic_rag import GenericRagDocument
from statgpt.common.services.discovery_reference_area import SENTINEL

_CHANNEL = "statgpt-channel"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _document(
    document_id: int = 1,
    display_name: str = "Bank of Japan (BOJ) - TEST_BOJ_MB [abc123].md",
    *,
    areas: list[str] | None = None,
    grade: str = "C",
    channel: str = _CHANNEL,
    **metadata,
) -> GenericRagDocument:
    return GenericRagDocument(
        id=document_id,
        display_name=display_name,
        status="ready",
        metadata={
            "grade": grade,
            "statgpt_channel": channel,
            "reference_area_values": areas if areas is not None else ["Japan (JPN)"],
            "agency": "Bank of Japan (BOJ)",
            "dataset_id": "TEST_BOJ_MB",
            "name": "Monetary Base",
            "url": "https://example.com/boj",
            "reference_area": "Japan (JPN)",
            **metadata,
        },
    )


def _attachment(display_name: str, data: str = "chunk text", index: int = 1) -> dict:
    """An attachment shaped the way `create_attachment` builds one.

    The title is `f"[{citation_index}] {source_display_name}"`, which is the only handle the
    retrieval response gives on which document a chunk came from.
    """
    return {"type": "text/markdown", "title": f"[{index}] {display_name}", "data": data}


class _FakeCompletions:
    def __init__(self, attachments: list[dict]):
        self._attachments = attachments
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(
            model_extra={"custom_content": {"attachments": self._attachments}}
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _service(
    monkeypatch: pytest.MonkeyPatch,
    *,
    documents: list[GenericRagDocument] | None = None,
    attachments: list[dict] | None = None,
    judgement: DiscoveryJudgement | None = None,
    config: DiscoveryFallbackConfig | None = None,
) -> tuple[DiscoverySearchService, _FakeCompletions]:
    """Build a service with its channel client, retrieval call, and judge all faked."""
    client = AsyncMock(spec=search_module.GenericRagIngestionClient)
    client.list_documents.return_value = list(documents or [])
    monkeypatch.setattr(
        search_module.GenericRagIngestionClient,
        "for_application",
        classmethod(lambda cls, application_id: client),
    )

    completions = _FakeCompletions(list(attachments or []))
    monkeypatch.setattr(
        search_module.openai_utils,
        "get_async_client",
        lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    service = DiscoverySearchService(
        config=config or DiscoveryFallbackConfig(enabled=True),
        application_id="generic-rag-app",
        statgpt_channel=_CHANNEL,
        auth_context=Mock(api_key="key"),
    )
    verdict = judgement if judgement is not None else DiscoveryJudgement(selections=[])
    service._judge_chain = lambda: RunnableLambda(lambda _: verdict)  # type: ignore[method-assign]
    return service, completions


def _filters(completions: _FakeCompletions) -> list[dict]:
    configuration = completions.calls[0]["extra_body"]["custom_fields"]["configuration"]
    return configuration["retriever"]["document_selector"]["filters"]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the retrieval request ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_retrieval_asks_for_no_answer_generation(monkeypatch: pytest.MonkeyPatch):
    """Discovery owns its own selection step, so a generated answer would be a second, unowned
    prompt between the record and the user."""
    service, completions = _service(monkeypatch, documents=[_document()])

    await service.search("monetary base in Japan", ["Japan"])

    configuration = completions.calls[0]["extra_body"]["custom_fields"]["configuration"]
    assert configuration["generation"] == {"type": "retrieval_only"}


async def test_every_filter_entry_carries_the_grade_and_the_channel(
    monkeypatch: pytest.MonkeyPatch,
):
    """The service AND's the fields within an entry and OR's the entries, so an entry missing
    these would match another channel's documents."""
    service, completions = _service(monkeypatch, documents=[_document()])

    await service.search("monetary base", ["Japan"])

    for entry in _filters(completions):
        assert entry["grade"] == "C"
        assert entry["statgpt_channel"] == _CHANNEL


async def test_a_grounded_country_becomes_one_filter_entry_plus_the_sentinel(
    monkeypatch: pytest.MonkeyPatch,
):
    documents = [_document(areas=["Japan (JPN)"]), _document(2, "world.md", areas=[SENTINEL])]
    service, completions = _service(monkeypatch, documents=documents)

    await service.search("monetary base", ["Japan"])

    assert [entry["reference_area_values"] for entry in _filters(completions)] == [
        "Japan (JPN)",
        SENTINEL,
    ]


async def test_several_countries_become_several_entries(monkeypatch: pytest.MonkeyPatch):
    documents = [
        _document(1, "jp.md", areas=["Japan (JPN)"]),
        _document(2, "id.md", areas=["Indonesia (IDN)"]),
    ]
    service, completions = _service(monkeypatch, documents=documents)

    await service.search("broad money", ["Japan", "Indonesia"])

    assert {entry["reference_area_values"] for entry in _filters(completions)} == {
        "Japan (JPN)",
        "Indonesia (IDN)",
    }


async def test_a_country_the_channel_has_no_value_for_leaves_the_search_unfiltered(
    monkeypatch: pytest.MonkeyPatch,
):
    """An over-narrow filter loses records irrecoverably; an unfiltered search is a precision
    problem the judge absorbs. Sending an ungrounded value would fail the request outright,
    because the service types the field as a `Literal` of the values present."""
    service, completions = _service(monkeypatch, documents=[_document(areas=["Japan (JPN)"])])

    result = await service.search("Brazilian GDP", ["Brazil"])

    entries = _filters(completions)
    assert len(entries) == 1
    assert "reference_area_values" not in entries[0]
    assert result.unmatched_areas == ["Brazil"]
    assert result.grounded_areas == []


async def test_a_request_naming_no_country_is_unfiltered(monkeypatch: pytest.MonkeyPatch):
    service, completions = _service(monkeypatch, documents=[_document()])

    await service.search("broad money", [])

    assert "reference_area_values" not in _filters(completions)[0]


async def test_available_countries_come_only_from_this_channel_and_grade(
    monkeypatch: pytest.MonkeyPatch,
):
    """Grounding against the whole application's dimensions would let a value that exists only in
    another channel's records into the filter, narrowing this search to nothing."""
    documents = [
        _document(1, "ours.md", areas=["Japan (JPN)"]),
        _document(2, "theirs.md", areas=["Brazil (BRA)"], channel="another-channel"),
        _document(3, "gradeb.md", areas=["Canada (CAN)"], grade="B"),
    ]
    service, completions = _service(monkeypatch, documents=documents)

    result = await service.search("GDP", ["Brazil", "Canada"])

    assert result.grounded_areas == []
    assert "reference_area_values" not in _filters(completions)[0]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ folding chunks into datasets ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_chunks_of_one_document_fold_into_a_single_candidate(
    monkeypatch: pytest.MonkeyPatch,
):
    """Retrieval returns chunks; a referral is about datasets."""
    document = _document(display_name="boj.md")
    service, _ = _service(
        monkeypatch,
        documents=[document],
        attachments=[
            _attachment("boj.md", "first", index=1),
            _attachment("boj.md", "second", index=2),
        ],
        judgement=DiscoveryJudgement(selections=[DiscoverySelection(index=1, reason="covers it")]),
    )

    result = await service.search("monetary base", ["Japan"])

    assert result.retrieved == 1
    assert result.items[0].candidate.chunks == ["first", "second"]


async def test_a_candidate_carries_every_metadata_field_not_only_what_the_chunk_held(
    monkeypatch: pytest.MonkeyPatch,
):
    """The judge reads the negative fields to rule a dataset out, and the referral needs the URL
    whichever section of the record happened to match."""
    document = _document(
        display_name="boj.md",
        missing_indicators="gross domestic product, GDP",
        excluded_regional_values="Hokkaido is absent",
        time_coverage="From 1970-01 to present",
        frequency_coverage="Monthly",
        indicators_coverage="monetary base (in JPY billions)",
    )
    service, _ = _service(
        monkeypatch,
        documents=[document],
        attachments=[_attachment("boj.md")],
        judgement=DiscoveryJudgement(selections=[DiscoverySelection(index=1)]),
    )

    candidate = (await service.search("monetary base", ["Japan"])).items[0].candidate

    assert candidate.url == "https://example.com/boj"
    assert candidate.missing_indicators == "gross domestic product, GDP"
    assert candidate.excluded_regional_values == "Hokkaido is absent"
    assert candidate.label == "Monetary Base"


async def test_a_retrieved_document_that_is_not_ours_is_dropped(monkeypatch: pytest.MonkeyPatch):
    """The filter should have excluded it, and a candidate whose metadata cannot be read is one
    the referral could not link to."""
    service, _ = _service(
        monkeypatch,
        documents=[_document(display_name="ours.md")],
        attachments=[_attachment("ours.md"), _attachment("stranger.md")],
        judgement=DiscoveryJudgement(selections=[DiscoverySelection(index=1)]),
    )

    result = await service.search("monetary base", ["Japan"])

    assert result.retrieved == 1
    assert result.items[0].candidate.display_name == "ours.md"


async def test_the_citation_prefix_is_stripped_from_the_attachment_title(
    monkeypatch: pytest.MonkeyPatch,
):
    """The document's name is recovered from a title the application formats as '[3] name'."""
    service, _ = _service(
        monkeypatch,
        documents=[_document(display_name="boj.md")],
        attachments=[_attachment("boj.md", index=17)],
        judgement=DiscoveryJudgement(selections=[DiscoverySelection(index=1)]),
    )

    assert (await service.search("q", ["Japan"])).retrieved == 1


async def test_candidates_are_capped_before_the_judge_sees_them(monkeypatch: pytest.MonkeyPatch):
    documents = [_document(i, f"doc{i}.md") for i in range(1, 6)]
    service, _ = _service(
        monkeypatch,
        documents=documents,
        attachments=[_attachment(f"doc{i}.md") for i in range(1, 6)],
        config=DiscoveryFallbackConfig(enabled=True, max_candidates=2),
    )

    assert (await service.search("q", ["Japan"])).retrieved == 2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ judging ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_nothing_retrieved_means_no_referral(monkeypatch: pytest.MonkeyPatch):
    service, _ = _service(monkeypatch, documents=[_document()], attachments=[])

    result = await service.search("q", ["Japan"])

    assert result.retrieved == 0
    assert not result.has_referral


async def test_a_judge_that_selects_nothing_produces_no_referral(
    monkeypatch: pytest.MonkeyPatch,
):
    """Retrieval always returns something, so this is the common path: the index holds datasets
    for this country but none that answers the question."""
    service, _ = _service(
        monkeypatch,
        documents=[_document(display_name="boj.md")],
        attachments=[_attachment("boj.md")],
        judgement=DiscoveryJudgement(selections=[]),
    )

    result = await service.search("q", ["Japan"])

    assert result.retrieved == 1
    assert not result.has_referral


async def test_the_judges_reason_and_missing_travel_with_the_referral(
    monkeypatch: pytest.MonkeyPatch,
):
    service, _ = _service(
        monkeypatch,
        documents=[_document(display_name="boj.md")],
        attachments=[_attachment("boj.md")],
        judgement=DiscoveryJudgement(
            selections=[
                DiscoverySelection(index=1, reason="publishes the monetary base", missing="GDP")
            ]
        ),
    )

    item = (await service.search("q", ["Japan"])).items[0]

    assert item.reason == "publishes the monetary base"
    assert item.missing == "GDP"


async def test_selections_keep_the_judges_order(monkeypatch: pytest.MonkeyPatch):
    documents = [_document(1, "a.md"), _document(2, "b.md")]
    service, _ = _service(
        monkeypatch,
        documents=documents,
        attachments=[_attachment("a.md"), _attachment("b.md")],
        judgement=DiscoveryJudgement(
            selections=[DiscoverySelection(index=2), DiscoverySelection(index=1)]
        ),
    )

    items = (await service.search("q", ["Japan"])).items

    assert [item.candidate.display_name for item in items] == ["b.md", "a.md"]


@pytest.mark.parametrize("index", [0, 3, -1, 99])
async def test_an_out_of_range_selection_is_discarded(monkeypatch: pytest.MonkeyPatch, index: int):
    """A hallucinated number must not raise, and must not silently refer to the wrong dataset."""
    service, _ = _service(
        monkeypatch,
        documents=[_document(display_name="boj.md")],
        attachments=[_attachment("boj.md")],
        judgement=DiscoveryJudgement(selections=[DiscoverySelection(index=index)]),
    )

    assert not (await service.search("q", ["Japan"])).has_referral


async def test_referrals_are_capped(monkeypatch: pytest.MonkeyPatch):
    documents = [_document(i, f"doc{i}.md") for i in range(1, 4)]
    service, _ = _service(
        monkeypatch,
        documents=documents,
        attachments=[_attachment(f"doc{i}.md") for i in range(1, 4)],
        judgement=DiscoveryJudgement(selections=[DiscoverySelection(index=i) for i in range(1, 4)]),
        config=DiscoveryFallbackConfig(enabled=True, max_referrals=2),
    )

    assert len((await service.search("q", ["Japan"])).items) == 2


async def test_a_failing_judge_leaves_the_answer_without_a_referral(
    monkeypatch: pytest.MonkeyPatch,
):
    """A referral is an extra on top of an answer the user is already getting."""
    service, _ = _service(
        monkeypatch,
        documents=[_document(display_name="boj.md")],
        attachments=[_attachment("boj.md")],
    )

    def _boom(_):
        raise RuntimeError("model unavailable")

    service._judge_chain = lambda: RunnableLambda(_boom)  # type: ignore[method-assign]

    result = await service.search("q", ["Japan"])

    assert result.retrieved == 1
    assert not result.has_referral


async def test_a_failing_retrieval_call_yields_no_candidates(monkeypatch: pytest.MonkeyPatch):
    from openai import APIError

    service, completions = _service(monkeypatch, documents=[_document()])

    async def _boom(**kwargs):
        raise APIError("down", request=Mock(), body=None)

    completions.create = _boom  # type: ignore[method-assign]

    result = await service.search("q", ["Japan"])

    assert result.retrieved == 0
    assert not result.has_referral


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the judge's candidate list ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_the_negative_fields_are_labeled_as_exclusions_for_the_judge():
    """Retrieval cannot honor them - a record naming an indicator under "not present" contains
    that indicator's words - so their meaning must not depend on the judge inferring it from a
    workbook column heading."""
    from statgpt.app.schemas.discovery import DiscoveryCandidate

    rendered = DiscoverySearchService._render_candidates(
        [
            DiscoveryCandidate(
                document_id=1,
                display_name="boj.md",
                name="Monetary Base",
                missing_indicators="gross domestic product, GDP",
                excluded_regional_values="Hokkaido is absent",
            )
        ]
    )

    assert "1. Monetary Base" in rendered
    assert "Indicators NOT present: gross domestic product, GDP" in rendered
    assert "Excluded regions: Hokkaido is absent" in rendered


def test_empty_fields_are_omitted_from_the_candidate_list():
    from statgpt.app.schemas.discovery import DiscoveryCandidate

    rendered = DiscoverySearchService._render_candidates(
        [DiscoveryCandidate(document_id=1, display_name="d.md", name="Only A Name")]
    )

    assert "Indicators NOT present" not in rendered
    assert rendered.strip() == "1. Only A Name"
