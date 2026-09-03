"""Tests for narrowing the discovery search before it ranks.

Two things decide whether this feature helps or hurts, and both are here: that a value which
reaches a request is one the channel actually holds - anything else is a rejected search rather
than a narrower one - and that every way this can fail leaves the lookup searching the whole
channel, as it did before pre-filtering existed.
"""

from typing import Any, Self

import pytest
from langchain_core.runnables import RunnableLambda

from statgpt.app.chains.discovery_datasets import prefilter as prefilter_module
from statgpt.app.chains.discovery_datasets.prefilter import (
    DiscoveryPreFilter,
    DiscoveryPreFilterBuilder,
)
from statgpt.app.schemas.discovery_datasets import (
    DiscoveryAxisSelection,
    DiscoveryPreFilterAxisReport,
)
from statgpt.common.schemas import (
    REFERENCE_AREA_KIND,
    DiscoveryPreFilterAxis,
    GenericRagDocument,
    GenericRagDocumentFilter,
    GenericRagDocumentMatcher,
    GenericRagMetadataSchema,
    ReferenceAreaRole,
)
from statgpt.common.schemas.discovery_datasets_tool import (
    DiscoveryDatasetsDetails,
    DiscoveryDatasetsTemplates,
)
from statgpt.common.services.generic_rag import GenericRagChannelError
from statgpt.common.utils import AsyncLoadingCache

_APPLICATION = "generic-rag-app"
_AREA_APPLICATION = "generic-rag-areas"
_CHANNEL = "statgpt-gtdc"

_DIMENSIONS: dict[str, list[str]] = {
    "agency": ["IMF", "Eurostat"],
    "parsed_reference_areas": ["France", "Germany", "Euro area"],
    "parsed_partner_reference_areas": ["China"],
    "parsed_frequencies": ["Monthly", "Annual"],
}

_AXIS_MARKERS = {
    "# Available reference areas": DiscoveryPreFilterAxis.REFERENCE_AREA,
    "# Available partner areas": DiscoveryPreFilterAxis.PARTNER_REFERENCE_AREA,
    "# Available frequencies": DiscoveryPreFilterAxis.FREQUENCY,
    "# Available agencies": DiscoveryPreFilterAxis.AGENCY,
}
"""How the fake model tells which axis it is being asked about.

The sub-chains differ only in the prompt they carry, and the branches of a `RunnableParallel`
all receive the same input - so the prompt is the only thing a stand-in can dispatch on.
"""


def _details(**overrides: Any) -> DiscoveryDatasetsDetails:
    fields: dict[str, Any] = {
        "application_id": _APPLICATION,
        "reference_area_application_id": _AREA_APPLICATION,
        "top_n": 5,
        "templates": DiscoveryDatasetsTemplates(wrapper="{items}", item="- {name}"),
    }
    fields.update(overrides)
    return DiscoveryDatasetsDetails(**fields)


def _area_document(document_id: int, value: str, *roles: ReferenceAreaRole) -> GenericRagDocument:
    return GenericRagDocument(
        id=document_id,
        display_name=f"{value}.txt",
        metadata={
            "kind": REFERENCE_AREA_KIND,
            "statgpt_channel": _CHANNEL,
            "value": value,
            "roles": sorted(roles or (ReferenceAreaRole.SUBJECT,)),
        },
    )


def _requested_role(matcher: GenericRagDocumentMatcher | None) -> str | None:
    """The role a vocabulary search asked for, if it asked for one."""
    if matcher is None or not matcher.filters:
        return None
    return matcher.filters[0].roles


class _FakeClient:
    """Stands in for `GenericRagSearchClient`, recording what was asked of it."""

    def __init__(
        self,
        *,
        dimensions: dict[str, list[str]] | None = None,
        documents: list[GenericRagDocument] | None = None,
        metadata_error: Exception | None = None,
        search_error: Exception | None = None,
        failing_role: ReferenceAreaRole | None = None,
    ) -> None:
        self._dimensions = dimensions if dimensions is not None else dict(_DIMENSIONS)
        self._documents = documents or []
        self._metadata_error = metadata_error
        self._search_error = search_error
        self._failing_role = failing_role
        self.searches: list[tuple[str, int, GenericRagDocumentMatcher | None]] = []
        self.metadata_reads = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get_metadata_schema(self) -> GenericRagMetadataSchema:
        self.metadata_reads += 1
        if self._metadata_error is not None:
            raise self._metadata_error
        return GenericRagMetadataSchema(schema={}, dimensions=self._dimensions)

    async def search_documents(
        self,
        query: str,
        limit: int,
        indexes: list[str] | None = None,
        matcher: GenericRagDocumentMatcher | None = None,
    ) -> list[GenericRagDocument]:
        self.searches.append((query, limit, matcher))
        role = _requested_role(matcher)
        if self._search_error is not None and self._failing_role in (None, role):
            raise self._search_error
        if role is None:
            return self._documents
        return [
            document for document in self._documents if role in document.metadata.get("roles", [])
        ]


class _FakeLLM:
    """Answers each axis's sub-chain with a canned selection, or raises."""

    def __init__(
        self,
        answers: dict[DiscoveryPreFilterAxis, list[str]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._answers = answers or {}
        self._error = error
        self.asked: list[DiscoveryPreFilterAxis] = []

    def with_structured_output(self, schema: Any, method: str | None = None) -> Any:
        return RunnableLambda(self._answer)

    def _answer(self, value: Any) -> DiscoveryAxisSelection:
        if self._error is not None:
            raise self._error
        text = value.to_string()
        for marker, axis in _AXIS_MARKERS.items():
            if marker in text:
                self.asked.append(axis)
                return DiscoveryAxisSelection(values=list(self._answers.get(axis, [])))
        raise AssertionError(f"A sub-chain was given an unrecognizable prompt: {text[:200]}")


class _AuthContext:
    api_key: str = "user-key"


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dimensions cache is module-level, so it would otherwise outlive a test."""
    monkeypatch.setattr(prefilter_module, "_dimensions_cache", AsyncLoadingCache(ttl=None))


def _install(
    monkeypatch: pytest.MonkeyPatch,
    llm: _FakeLLM | None = None,
    discovery: _FakeClient | None = None,
    areas: _FakeClient | None = None,
) -> tuple[_FakeLLM, dict[str, _FakeClient]]:
    llm = llm or _FakeLLM()
    clients = {
        _APPLICATION: discovery or _FakeClient(),
        _AREA_APPLICATION: areas
        or _FakeClient(
            documents=[
                _area_document(1, "France"),
                _area_document(2, "Euro area"),
                _area_document(3, "China", ReferenceAreaRole.PARTNER),
            ]
        ),
    }
    monkeypatch.setattr(
        prefilter_module.GenericRagSearchClient,
        "for_application",
        classmethod(lambda cls, application_id, auth_context: clients[application_id]),
    )
    monkeypatch.setattr(prefilter_module, "get_chat_model", lambda **_: llm)
    return llm, clients


async def _build(
    monkeypatch: pytest.MonkeyPatch,
    answers: dict[DiscoveryPreFilterAxis, list[str]] | None = None,
    *,
    details: DiscoveryDatasetsDetails | None = None,
    discovery: _FakeClient | None = None,
    areas: _FakeClient | None = None,
    llm: _FakeLLM | None = None,
    query: str = "gdp in france",
) -> tuple[DiscoveryPreFilter, _FakeLLM, dict[str, _FakeClient]]:
    llm, clients = _install(monkeypatch, llm or _FakeLLM(answers), discovery=discovery, areas=areas)
    result = await DiscoveryPreFilterBuilder(details or _details()).build(
        query, _AuthContext(), _CHANNEL  # type: ignore[arg-type]
    )
    return result, llm, clients


def _report_of(
    result: DiscoveryPreFilter, axis: DiscoveryPreFilterAxis
) -> DiscoveryPreFilterAxisReport:
    return next(item for item in result.report.axes if item.axis == axis)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ building the matcher ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_the_axes_combine_into_a_cross_product(monkeypatch: pytest.MonkeyPatch) -> None:
    """A filter entry takes one value per field, so several values mean several entries."""
    result, _, _ = await _build(
        monkeypatch,
        {
            DiscoveryPreFilterAxis.REFERENCE_AREA: ["France", "Euro area"],
            DiscoveryPreFilterAxis.FREQUENCY: ["Monthly"],
            DiscoveryPreFilterAxis.AGENCY: ["IMF", "Eurostat"],
        },
    )

    assert result.matcher == GenericRagDocumentMatcher(
        filters=[
            GenericRagDocumentFilter(
                statgpt_channel=_CHANNEL,
                parsed_reference_areas=area,
                parsed_frequencies="Monthly",
                agency=agency,
            )
            for area in ("France", "Euro area")
            for agency in ("IMF", "Eurostat")
        ]
    )
    assert result.report.filters == 4
    assert result.report.fallback_reason is None


async def test_every_entry_carries_the_channel_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without it, one entry would let another channel's documents through the whole matcher."""
    result, _, _ = await _build(
        monkeypatch, {DiscoveryPreFilterAxis.REFERENCE_AREA: ["France", "Germany"]}
    )

    assert result.matcher is not None
    assert [entry.statgpt_channel for entry in result.matcher.filters] == [_CHANNEL, _CHANNEL]


async def test_an_axis_with_no_selection_drops_out_of_the_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It must not empty the product: a query naming no frequency restricts no frequency."""
    result, _, _ = await _build(monkeypatch, {DiscoveryPreFilterAxis.AGENCY: ["IMF"]})

    assert result.matcher == GenericRagDocumentMatcher(
        filters=[GenericRagDocumentFilter(statgpt_channel=_CHANNEL, agency="IMF")]
    )


async def test_the_two_area_axes_are_alternatives_rather_than_a_further_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record covering an area as its subject and one covering it as a partner both answer.

    So the two area axes share a slot of the product: multiplying them together would demand
    both at once, which no record naming one country twice would satisfy.
    """
    discovery = _FakeClient(
        dimensions={
            "parsed_reference_areas": ["USA", "Canada"],
            "parsed_partner_reference_areas": ["USA", "Canada"],
            "parsed_frequencies": ["Annual"],
        }
    )
    areas = _FakeClient(
        documents=[
            _area_document(1, "USA", ReferenceAreaRole.SUBJECT, ReferenceAreaRole.PARTNER),
            _area_document(2, "Canada", ReferenceAreaRole.SUBJECT, ReferenceAreaRole.PARTNER),
        ]
    )
    both = ["USA", "Canada"]

    result, _, _ = await _build(
        monkeypatch,
        {
            DiscoveryPreFilterAxis.REFERENCE_AREA: both,
            DiscoveryPreFilterAxis.PARTNER_REFERENCE_AREA: both,
            DiscoveryPreFilterAxis.FREQUENCY: ["Annual"],
        },
        discovery=discovery,
        areas=areas,
    )

    assert result.matcher == GenericRagDocumentMatcher(
        filters=[
            GenericRagDocumentFilter(
                statgpt_channel=_CHANNEL,
                parsed_reference_areas="USA",
                parsed_frequencies="Annual",
            ),
            GenericRagDocumentFilter(
                statgpt_channel=_CHANNEL,
                parsed_reference_areas="Canada",
                parsed_frequencies="Annual",
            ),
            GenericRagDocumentFilter(
                statgpt_channel=_CHANNEL,
                parsed_partner_reference_areas="USA",
                parsed_frequencies="Annual",
            ),
            GenericRagDocumentFilter(
                statgpt_channel=_CHANNEL,
                parsed_partner_reference_areas="Canada",
                parsed_frequencies="Annual",
            ),
        ]
    )


async def test_dropping_the_partner_axis_leaves_one_vocabulary_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A channel that does not want partner matches must not pay for their vocabulary either."""
    areas = _FakeClient(documents=[_area_document(1, "France")])
    result, llm, _ = await _build(
        monkeypatch,
        {DiscoveryPreFilterAxis.REFERENCE_AREA: ["France"]},
        details=_details(pre_filter={"axes": ["reference_area", "frequency", "agency"]}),
        areas=areas,
    )

    assert [_requested_role(matcher) for _, _, matcher in areas.searches] == [
        ReferenceAreaRole.SUBJECT
    ]
    assert DiscoveryPreFilterAxis.PARTNER_REFERENCE_AREA not in llm.asked
    assert result.matcher == GenericRagDocumentMatcher(
        filters=[
            GenericRagDocumentFilter(statgpt_channel=_CHANNEL, parsed_reference_areas="France")
        ]
    )


async def test_only_the_configured_axes_are_asked_about(monkeypatch: pytest.MonkeyPatch) -> None:
    result, llm, _ = await _build(
        monkeypatch,
        {DiscoveryPreFilterAxis.AGENCY: ["IMF"], DiscoveryPreFilterAxis.FREQUENCY: ["Monthly"]},
        details=_details(pre_filter={"axes": ["agency"]}),
    )

    assert llm.asked == [DiscoveryPreFilterAxis.AGENCY]
    assert [item.axis for item in result.report.axes] == [DiscoveryPreFilterAxis.AGENCY]
    assert result.matcher == GenericRagDocumentMatcher(
        filters=[GenericRagDocumentFilter(statgpt_channel=_CHANNEL, agency="IMF")]
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ grounding ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_value_the_channel_does_not_hold_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service validates a filter value against its dimensions and rejects the request.

    So an ungrounded value is not a clause that matches nothing - it is a 500-shaped answer to
    the whole search. It has to be dropped here.
    """
    result, _, _ = await _build(
        monkeypatch,
        {DiscoveryPreFilterAxis.REFERENCE_AREA: ["France", "Atlantis"]},
    )

    assert result.matcher == GenericRagDocumentMatcher(
        filters=[
            GenericRagDocumentFilter(statgpt_channel=_CHANNEL, parsed_reference_areas="France")
        ]
    )
    report = _report_of(result, DiscoveryPreFilterAxis.REFERENCE_AREA)
    assert report.selected == ["France", "Atlantis"]
    assert report.grounded == ["France"]


async def test_a_recased_value_is_kept_under_the_channels_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-cased answer is a correct answer, and the service compares values exactly."""
    result, _, _ = await _build(monkeypatch, {DiscoveryPreFilterAxis.REFERENCE_AREA: ["EURO AREA"]})

    assert result.matcher == GenericRagDocumentMatcher(
        filters=[
            GenericRagDocumentFilter(statgpt_channel=_CHANNEL, parsed_reference_areas="Euro area")
        ]
    )


async def test_two_selections_folding_onto_one_value_make_one_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both are one requirement, so keeping both would only duplicate the cross product."""
    result, _, _ = await _build(
        monkeypatch, {DiscoveryPreFilterAxis.REFERENCE_AREA: ["France", "france", " FRANCE "]}
    )

    assert result.matcher == GenericRagDocumentMatcher(
        filters=[
            GenericRagDocumentFilter(statgpt_channel=_CHANNEL, parsed_reference_areas="France")
        ]
    )
    assert _report_of(result, DiscoveryPreFilterAxis.REFERENCE_AREA).grounded == ["France"]


async def test_a_padded_value_is_recovered_rather_than_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever a model pads its answer with, the value it named is the value it meant."""
    result, _, _ = await _build(monkeypatch, {DiscoveryPreFilterAxis.AGENCY: ["  IMF ", "   "]})

    assert result.matcher == GenericRagDocumentMatcher(
        filters=[GenericRagDocumentFilter(statgpt_channel=_CHANNEL, agency="IMF")]
    )
    assert _report_of(result, DiscoveryPreFilterAxis.AGENCY).selected == ["IMF"]


async def test_nothing_grounded_leaves_no_matcher(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _, _ = await _build(monkeypatch, {DiscoveryPreFilterAxis.REFERENCE_AREA: ["Atlantis"]})

    assert result.matcher is None
    assert result.report.filters == 0
    assert "named no value this channel holds" in (result.report.fallback_reason or "")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the vocabularies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_each_area_axis_searches_the_vocabulary_for_its_own_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two searches over the one channel, each scoped to its role and given its own limit."""
    areas = _FakeClient(
        documents=[
            _area_document(1, "France"),
            _area_document(2, "China", ReferenceAreaRole.PARTNER),
        ]
    )
    result, _, _ = await _build(
        monkeypatch,
        {DiscoveryPreFilterAxis.REFERENCE_AREA: ["France"]},
        details=_details(pre_filter={"referenceAreaTopN": 7}),
        areas=areas,
    )

    assert areas.searches == [
        (
            "gdp in france",
            7,
            GenericRagDocumentMatcher(
                filters=[
                    GenericRagDocumentFilter(
                        kind=REFERENCE_AREA_KIND, statgpt_channel=_CHANNEL, roles=role
                    )
                ]
            ),
        )
        for role in (ReferenceAreaRole.SUBJECT, ReferenceAreaRole.PARTNER)
    ]
    assert _report_of(result, DiscoveryPreFilterAxis.REFERENCE_AREA).offered == ["France"]
    assert _report_of(result, DiscoveryPreFilterAxis.PARTNER_REFERENCE_AREA).offered == ["China"]


async def test_the_offered_vocabularies_come_from_the_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = await _build(monkeypatch)

    assert _report_of(result, DiscoveryPreFilterAxis.AGENCY).offered == ["IMF", "Eurostat"]
    assert "Semi-annual" in _report_of(result, DiscoveryPreFilterAxis.FREQUENCY).offered
    assert _report_of(result, DiscoveryPreFilterAxis.REFERENCE_AREA).offered == [
        "France",
        "Euro area",
    ]
    assert _report_of(result, DiscoveryPreFilterAxis.PARTNER_REFERENCE_AREA).offered == ["China"]


async def test_the_dimensions_are_read_once_per_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner is built per turn, so the cache has to outlive it to be worth anything."""
    discovery = _FakeClient()
    llm, _ = _install(monkeypatch, _FakeLLM(), discovery=discovery)
    builder = DiscoveryPreFilterBuilder(_details())

    for _ in range(3):
        await builder.build("gdp", _AuthContext(), _CHANNEL)  # type: ignore[arg-type]

    assert discovery.metadata_reads == 1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ falling back ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_disabled_prefilter_asks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    result, llm, clients = await _build(
        monkeypatch, details=_details(pre_filter={"enabled": False})
    )

    assert result.matcher is None
    assert result.report.enabled is False
    assert result.report.fallback_reason == "pre-filtering is disabled for this channel"
    assert llm.asked == []
    assert clients[_APPLICATION].metadata_reads == 0


async def test_an_unreadable_channel_gives_up_on_every_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the channel's values nothing can be grounded, so nothing may be sent."""
    discovery = _FakeClient(metadata_error=GenericRagChannelError("metadata read", "boom", 503))
    result, llm, _ = await _build(
        monkeypatch,
        {DiscoveryPreFilterAxis.AGENCY: ["IMF"]},
        discovery=discovery,
    )

    assert result.matcher is None
    assert "could not be read" in (result.report.fallback_reason or "")
    assert llm.asked == []


async def test_a_failed_vocabulary_search_costs_only_its_own_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two area searches fail apart: a channel holding no partner rejects only that one."""
    areas = _FakeClient(
        documents=[_area_document(1, "France")],
        search_error=GenericRagChannelError("document search", "boom", 422),
        failing_role=ReferenceAreaRole.PARTNER,
    )
    result, *_ = await _build(
        monkeypatch,
        {DiscoveryPreFilterAxis.REFERENCE_AREA: ["France"]},
        areas=areas,
    )

    assert result.matcher == GenericRagDocumentMatcher(
        filters=[
            GenericRagDocumentFilter(statgpt_channel=_CHANNEL, parsed_reference_areas="France")
        ]
    )
    assert _report_of(result, DiscoveryPreFilterAxis.REFERENCE_AREA).offered == ["France"]
    partner = _report_of(result, DiscoveryPreFilterAxis.PARTNER_REFERENCE_AREA)
    assert partner.error == "the reference-area vocabulary could not be read"
    assert partner.offered == []


async def test_no_vocabulary_channel_leaves_the_other_axes_standing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = await _build(
        monkeypatch,
        {DiscoveryPreFilterAxis.AGENCY: ["IMF"]},
        details=_details(reference_area_application_id=None),
    )

    assert result.matcher == GenericRagDocumentMatcher(
        filters=[GenericRagDocumentFilter(statgpt_channel=_CHANNEL, agency="IMF")]
    )
    for axis in (
        DiscoveryPreFilterAxis.REFERENCE_AREA,
        DiscoveryPreFilterAxis.PARTNER_REFERENCE_AREA,
    ):
        report = _report_of(result, axis)
        assert report.offered == []
        assert report.error == "no reference-area vocabulary is configured for this channel"


async def test_a_failed_selection_falls_back_naming_the_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not "the query named nothing": a reviewer has to be able to tell the two apart."""
    result, _, _ = await _build(monkeypatch, llm=_FakeLLM(error=RuntimeError("boom")))

    assert result.matcher is None
    assert result.report.fallback_reason == "RuntimeError: boom"


async def test_an_unexpected_failure_falls_back_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lookup is decoration on someone else's answer; it must never fail the turn."""
    _install(monkeypatch)
    monkeypatch.setattr(
        DiscoveryPreFilterBuilder,
        "_axes",
        lambda self: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = await DiscoveryPreFilterBuilder(_details()).build(
        "gdp", _AuthContext(), _CHANNEL  # type: ignore[arg-type]
    )

    assert result.matcher is None
    assert result.report.fallback_reason == "RuntimeError: boom"
