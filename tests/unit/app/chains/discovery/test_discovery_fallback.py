"""Tests for the entry condition and orchestration of the discovery fallback."""

from unittest.mock import AsyncMock, Mock

import pytest

from statgpt.app.chains.discovery import fallback as fallback_module
from statgpt.app.chains.discovery.fallback import is_data_query_miss, refer_to_discovery
from statgpt.app.schemas.data_query_outcome import DataQueryStatus
from statgpt.app.schemas.discovery import (
    DiscoveryCandidate,
    DiscoveryReferralItem,
    DiscoverySearchResult,
)
from statgpt.common.schemas import DiscoveryFallbackConfig

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the entry condition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_no_data_is_a_miss():
    """No query survived, so Grade A has nothing for this request."""
    assert is_data_query_miss(DataQueryStatus.NO_DATA, DiscoveryFallbackConfig())


def test_executed_no_data_is_a_miss_by_default():
    """A cut StatGPT cannot serve is still a cut another official source may publish."""
    assert is_data_query_miss(DataQueryStatus.EXECUTED_NO_DATA, DiscoveryFallbackConfig())


def test_executed_no_data_can_be_switched_off():
    config = DiscoveryFallbackConfig(on_executed_no_data=False)

    assert not is_data_query_miss(DataQueryStatus.EXECUTED_NO_DATA, config)


@pytest.mark.parametrize(
    "status",
    [
        DataQueryStatus.DATASET_SELECTION_REQUIRED,
        DataQueryStatus.MISSING_DIMENSIONS,
    ],
)
def test_an_outcome_the_user_can_still_resolve_is_not_a_miss(status: DataQueryStatus):
    """Grade A can still answer once the user picks a dataset or supplies a value, so referring
    them elsewhere would abandon an answer that is still available."""
    assert not is_data_query_miss(status, DiscoveryFallbackConfig())


def test_an_error_is_not_a_miss():
    """`failed` means a fetch errored, so StatGPT does not know whether the data exists. Telling
    the user it lives elsewhere would be a guess dressed as help."""
    assert not is_data_query_miss(DataQueryStatus.FAILED, DiscoveryFallbackConfig())


@pytest.mark.parametrize(
    "status",
    [
        DataQueryStatus.DATA_AVAILABLE,
        DataQueryStatus.INVALID_TIME_PERIOD,
        DataQueryStatus.NOT_EXECUTED,
    ],
)
def test_the_remaining_outcomes_are_not_misses(status: DataQueryStatus):
    assert not is_data_query_miss(status, DiscoveryFallbackConfig())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ orchestration ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _data_service(application_id: str | None = "generic-rag-app") -> Mock:
    discovery_rag = None
    if application_id is not None:
        discovery_rag = Mock()
        discovery_rag.get_application_id.return_value = application_id

    service = Mock()
    service.channel_config = Mock(discovery_rag=discovery_rag)
    service.deployment_id = "statgpt-channel"
    return service


def _result(with_referral: bool = True) -> DiscoverySearchResult:
    if not with_referral:
        return DiscoverySearchResult(retrieved=3)
    return DiscoverySearchResult(
        retrieved=3,
        items=[
            DiscoveryReferralItem(
                candidate=DiscoveryCandidate(
                    document_id=1,
                    display_name="boj.md",
                    name="Monetary Base",
                    url="https://example.com/boj",
                )
            )
        ],
    )


def _patch_service(monkeypatch: pytest.MonkeyPatch, result=None, error: Exception | None = None):
    """Replace the search component, returning the mock so calls can be asserted on."""
    instance = AsyncMock()
    if error is not None:
        instance.search.side_effect = error
    else:
        instance.search.return_value = result if result is not None else _result()
    instance.__aenter__.return_value = instance
    instance.__aexit__.return_value = None

    factory = Mock(return_value=instance)
    monkeypatch.setattr(fallback_module, "DiscoverySearchService", factory)
    return factory, instance


async def _refer(monkeypatch, **overrides) -> str:
    kwargs = {
        "question": "monetary base in Japan",
        "status": DataQueryStatus.NO_DATA,
        "countries": ["Japan"],
        "config": DiscoveryFallbackConfig(enabled=True),
        "data_service": _data_service(),
        "auth_context": Mock(api_key="key"),
    }
    kwargs.update(overrides)
    return await refer_to_discovery(**kwargs)


async def test_a_referral_is_rendered_on_a_miss(monkeypatch: pytest.MonkeyPatch):
    _patch_service(monkeypatch)

    assert "Monetary Base" in await _refer(monkeypatch)


async def test_the_fallback_is_off_unless_enabled(monkeypatch: pytest.MonkeyPatch):
    """A channel that has not opted in produces a byte-identical response."""
    factory, _ = _patch_service(monkeypatch)

    assert await _refer(monkeypatch, config=DiscoveryFallbackConfig(enabled=False)) == ""
    factory.assert_not_called()


async def test_no_search_runs_when_the_outcome_is_not_a_miss(monkeypatch: pytest.MonkeyPatch):
    factory, _ = _patch_service(monkeypatch)

    assert await _refer(monkeypatch, status=DataQueryStatus.DATA_AVAILABLE) == ""
    factory.assert_not_called()


async def test_a_channel_without_a_discovery_application_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
):
    """The fallback needs somewhere to search. A channel holding no discovery records behaves
    exactly as it does with the fallback off."""
    factory, _ = _patch_service(monkeypatch)

    assert await _refer(monkeypatch, data_service=_data_service(application_id=None)) == ""
    factory.assert_not_called()


async def test_the_search_is_scoped_to_this_channel_and_its_application(
    monkeypatch: pytest.MonkeyPatch,
):
    factory, _ = _patch_service(monkeypatch)

    await _refer(monkeypatch)

    kwargs = factory.call_args.kwargs
    assert kwargs["application_id"] == "generic-rag-app"
    assert kwargs["statgpt_channel"] == "statgpt-channel"


async def test_the_question_and_the_countries_reach_the_search(monkeypatch: pytest.MonkeyPatch):
    """The countries come from the run that just failed, so the fallback needs no country prompt
    of its own."""
    _, instance = _patch_service(monkeypatch)

    await _refer(monkeypatch, question="broad money", countries=["Japan", "Indonesia"])

    instance.search.assert_awaited_once_with("broad money", ["Japan", "Indonesia"])


async def test_a_search_that_finds_nothing_relevant_adds_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_service(monkeypatch, result=_result(with_referral=False))

    assert await _refer(monkeypatch) == ""


async def test_a_failing_search_does_not_break_the_answer(monkeypatch: pytest.MonkeyPatch):
    """A referral is an extra on top of an answer the user is already getting, so a discovery
    failure is dropped rather than turned into a failed tool call."""
    _patch_service(monkeypatch, error=RuntimeError("channel unreachable"))

    assert await _refer(monkeypatch) == ""
