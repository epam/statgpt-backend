"""Tests for deriving the indicator list a discovery document is published with.

The LLM is replaced at `get_chat_model`, below the structured-output and retry layers, so what
is tested includes how an answer past the output limits is retried and then failed.
"""

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from statgpt.admin.services import discovery_summarization as summarization_module
from statgpt.admin.services.discovery_summarization import DiscoveryIndicatorSummarizer
from statgpt.common import models, schemas
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.default_prompts import discovery_indexing_default_prompts

_UNIT = "Rp billions"
_MERGE_HEADING = "# Groups of every part"

Responder = Callable[[str, str], list[str]]
"""Answers one call, given its system and user message, with the groups the model returns."""


class _FakeLlm:
    """Stands in for the chat model, recording every call that reaches it."""

    def __init__(self, responder: Responder) -> None:
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema: type[BaseModel], **_: Any) -> RunnableLambda:
        def answer(prompt: ChatPromptValue) -> BaseModel:
            system, user = (str(message.content) for message in prompt.to_messages())
            self.calls.append((system, user))
            # Validated as the SDK validates a parsed answer, so a breach raises the same way.
            return schema.model_validate({"groups": self._responder(system, user)})

        return RunnableLambda(answer)


def _install(monkeypatch: pytest.MonkeyPatch, responder: Responder) -> _FakeLlm:
    llm = _FakeLlm(responder)
    monkeypatch.setattr(summarization_module, "get_chat_model", lambda **_: llm)
    return llm


def _groups(*groups: str) -> Responder:
    return lambda _system, _user: list(groups)


def _record(**overrides: Any) -> models.DiscoveryDataset:
    values: dict[str, Any] = {
        "id": 1,
        "agency": "Bank Indonesia (BI)",
        "dataset_id": "TABEL1_1",
        "indicators_coverage": f"M1 ({_UNIT}); M2 ({_UNIT})",
        "indicators_summary": "",
        "missing_indicators": "",
        "indicators_summary_source_hash": None,
        "indicators_summarized_at": None,
        "validation_status": schemas.DiscoveryValidationStatus.VALID,
        "indexing_status": schemas.DiscoveryIndexingStatus.NEW,
        "index_error": None,
    }
    values.update(overrides)
    return cast(models.DiscoveryDataset, SimpleNamespace(**values))


def _long_coverage(count: int = 60, unit: str = _UNIT) -> str:
    """A uniform-unit list longer than a threshold of 100 once compacted."""
    return "; ".join(f"Indicator number {i} ({unit})" for i in range(count))


def _config(**overrides: Any) -> schemas.DiscoveryIndexingConfig:
    values: dict[str, Any] = {"summary_threshold_chars": 100}
    values.update(overrides)
    return schemas.DiscoveryIndexingConfig(**values)


async def _no_persist() -> None:
    pass


def _summarizer(
    config: schemas.DiscoveryIndexingConfig | None = None,
    persist: Callable[[], Awaitable[None]] = _no_persist,
    **kwargs: Any,
) -> DiscoveryIndicatorSummarizer:
    kwargs.setdefault("concurrency", 4)
    kwargs.setdefault("timeout_seconds", 1)
    kwargs.setdefault("validation_retries", 2)
    auth_context = cast(AuthContext, SimpleNamespace(api_key="test-key"))
    return DiscoveryIndicatorSummarizer(config or _config(), auth_context, persist, **kwargs)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ copied lists ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_short_list_is_copied_with_its_shared_unit_stated_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _install(monkeypatch, _groups("unused"))
    record = _record()

    outcome = await _summarizer().summarize([record])

    assert record.indicators_summary == f"Unit: {_UNIT}; M1; M2"
    assert record.indicators_summarized_at is None
    assert record.indicators_summary_source_hash is not None
    assert (outcome.copied, outcome.summarized) == (1, 0)
    assert llm.calls == []


async def test_a_short_list_with_mixed_units_is_copied_as_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _groups("unused"))
    record = _record(indicators_coverage="M1  (Rp billions);; M2 (USD millions); m2 (USD millions)")

    await _summarizer().summarize([record])

    assert record.indicators_summary == "M1 (Rp billions); M2 (USD millions)"


async def test_the_threshold_is_measured_on_the_compacted_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Units repeated after every name would otherwise send a list to the LLM for nothing."""
    llm = _install(monkeypatch, _groups("unused"))
    coverage = "; ".join(f"I{i} ({_UNIT})" for i in range(8))
    record = _record(indicators_coverage=coverage)
    compacted = "Unit: Rp billions; " + "; ".join(f"I{i}" for i in range(8))
    assert len(coverage) > 100 >= len(compacted)

    await _summarizer().summarize([record])

    assert record.indicators_summary == compacted
    assert llm.calls == []


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ summarized lists ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_long_list_is_summarized_by_one_call_when_it_fits_one_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _install(monkeypatch, _groups("Indicators by number", "Other; more"))
    record = _record(indicators_coverage=_long_coverage())

    outcome = await _summarizer().summarize([record])

    assert len(llm.calls) == 1
    # The unit is the model's context, but the prefix is code's; ';' never survives in a group.
    assert record.indicators_summary == f"Unit: {_UNIT}; Indicators by number; Other, more"
    assert record.indicators_summarized_at is not None
    assert (outcome.summarized, outcome.copied) == (1, 0)
    _, user = llm.calls[0]
    assert _UNIT in user
    assert f"({_UNIT})" not in user
    assert "Indicator number 0\n" in user


async def test_a_list_longer_than_a_chunk_is_summarized_in_parts_and_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def responder(_system: str, user: str) -> list[str]:
        if _MERGE_HEADING in user:
            return ["Merged indicators"]
        first = next(line for line in user.splitlines() if line.startswith("Indicator number"))
        return [f"Group from {first}"]

    llm = _install(monkeypatch, responder)
    record = _record(indicators_coverage=_long_coverage())

    await _summarizer(_config(chunk_size_chars=400)).summarize([record])

    *chunk_calls, merge_call = llm.calls
    assert len(chunk_calls) > 1
    assert all(_MERGE_HEADING not in user for _, user in chunk_calls)
    # The merge is given every chunk's groups, and every call is given the unit.
    assert _MERGE_HEADING in merge_call[1]
    assert merge_call[1].count("Group from Indicator number") == len(chunk_calls)
    assert all(_UNIT in user for _, user in llm.calls)
    assert record.indicators_summary == f"Unit: {_UNIT}; Merged indicators"


async def test_chunking_never_splits_an_indicator(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _install(monkeypatch, _groups("Group"))
    names = [f"Indicator number {i}" for i in range(60)]
    record = _record(indicators_coverage=_long_coverage())

    await _summarizer(_config(chunk_size_chars=333)).summarize([record])

    chunked: list[str] = []
    for _, user in llm.calls[:-1]:
        lines = user.split("# Indicators, one per line\n", 1)[1].splitlines()
        assert sum(len(line) + 1 for line in lines) + len(_UNIT) <= 333
        chunked.extend(lines)
    assert chunked == names


async def test_an_indicator_longer_than_a_chunk_is_a_chunk_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _install(monkeypatch, _groups("Group"))
    long_name = "x" * 300
    record = _record(indicators_coverage=f"{long_name}; short one; another")

    await _summarizer(_config(chunk_size_chars=100)).summarize([record])

    first_chunk = llm.calls[0][1].split("# Indicators, one per line\n", 1)[1]
    assert first_chunk == long_name


async def test_a_configured_prompt_replaces_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _install(monkeypatch, _groups("Group"))
    prompts = schemas.DiscoveryIndexingPrompts(
        chunk_summary_prompt=schemas.SystemUserPrompt(
            system_message="Custom, at most {max_groups} groups.", user_message="{indicators}"
        )
    )

    await _summarizer(_config(prompts=prompts)).summarize(
        [_record(indicators_coverage=_long_coverage())]
    )

    system, user = llm.calls[0]
    assert system == "Custom, at most 30 groups."
    assert user.startswith("Indicator number 0\n")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ output limits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_an_answer_past_the_limits_is_asked_for_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter([["x" * 121], [], ["Fits"]])
    llm = _install(monkeypatch, lambda _system, _user: next(answers))
    record = _record(indicators_coverage=_long_coverage())

    outcome = await _summarizer().summarize([record])

    assert len(llm.calls) == 3
    assert record.indicators_summary == f"Unit: {_UNIT}; Fits"
    assert outcome.summarized == 1


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param([], id="no groups"),
        pytest.param(["   "], id="blank group"),
        pytest.param(["x" * 121], id="group too long"),
        pytest.param([f"g{i}" for i in range(31)], id="too many groups"),
    ],
)
async def test_an_answer_past_the_limits_after_every_retry_fails_the_record(
    monkeypatch: pytest.MonkeyPatch, answer: list[str]
) -> None:
    llm = _install(monkeypatch, lambda _system, _user: answer)
    record = _record(
        indicators_coverage=_long_coverage(),
        indicators_summary="the live summary",
        indicators_summary_source_hash="old-hash",
        indexing_status=schemas.DiscoveryIndexingStatus.INDEXED,
    )

    outcome = await _summarizer(validation_retries=2).summarize([record])

    assert len(llm.calls) == 3
    assert outcome.failed_ids == {1}
    assert record.indexing_status is schemas.DiscoveryIndexingStatus.FAILED
    assert record.index_error is not None
    assert record.index_error.startswith("Summarizing the indicator list failed:")
    # Left as it was, so the next run tries again and the live document stays consistent.
    assert record.indicators_summary == "the live summary"
    assert record.indicators_summary_source_hash == "old-hash"


async def test_one_failing_record_does_not_stop_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(_system: str, user: str) -> list[str]:
        if "broken" in user:
            raise RuntimeError("model unavailable")
        return ["Group"]

    _install(monkeypatch, responder)
    broken = _record(id=1, indicators_coverage=_long_coverage(unit="broken"))
    healthy = _record(id=2, indicators_coverage=_long_coverage())

    outcome = await _summarizer().summarize([broken, healthy])

    assert outcome.failed_ids == {1}
    assert outcome.summarized == 1
    assert "model unavailable" in (broken.index_error or "")
    assert healthy.indicators_summary == f"Unit: {_UNIT}; Group"
    assert healthy.indexing_status is schemas.DiscoveryIndexingStatus.NEW


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ persisting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_every_derived_record_is_persisted_as_soon_as_it_is_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run cut short keeps what it had derived; an unchanged record has nothing to persist."""
    _install(monkeypatch, _groups("Group"))
    summarized = _record(id=1, indicators_coverage=_long_coverage())
    copied = _record(id=2)
    unchanged = _record(id=3)
    await _summarizer().summarize([unchanged])
    persisted: list[dict[int, str]] = []
    records = [summarized, copied, unchanged]

    async def persist() -> None:
        persisted.append({record.id: record.indicators_summary for record in records})

    await _summarizer(persist=persist, concurrency=1).summarize(records)

    assert len(persisted) == 2
    assert persisted[0][1] == f"Unit: {_UNIT}; Group"
    assert persisted[1][2] == f"Unit: {_UNIT}; M1; M2"


async def test_a_failed_record_is_persisted_with_its_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def responder(_system: str, _user: str) -> list[str]:
        raise RuntimeError("model unavailable")

    _install(monkeypatch, responder)
    record = _record(indicators_coverage=_long_coverage())
    persisted: list[schemas.DiscoveryIndexingStatus] = []

    async def persist() -> None:
        persisted.append(record.indexing_status)

    await _summarizer(persist=persist).summarize([record])

    assert persisted == [schemas.DiscoveryIndexingStatus.FAILED]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ change detection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_an_unchanged_record_is_not_summarized_again(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _install(monkeypatch, _groups("Group"))
    record = _record(indicators_coverage=_long_coverage())
    await _summarizer().summarize([record])
    summarized_at = record.indicators_summarized_at

    outcome = await _summarizer().summarize([record])

    assert len(llm.calls) == 1
    assert outcome.unchanged == 1
    assert record.indicators_summarized_at == summarized_at


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"max_groups": 10}, id="output limit"),
        pytest.param({"summary_threshold_chars": 101}, id="threshold"),
        pytest.param(
            {
                "prompts": schemas.DiscoveryIndexingPrompts(
                    merge_summary_prompt=schemas.SystemUserPrompt(
                        system_message="Merge.", user_message="{groups}"
                    )
                )
            },
            id="prompt",
        ),
    ],
)
async def test_a_changed_config_derives_the_summary_again(
    monkeypatch: pytest.MonkeyPatch, changed: dict[str, Any]
) -> None:
    llm = _install(monkeypatch, _groups("Group"))
    record = _record(indicators_coverage=_long_coverage())
    await _summarizer().summarize([record])

    await _summarizer(_config(**changed)).summarize([record])

    assert len(llm.calls) == 2


async def test_a_prompt_left_unset_hashes_like_the_default_it_falls_back_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _install(monkeypatch, _groups("Group"))
    record = _record(indicators_coverage=_long_coverage())
    await _summarizer().summarize([record])
    spelled_out = schemas.DiscoveryIndexingPrompts(
        chunk_summary_prompt=discovery_indexing_default_prompts.chunk_summary_prompt,
        merge_summary_prompt=discovery_indexing_default_prompts.merge_summary_prompt,
    )

    outcome = await _summarizer(_config(prompts=spelled_out)).summarize([record])

    assert outcome.unchanged == 1
    assert len(llm.calls) == 1


async def test_a_changed_list_derives_the_summary_again(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _groups("Group"))
    record = _record()
    await _summarizer().summarize([record])

    record.indicators_coverage = f"M3 ({_UNIT})"
    await _summarizer().summarize([record])

    assert record.indicators_summary == f"Unit: {_UNIT}; M3"


async def test_a_forced_run_derives_every_summary_again(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _install(monkeypatch, _groups("Group"))
    record = _record(indicators_coverage=_long_coverage())
    await _summarizer().summarize([record])

    outcome = await _summarizer(force=True).summarize([record])

    assert len(llm.calls) == 2
    assert outcome.summarized == 1


async def test_a_summary_that_becomes_a_copy_drops_its_summarized_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _groups("Group"))
    record = _record(indicators_coverage=_long_coverage())
    await _summarizer().summarize([record])
    assert record.indicators_summarized_at is not None

    await _summarizer(_config(summary_threshold_chars=10_000)).summarize([record])

    assert record.indicators_summarized_at is None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ statuses ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_an_indexed_record_whose_summary_changed_is_outdated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _groups("unused"))
    record = _record(indexing_status=schemas.DiscoveryIndexingStatus.INDEXED)

    await _summarizer().summarize([record])

    assert record.indexing_status is schemas.DiscoveryIndexingStatus.OUTDATED


async def test_an_indexed_record_whose_summary_came_out_the_same_stays_indexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-derived because its hash is new, but its document already carries this summary."""
    _install(monkeypatch, _groups("unused"))
    record = _record(
        indicators_summary=f"Unit: {_UNIT}; M1; M2",
        indexing_status=schemas.DiscoveryIndexingStatus.INDEXED,
    )

    await _summarizer().summarize([record])

    assert record.indexing_status is schemas.DiscoveryIndexingStatus.INDEXED


async def test_old_template_missing_indicators_are_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _groups("unused"))
    record = _record(
        indicators_summary=f"Unit: {_UNIT}; M1; M2",
        missing_indicators="policy interest rates",
        indexing_status=schemas.DiscoveryIndexingStatus.INDEXED,
    )

    await _summarizer().summarize([record])

    assert record.missing_indicators == ""
    assert record.indexing_status is schemas.DiscoveryIndexingStatus.OUTDATED


async def test_an_invalid_record_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing is published for it, so there is nothing to derive."""
    _install(monkeypatch, _groups("unused"))
    record = _record(
        validation_status=schemas.DiscoveryValidationStatus.INVALID,
        indicators_summary="stale",
        missing_indicators="kept",
    )

    outcome = await _summarizer().summarize([record])

    assert (record.indicators_summary, record.missing_indicators) == ("stale", "kept")
    assert record.indicators_summary_source_hash is None
    assert outcome == summarization_module.SummarizeOutcome()
