"""Tests for deriving the indicator list a discovery document is published with.

The LLM is replaced at `get_chat_model`, below the structured-output format and the answer's
validation, so what is tested includes how an invalid answer is corrected and then failed.
"""

import json
from collections.abc import Awaitable, Callable, Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from statgpt.admin.services import discovery_summarization as summarization_module
from statgpt.admin.services.discovery_summarization import DiscoveryIndicatorSummarizer
from statgpt.common import models, schemas
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.default_prompts import discovery_indexing_default_prompts

_UNIT = "Rp billions"
_MERGE_HEADING = "# Groups of every part"
_LIMIT = schemas.DiscoveryIndexingConfig().max_summary_chars
_TOO_LONG = "x" * _LIMIT
"""One group as long as the limit: past it once published, with the unit prefix."""

Answer = list[str] | dict[str, Any] | str
Responder = Callable[[str, str], Answer]
"""Answers one call, given its system and first user message: with the groups the model
returns, with its whole answer, or with the raw text of one."""


def _answer_text(answer: Answer, schema: dict[str, Any]) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        answer = {"reasoning": "Grouped by subject.", "groups": answer}
        if "units" in schema["properties"]:
            answer["units"] = None
    return json.dumps(answer)


class _FakeLlm:
    """Stands in for the chat model, recording every call that reaches it."""

    def __init__(self, responder: Responder) -> None:
        self._responder = responder
        self.calls: list[tuple[str, str]] = []
        self.conversations: list[list[BaseMessage]] = []
        self.schemas: list[dict[str, Any]] = []

    def bind(self, *, response_format: dict[str, Any]) -> RunnableLambda:
        schema = response_format["json_schema"]["schema"]

        def answer(messages: list[BaseMessage]) -> AIMessage:
            system, user = (str(message.content) for message in messages[:2])
            self.calls.append((system, user))
            self.conversations.append(list(messages))
            self.schemas.append(schema)
            return AIMessage(content=_answer_text(self._responder(system, user), schema))

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
        "indicators_summary_calls": None,
        "indicators_summary_failed_calls": None,
        "validation_status": schemas.DiscoveryValidationStatus.VALID,
        "indexing_status": schemas.DiscoveryIndexingStatus.NEW,
        "index_error": None,
    }
    values.update(overrides)
    return cast(models.DiscoveryDataset, SimpleNamespace(**values))


def _long_coverage(count: int = 60, unit: str = _UNIT) -> str:
    """A uniform-unit list longer than a threshold of 100 once compacted."""
    return "; ".join(f"Indicator number {i} ({unit})" for i in range(count))


def _mixed_coverage(count: int = 60) -> str:
    """A list longer than a threshold of 100 whose indicators share no unit."""
    return "; ".join(
        f"Indicator number {i} ({'Rp billions' if i % 2 else 'USD millions'})" for i in range(count)
    )


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
    system, user = llm.calls[0]
    assert f"Every indicator is measured in {_UNIT}." in system
    assert _UNIT not in user
    assert "Indicator number 0\n" in user


async def test_units_that_differ_are_left_in_the_list_for_the_model_to_summarize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _install(monkeypatch, _groups("Money supply (Rp billions, USD millions)"))
    record = _record(indicators_coverage=_mixed_coverage())

    await _summarizer().summarize([record])

    system, user = llm.calls[0]
    assert "Summarize them as well" in system
    assert "Every indicator is measured in" not in system
    assert "Indicator number 0 (USD millions)\n" in user
    # A null `units` leaves the units in the groups, and adds no prefix.
    assert record.indicators_summary == "Money supply (Rp billions, USD millions)"


async def test_units_the_model_summarized_as_one_value_are_published_as_the_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = {"reasoning": "...", "units": "Rp billions; USD millions", "groups": ["Money supply"]}
    _install(monkeypatch, lambda _system, _user: answer)
    record = _record(indicators_coverage=_mixed_coverage())

    await _summarizer().summarize([record])

    assert record.indicators_summary == "Unit: Rp billions, USD millions; Money supply"


async def test_the_units_field_is_offered_only_when_the_indicators_share_no_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _install(monkeypatch, _groups("Group"))

    await _summarizer(concurrency=1).summarize(
        [
            _record(id=1, indicators_coverage=_long_coverage()),
            _record(id=2, indicators_coverage=_mixed_coverage()),
        ]
    )

    shared, own = llm.schemas
    # In this order, so the model reasons before it commits to units and groups.
    assert list(shared["properties"]) == ["reasoning", "groups"]
    assert list(own["properties"]) == ["reasoning", "units", "groups"]
    assert own["required"] == ["reasoning", "units", "groups"]
    assert own["additionalProperties"] is False


async def test_a_list_longer_than_a_chunk_is_summarized_in_parts_and_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def responder(_system: str, user: str) -> Answer:
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
    assert all(_UNIT in system for system, _ in llm.calls)
    assert record.indicators_summary == f"Unit: {_UNIT}; Merged indicators"


async def test_the_merge_is_given_the_units_each_part_summarized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def responder(_system: str, user: str) -> Answer:
        if _MERGE_HEADING in user:
            return {"reasoning": "...", "units": None, "groups": ["Merged (Rp billions, USD)"]}
        if "Indicator number 0 " in user:
            return {"reasoning": "...", "units": "USD millions", "groups": ["First"]}
        return {"reasoning": "...", "units": None, "groups": ["Later (Rp billions)"]}

    llm = _install(monkeypatch, responder)
    record = _record(indicators_coverage=_mixed_coverage())

    await _summarizer(_config(chunk_size_chars=400)).summarize([record])

    merge_input = llm.calls[-1][1]
    assert merge_input.startswith(
        f"{_MERGE_HEADING}\n## Part 1\nUnits of every group: USD millions\nFirst\n\n## Part 2\n"
    )
    assert "Later (Rp billions)" in merge_input
    assert record.indicators_summary == "Merged (Rp billions, USD)"


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
            system_message="Custom, at most {max_summary_chars} chars.",
            user_message="{indicators}",
        )
    )

    await _summarizer(_config(prompts=prompts)).summarize(
        [_record(indicators_coverage=_long_coverage())]
    )

    system, user = llm.calls[0]
    assert system == f"Custom, at most {_LIMIT} chars."
    assert user.startswith("Indicator number 0\n")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ output limit ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_an_invalid_answer_is_corrected_until_it_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers: Iterator[Answer] = iter([[_TOO_LONG], [], ["Fits"]])
    llm = _install(monkeypatch, lambda _system, _user: next(answers))
    record = _record(indicators_coverage=_long_coverage())

    outcome = await _summarizer().summarize([record])

    # Every correction continues the conversation rather than starting it over.
    assert [len(conversation) for conversation in llm.conversations] == [2, 4, 6]
    assert record.indicators_summary == f"Unit: {_UNIT}; Fits"
    assert outcome.summarized == 1


async def test_the_limit_holds_for_the_published_form_and_its_breach_is_explained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fits = f"Unit: {_UNIT}; Fits"
    answers: Iterator[Answer] = iter([["Fits", "more"], ["Fits"]])
    llm = _install(monkeypatch, lambda _system, _user: next(answers))
    record = _record(indicators_coverage=_long_coverage())

    await _summarizer(_config(max_summary_chars=len(fits))).summarize([record])

    assert record.indicators_summary == fits
    *_, rejected, feedback = llm.conversations[1]
    assert isinstance(rejected, AIMessage)
    assert json.loads(str(rejected.content))["groups"] == ["Fits", "more"]
    assert isinstance(feedback, HumanMessage)
    # The prefix and the separators count: "Unit: Rp billions; Fits; more".
    published = len(f"{fits}; more")
    assert f"{published} characters long, 6 over the limit of {len(fits)}" in str(feedback.content)


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param([], id="no groups"),
        pytest.param(["   "], id="blank group"),
        pytest.param([_TOO_LONG], id="past the limit"),
        pytest.param("not json", id="unparseable"),
    ],
)
async def test_an_invalid_answer_after_every_correction_fails_the_record(
    monkeypatch: pytest.MonkeyPatch, answer: Answer
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
    def responder(system: str, _user: str) -> Answer:
        if "broken" in system:
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ stored calls ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _calls(stored: list[dict[str, Any]] | None) -> list[schemas.DiscoverySummaryCall]:
    assert stored is not None
    return [schemas.DiscoverySummaryCall.model_validate(call) for call in stored]


async def test_every_answer_of_a_summary_is_stored_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers: Iterator[Answer] = iter([[_TOO_LONG], "not json", ["Fits"]])
    _install(monkeypatch, lambda _system, _user: next(answers))
    record = _record(indicators_coverage=_long_coverage(), indicators_summary_failed_calls=[])

    await _summarizer().summarize([record])

    rejected, unparseable, accepted = _calls(record.indicators_summary_calls)
    assert rejected.answer == {"reasoning": "Grouped by subject.", "groups": [_TOO_LONG]}
    assert rejected.error is not None and "over the limit" in rejected.error
    # Its text is not kept, but the entry is: the retries are the entries with an error.
    assert unparseable.answer is None
    assert unparseable.error is not None
    assert accepted.answer == {"reasoning": "Grouped by subject.", "groups": ["Fits"]}
    assert accepted.error is None
    assert {(call.step, call.part) for call in (rejected, unparseable, accepted)} == {
        (schemas.DiscoverySummaryStep.CHUNK, 1)
    }
    assert record.indicators_summary_failed_calls is None


async def test_the_stored_calls_name_the_part_or_the_merge_they_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _install(monkeypatch, _groups("Group"))
    record = _record(indicators_coverage=_long_coverage())

    await _summarizer(_config(chunk_size_chars=400)).summarize([record])

    steps = [(call.step, call.part) for call in _calls(record.indicators_summary_calls)]
    chunk_count = len(llm.calls) - 1
    assert steps == [
        *((schemas.DiscoverySummaryStep.CHUNK, number) for number in range(1, chunk_count + 1)),
        (schemas.DiscoverySummaryStep.MERGE, None),
    ]


async def test_a_copied_list_stores_no_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _groups("Group"))
    record = _record(indicators_coverage=_long_coverage())
    await _summarizer().summarize([record])
    record.indicators_summary_failed_calls = []

    await _summarizer(_config(summary_threshold_chars=10_000)).summarize([record])

    assert record.indicators_summary_calls is None
    assert record.indicators_summary_failed_calls is None


async def test_a_failed_summary_stores_its_answers_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _groups("Group"))
    record = _record(indicators_coverage=_long_coverage())
    await _summarizer().summarize([record])
    published_calls = record.indicators_summary_calls
    _install(monkeypatch, _groups())
    record.indicators_coverage = _long_coverage(count=61)

    await _summarizer(validation_retries=1).summarize([record])

    # The published summary's calls stay with it; the failure's sit beside them.
    assert record.indicators_summary_calls == published_calls
    failed = _calls(record.indicators_summary_failed_calls)
    assert len(failed) == 2
    assert all(call.error is not None for call in failed)


async def test_a_summary_that_fails_before_any_answer_stores_no_failed_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def responder(_system: str, _user: str) -> Answer:
        raise RuntimeError("model unavailable")

    _install(monkeypatch, responder)
    record = _record(indicators_coverage=_long_coverage())

    await _summarizer().summarize([record])

    assert record.indicators_summary_failed_calls == []
    assert record.indicators_summary_calls is None


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
    def responder(_system: str, _user: str) -> Answer:
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
        pytest.param({"max_summary_chars": 1000}, id="output limit"),
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
