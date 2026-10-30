"""Deriving the indicator list a discovery document is published with.

A record's `indicators_coverage` is what its submitter authored: every indicator, each with its
units of measure. Indexed as it is, a list of a few hundred indicators would drown the
description a question is meant to match, so a run derives `indicators_summary` from it:

- a units of measure every indicator shares is stated once, as a
  `Unit: ...` prefix, instead of after every name;
- a list that is still longer than the channel's threshold is collapsed by an LLM into
  indicator groups. A list too long for one call is split between indicators, each part is
  summarized on its own, and the parts' groups are merged by one more call. Units the indicators
  do not share are summarized by the model: as one value for the whole list when they can be,
  published as the same prefix, or along with each group otherwise.

The prefix for a shared unit is added here, never by the model, so the shared unit a document
states is the one the submitter wrote.

An answer that does not fit the output limit is not simply asked for again: the conversation
goes on with the answer and what is wrong with it, so the model corrects it rather than
repeating it.
"""

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Self, TypedDict, cast

import httpx
from aidial_client.types.model import ModelPricing
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    create_model,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from statgpt.admin.settings.discovery import DiscoverySummarizeSettings
from statgpt.common import models, schemas
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.default_prompts import discovery_indexing_default_prompts
from statgpt.common.schemas.token_usage import TokenUsageItem
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils import (
    AsyncLoadingCache,
    async_utils,
    dial_client_factory,
    format_exception_reason,
    get_ts_utcnow,
    normalize_whitespace,
)
from statgpt.common.utils.dial.model_pricing import ModelPricingGetter
from statgpt.common.utils.models import get_chat_model
from statgpt.common.utils.token_usage_utils import TokenUsageCostCalculator

_log = logging.getLogger(__name__)

_SETTINGS = DiscoverySummarizeSettings()

_SEPARATOR = "; "
_UNIT_PREFIX = "Unit: "

_DERIVATION_VERSION = 2
"""Part of the source hash. Bump it when what this module makes of a list changes - the unit
parsing, the published form, the output model - so existing summaries are derived again."""

_HTTP_RETRIES = 3
"""Retries of a failed request. Fewer than the default: a run waits on every one of them."""

_CONNECT_TIMEOUT_SECONDS = 4


def _render_indicators(unit: str | None, items: Sequence[str]) -> str:
    """Join indicators or groups into the published form, the shared unit leading."""
    joined = _SEPARATOR.join(items)
    return f"{_UNIT_PREFIX}{unit}{_SEPARATOR}{joined}" if unit else joined


def _clean(text: str) -> str:
    # ';' separates the items of the published list, so an item must not contain one.
    return normalize_whitespace(text.replace(";", ","))


def _compact_units(
    indicators: Sequence[schemas.DiscoveryIndicator],
) -> tuple[str | None, list[str]]:
    """Factor a units of measure shared by every indicator out of the list.

    Returns `(unit, names)` when every indicator has units and they are all equal -
    case-sensitive, as submitted. Otherwise `(None, entries)`, each entry with its own units in
    parentheses, the way the flat form writes them. An indicator the list repeats, ignoring
    case, is listed once.
    """
    distinct: dict[str, schemas.DiscoveryIndicator] = {}
    for indicator in indicators:
        distinct.setdefault(indicator.flat.casefold(), indicator)
    units = {indicator.units for indicator in distinct.values()}
    if len(units) == 1 and (unit := units.pop()):
        return unit, [_clean(indicator.name) for indicator in distinct.values()]
    return None, [indicator.flat for indicator in distinct.values()]


class _AnswerContext(TypedDict):
    unit: str | None
    """The unit every indicator shares, published as the prefix, or `None`."""
    max_chars: int


class _IndicatorGroups(BaseModel):
    """What one summarization call answers with. Narrowed per run by `_output_model`.

    The fields are generated in their order, so the model reasons before it commits to units
    and groups. Validated with an `_AnswerContext`: the limit holds for the published form.
    """

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        description="Plan the groups here before writing them: subjects, breakdowns, units."
    )
    units: str | None = Field(
        description=(
            "The units of measure of every group as one value, or null when the groups keep"
            " their own units."
        )
    )
    groups: list[str]

    @field_validator("units", mode="after")
    @classmethod
    def _clean_units(cls, units: str | None) -> str | None:
        return (_clean(units) or None) if units is not None else None

    @field_validator("groups", mode="after")
    @classmethod
    def _clean_groups(cls, groups: list[str]) -> list[str]:
        return [_clean(group) for group in groups]

    @model_validator(mode="after")
    def _fits_the_limit(self, info: ValidationInfo) -> Self:
        context = cast(_AnswerContext, info.context)
        limit = context["max_chars"]
        length = len(_render_indicators(self.units or context["unit"], self.groups))
        if length > limit:
            raise ValueError(
                f"Published, the summary is {length} characters long, {length - limit} over"
                f" the limit of {limit}. Merge or shorten groups, naming breakdowns instead of"
                " listing their members."
            )
        return self


def _output_model(
    config: schemas.DiscoveryIndexingConfig, *, with_units: bool
) -> type[_IndicatorGroups]:
    """The answer's model, bounded by the channel's output limit.

    Built per run because the limit is configuration. `with_units` keeps the `units` field,
    for indicators that share no unit; otherwise it is left out of the schema.

    A group may take the whole limit, which providers do not enforce today; the published
    form is checked against it by the model's validator.
    """
    group = Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=config.max_summary_chars),
    ]
    fields: dict[str, Any] = {
        "groups": (
            list[group],  # type: ignore[valid-type]
            Field(
                min_length=1,
                description="The indicator groups, each naming its subject and breakdowns.",
            ),
        )
    }
    if not with_units:
        fields["units"] = (SkipJsonSchema[None], None)  # type: ignore[misc]
    return create_model(
        "IndicatorGroups",
        __base__=_IndicatorGroups,
        __doc__="Indicator groups that together cover every indicator of the dataset.",
        **fields,
    )


def _response_format(output_model: type[BaseModel]) -> dict[str, Any]:
    """The strict structured-output format of an answer's model.

    A plain schema rather than the model, so the answer comes back raw and is validated here:
    given the model, the SDK would raise on an invalid answer and lose the text the model is
    then asked to correct.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": output_model.__name__,
            "schema": output_model.model_json_schema(),
            "strict": True,
        },
    }


_ANSWER_JSON: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


def _stored_answer(text: str) -> dict[str, JsonValue] | None:
    """An answer as it is stored on its record: parsed, or `None` when it is not JSON."""
    try:
        return _ANSWER_JSON.validate_json(text)
    except ValidationError:
        return None


def _dump_run(calls: Sequence[schemas.DiscoverySummaryCall]) -> dict[str, Any]:
    return schemas.DiscoverySummaryRun(calls=list(calls)).model_dump(mode="json")


def _describe(error: ValidationError) -> str:
    """What is wrong with an answer, one line per error, for the model to correct."""
    return "\n".join(
        f"- {'.'.join(map(str, e['loc'])) or 'answer'}: {e['msg']}"
        for e in error.errors(include_url=False)
    )


def _units_instruction(unit: str | None) -> str:
    """The prompts' units line: leave a shared unit to the prefix, or summarize differing ones."""
    if unit:
        return discovery_indexing_default_prompts.shared_unit_instruction.format(unit=unit)
    return discovery_indexing_default_prompts.per_indicator_units_instruction


class _Result(StrEnum):
    SUMMARIZED = "summarized"
    COPIED = "copied"
    UNCHANGED = "unchanged"
    FAILED = "failed"


@dataclass(frozen=True)
class _Summary:
    """A derived summary, not yet written to its record."""

    text: str
    by_llm: bool


@dataclass(frozen=True)
class _CallLog:
    """Where one summarization call records its answers, in a record's list of calls."""

    calls: list[schemas.DiscoverySummaryCall]
    step: schemas.DiscoverySummaryStep
    part: int | None

    def add(self, text: str, error: str | None, usage: schemas.DiscoveryLLMCallUsage) -> None:
        self.calls.append(
            schemas.DiscoverySummaryCall(
                step=self.step,
                part=self.part,
                answer=_stored_answer(text),
                error=error,
                usage=usage,
            )
        )


@dataclass
class SummarizeOutcome:
    """What a summarization stage did. Reported on the job row."""

    summarized: int = 0
    """Records whose summary an LLM produced."""

    copied: int = 0
    """Records short enough to be published as the parsed list."""

    unchanged: int = 0
    """Records whose summary was derived from the same list and config, so left alone."""

    failed_ids: set[int] = field(default_factory=set)
    """Records whose summarization failed. Their documents must be left as they are."""


class DiscoveryIndicatorSummarizer:
    """Derives `indicators_summary` for a run's valid records, in place.

    One instance per run. Mutates the records it is given and calls `persist` after each one,
    so a run that dies midway - the job timing out on a channel of long lists, say - keeps
    every summary it had already paid for, and the next run skips them by their hash.
    """

    def __init__(
        self,
        config: schemas.DiscoveryIndexingConfig,
        auth_context: AuthContext,
        persist: Callable[[], Awaitable[None]],
        *,
        force: bool = False,
        concurrency: int | None = None,
        timeout_seconds: float | None = None,
        validation_retries: int | None = None,
    ) -> None:
        """`force` re-derives every valid record's summary, even if its source hash matches.

        `persist` writes the records' pending changes; it is never awaited concurrently.

        The concurrency, timeout and retries default to the configured settings.
        """
        self._chunk_prompt = (
            config.prompts.chunk_summary_prompt
            or discovery_indexing_default_prompts.chunk_summary_prompt
        )
        self._merge_prompt = (
            config.prompts.merge_summary_prompt
            or discovery_indexing_default_prompts.merge_summary_prompt
        )
        # The config with the defaults filled in is what a summary depends on, so it is what
        # the source hash covers: a changed default prompt re-derives the summaries made with
        # the old one. So does any field added to the config, or to its model config.
        self._config = config.model_copy(
            update={
                "prompts": schemas.DiscoveryIndexingPrompts(
                    chunk_summary_prompt=self._chunk_prompt,
                    merge_summary_prompt=self._merge_prompt,
                )
            }
        )
        self._fingerprint = (
            f"{_DERIVATION_VERSION}\0{self._config.model_dump_json()}"
            f"\0{discovery_indexing_default_prompts.shared_unit_instruction}"
            f"\0{discovery_indexing_default_prompts.per_indicator_units_instruction}"
            f"\0{discovery_indexing_default_prompts.invalid_answer_feedback}"
        )
        self._shared_unit_model = _output_model(self._config, with_units=False)
        self._own_units_model = _output_model(self._config, with_units=True)
        self._auth_context = auth_context
        self._persist = persist
        # Every write to a record happens under it, with the commit: a commit flushes every
        # pending change, and one made while another turn's flush is in flight could be lost.
        self._write_lock = asyncio.Lock()
        self._force = force
        self._concurrency = concurrency if concurrency is not None else _SETTINGS.concurrency
        self._timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else _SETTINGS.timeout_seconds
        )
        self._validation_retries = (
            validation_retries if validation_retries is not None else _SETTINGS.validation_retries
        )
        # Built on first use, so a run with nothing to summarize creates no client.
        self._llm: BaseChatModel | None = None
        # Loaded on first use too, once per run however many records call at once.
        self._pricing: AsyncLoadingCache[ModelPricing | None] = AsyncLoadingCache()

    async def summarize(self, records: Sequence[models.DiscoveryDataset]) -> SummarizeOutcome:
        """Derive the summary of every valid record. Invalid records are not touched.

        A failure on one record is recorded on that record and the run continues. A failure to
        persist is the run's own and propagates.
        """
        valid = [
            record
            for record in records
            if record.validation_status is schemas.DiscoveryValidationStatus.VALID
        ]
        results: list[_Result] = await async_utils.gather_with_concurrency(
            self._concurrency, *(self._summarize_record(record) for record in valid)
        )

        outcome = SummarizeOutcome()
        for record, result in zip(valid, results):
            match result:
                case _Result.SUMMARIZED:
                    outcome.summarized += 1
                case _Result.COPIED:
                    outcome.copied += 1
                case _Result.UNCHANGED:
                    outcome.unchanged += 1
                case _Result.FAILED:
                    outcome.failed_ids.add(record.id)
        return outcome

    async def _summarize_record(self, record: models.DiscoveryDataset) -> _Result:
        """One record's turn. Only `persist` may raise out of it.

        The turns share a task group, so a summarization failure escaping here would cancel
        every other record's turn.

        A failure leaves the summary, its hash and its calls as they were, so the next run
        tries again, and marks the record FAILED, which keeps its live document from being
        touched. The answers it collected are kept apart, as the failed calls.
        """
        indicators = schemas.read_indicators(record.indicators_coverage)
        source_hash = self._source_hash(indicators)
        if not self._force and record.indicators_summary_source_hash == source_hash:
            return _Result.UNCHANGED

        calls: list[schemas.DiscoverySummaryCall] = []
        try:
            summary = await self._derive(indicators, calls)
        except Exception as e:
            _log.exception(
                f"Failed to summarize the indicators of discovery record {record.id}"
                f" (agency={record.agency!r} dataset_id={record.dataset_id!r})"
            )
            async with self._write_lock:
                record.indexing_status = schemas.DiscoveryIndexingStatus.FAILED
                record.index_error = (
                    f"Summarizing the indicator list failed: {format_exception_reason(e)}"
                )
                record.indicators_summary_failed_calls = _dump_run(calls)
                await self._persist()
            return _Result.FAILED

        async with self._write_lock:
            self._apply(record, summary, calls, source_hash)
            await self._persist()
        return _Result.SUMMARIZED if summary.by_llm else _Result.COPIED

    @staticmethod
    def _apply(
        record: models.DiscoveryDataset,
        summary: _Summary,
        calls: Sequence[schemas.DiscoverySummaryCall],
        source_hash: str,
    ) -> None:
        """Write a derived summary to its record, outdating an indexed record it changes.

        `missing_indicators` is cleared on the way: it is no longer derived, and what the old
        template left in it would otherwise stay published.
        """
        changed = (record.indicators_summary, record.missing_indicators) != (summary.text, "")
        record.indicators_summary = summary.text
        record.missing_indicators = ""
        record.indicators_summarized_at = get_ts_utcnow() if summary.by_llm else None
        record.indicators_summary_calls = _dump_run(calls) if summary.by_llm else None
        record.indicators_summary_failed_calls = None
        record.indicators_summary_source_hash = source_hash
        if changed and record.indexing_status is schemas.DiscoveryIndexingStatus.INDEXED:
            record.indexing_status = schemas.DiscoveryIndexingStatus.OUTDATED

    async def _derive(
        self,
        indicators: Sequence[schemas.DiscoveryIndicator],
        calls: list[schemas.DiscoverySummaryCall],
    ) -> _Summary:
        """Derive a list's summary, appending every LLM answer to `calls` as it arrives."""
        unit, names = _compact_units(indicators)
        compacted = _render_indicators(unit, names)
        if len(compacted) <= self._config.summary_threshold_chars:
            return _Summary(text=compacted, by_llm=False)
        answer = await self._summarize_names(unit, names, calls)
        return _Summary(text=_render_indicators(unit or answer.units, answer.groups), by_llm=True)

    def _source_hash(self, indicators: Sequence[schemas.DiscoveryIndicator]) -> str:
        """What the summary was derived from: the submitted list and the effective config.

        The list is hashed in its flat form, the text it was stored as before it was stored as
        JSON, so a summary derived then is not derived - and paid for - again.
        """
        flat = schemas.render_flat_indicators(indicators)
        return hashlib.sha256(f"{self._fingerprint}\0{flat}".encode("utf-8")).hexdigest()

    async def _summarize_names(
        self, unit: str | None, names: list[str], calls: list[schemas.DiscoverySummaryCall]
    ) -> _IndicatorGroups:
        """Collapse the names into groups: one call per chunk, then one merging them."""
        parts = [
            await self._ask(
                self._chunk_prompt,
                unit,
                _CallLog(calls, schemas.DiscoverySummaryStep.CHUNK, part=number),
                indicators="\n".join(chunk),
            )
            for number, chunk in enumerate(self._chunk(unit, names), start=1)
        ]
        if len(parts) == 1:
            return parts[0]
        return await self._ask(
            self._merge_prompt,
            unit,
            _CallLog(calls, schemas.DiscoverySummaryStep.MERGE, part=None),
            groups=self._render_parts(parts),
        )

    @staticmethod
    def _render_parts(parts: Sequence[_IndicatorGroups]) -> str:
        """The merge call's input: each part's groups under its heading, with its units."""
        blocks = []
        for number, part in enumerate(parts, start=1):
            lines = [f"## Part {number}"]
            if part.units:
                lines.append(f"Units of every group: {part.units}")
            lines.extend(part.groups)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _chunk(self, unit: str | None, names: list[str]) -> list[list[str]]:
        """Split the names into chunks of at most `chunk_size_chars`, the unit line included.

        An indicator is never split: one longer than a chunk on its own makes a chunk of one.
        """
        base = len(unit or "")
        chunks: list[list[str]] = [[]]
        size = base
        for name in names:
            cost = len(name) + 1  # one per line
            if chunks[-1] and size + cost > self._config.chunk_size_chars:
                chunks.append([])
                size = base
            chunks[-1].append(name)
            size += cost
        return chunks

    async def _ask(
        self,
        prompt: schemas.SystemUserPrompt,
        unit: str | None,
        log: _CallLog,
        **inputs: str,
    ) -> _IndicatorGroups:
        """One summarization call, continued with feedback until its answer is valid.

        An invalid answer - unparseable, empty, past the limit - is appended to the
        conversation with what is wrong with it, up to `validation_retries` times; the last
        error then propagates. Every answer, valid or not, goes to `log` with its usage.
        """
        output_model = self._own_units_model if unit is None else self._shared_unit_model
        llm = self._chat_model().bind(response_format=_response_format(output_model))
        messages: list[BaseMessage] = prompt.get_template().format_messages(
            unit=unit or "",
            units_instruction=_units_instruction(unit),
            max_summary_chars=self._config.max_summary_chars,
            **inputs,
        )
        context = _AnswerContext(unit=unit, max_chars=self._config.max_summary_chars)
        corrections = 0
        while True:
            started = time.monotonic()
            answer = cast(AIMessage, await llm.ainvoke(messages))
            usage = await self._usage(answer, time.monotonic() - started)
            try:
                parsed = output_model.model_validate_json(answer.text, context=context)
            except ValidationError as e:
                error = _describe(e)
                log.add(answer.text, error, usage)
                if corrections == self._validation_retries:
                    raise
                corrections += 1
                _log.warning(f"Invalid summarization answer, asking for a correction: {e}")
                feedback = discovery_indexing_default_prompts.invalid_answer_feedback
                messages.extend([answer, HumanMessage(feedback.format(error=error))])
                continue
            log.add(answer.text, None, usage)
            return parsed

    async def _usage(
        self, answer: AIMessage, duration_seconds: float
    ) -> schemas.DiscoveryLLMCallUsage:
        """What a call used, priced with the deployment's DIAL pricing."""
        deployment = self._config.llm_model_config.deployment.deployment_id
        if (usage := answer.usage_metadata) is None:
            _log.warning(f"The answer of {deployment!r} reports no token usage")
            prompt_tokens = completion_tokens = 0
        else:
            prompt_tokens, completion_tokens = usage["input_tokens"], usage["output_tokens"]

        pricing = await self._pricing.get(deployment, lambda: self._load_pricing(deployment))
        [priced] = TokenUsageCostCalculator(
            {deployment: pricing} if pricing is not None else {}
        ).get_token_usage_with_costs(
            [
                TokenUsageItem(
                    deployment=deployment,
                    model=deployment,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            ]
        )
        return schemas.DiscoveryLLMCallUsage(
            deployment=deployment,
            model=answer.response_metadata.get("model_name") or deployment,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=priced.costs,
            duration_seconds=duration_seconds,
        )

    async def _load_pricing(self, deployment: str) -> ModelPricing | None:
        """The deployment's pricing, or `None` when DIAL has none or cannot be asked."""
        async with dial_client_factory(
            dial_settings.url, api_key=self._auth_context.api_key
        ) as dial:
            return await ModelPricingGetter(dial).get_model_pricing(deployment)

    def _chat_model(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_chat_model(
                api_key=self._auth_context.api_key,
                model_config=self._config.llm_model_config,
                timeout=httpx.Timeout(self._timeout_seconds, connect=_CONNECT_TIMEOUT_SECONDS),
                max_retries=_HTTP_RETRIES,
            )
        return self._llm
