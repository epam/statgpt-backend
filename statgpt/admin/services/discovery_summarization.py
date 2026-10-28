"""Deriving the indicator list a discovery document is published with.

A record's `indicators_coverage` is what its submitter authored: every indicator, each with its
units of measure. Indexed as it is, a list of a few hundred indicators would drown the
description a question is meant to match, so a run derives `indicators_summary` from it:

- the list is parsed, and a units of measure every indicator shares is stated once, as a
  `Unit: ...` prefix, instead of after every name;
- a list that is still longer than the channel's threshold is collapsed by an LLM into
  indicator groups. A list too long for one call is split between indicators, each part is
  summarized on its own, and the parts' groups are merged by one more call. Units the indicators
  do not share are summarized by the model along with the groups.

The prefix is added here, never by the model, so the shared unit a document states is the one
the submitter wrote.
"""

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, cast

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, StringConstraints, ValidationError, create_model

from statgpt.admin.settings.discovery import DiscoverySummarizeSettings
from statgpt.common import models, schemas
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.default_prompts import discovery_indexing_default_prompts
from statgpt.common.utils import (
    async_utils,
    compact_units,
    format_exception_reason,
    get_ts_utcnow,
    normalize_whitespace,
    split_cell,
)
from statgpt.common.utils.models import get_chat_model

_log = logging.getLogger(__name__)

_SETTINGS = DiscoverySummarizeSettings()

_SEPARATOR = "; "
_UNIT_PREFIX = "Unit: "

_DERIVATION_VERSION = 1
"""Part of the source hash. Bump it when what this module makes of a list changes - the unit
parsing, the published form, the output model - so existing summaries are derived again."""

_HTTP_RETRIES = 3
"""Retries of a failed request. Fewer than the default: a run waits on every one of them."""

_CONNECT_TIMEOUT_SECONDS = 4


class _IndicatorGroups(BaseModel):
    """What one summarization call answers with. Narrowed per run by `_output_model`."""

    groups: list[str]


def _output_model(config: schemas.DiscoveryIndexingConfig) -> type[_IndicatorGroups]:
    """The answer's model, bounded by the channel's output limits.

    Built per run because the limits are configuration. They are part of the JSON schema the
    model is held to, and enforced again on the answer, so an answer past them is retried.
    """
    group = Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=config.max_group_chars),
    ]
    return create_model(
        "IndicatorGroups",
        __base__=_IndicatorGroups,
        __doc__="Indicator groups that together cover every indicator of the dataset.",
        groups=(
            list[group],  # type: ignore[valid-type]
            Field(
                min_length=1,
                max_length=config.max_groups,
                description="The indicator groups, each naming its subject and breakdowns.",
            ),
        ),
    )


def _render_indicators(unit: str | None, items: Sequence[str]) -> str:
    """Join indicators or groups into the published form, the shared unit leading."""
    joined = _SEPARATOR.join(items)
    return f"{_UNIT_PREFIX}{unit}{_SEPARATOR}{joined}" if unit else joined


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
        )
        self._output_model = _output_model(self._config)
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
        self._llm: Runnable | None = None

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

        A failure leaves the summary and its hash as they were, so the next run tries again,
        and marks the record FAILED, which keeps its live document from being touched.
        """
        source_hash = self._source_hash(record.indicators_coverage)
        if not self._force and record.indicators_summary_source_hash == source_hash:
            return _Result.UNCHANGED

        try:
            summary = await self._derive(record.indicators_coverage)
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
                await self._persist()
            return _Result.FAILED

        async with self._write_lock:
            self._apply(record, summary, source_hash)
            await self._persist()
        return _Result.SUMMARIZED if summary.by_llm else _Result.COPIED

    @staticmethod
    def _apply(record: models.DiscoveryDataset, summary: _Summary, source_hash: str) -> None:
        """Write a derived summary to its record, outdating an indexed record it changes.

        `missing_indicators` is cleared on the way: it is no longer derived, and what the old
        template left in it would otherwise stay published.
        """
        changed = (record.indicators_summary, record.missing_indicators) != (summary.text, "")
        record.indicators_summary = summary.text
        record.missing_indicators = ""
        record.indicators_summarized_at = get_ts_utcnow() if summary.by_llm else None
        record.indicators_summary_source_hash = source_hash
        if changed and record.indexing_status is schemas.DiscoveryIndexingStatus.INDEXED:
            record.indexing_status = schemas.DiscoveryIndexingStatus.OUTDATED

    async def _derive(self, indicators_coverage: str) -> _Summary:
        unit, names = compact_units(split_cell(indicators_coverage))
        compacted = _render_indicators(unit, names)
        if len(compacted) <= self._config.summary_threshold_chars:
            return _Summary(text=compacted, by_llm=False)
        groups = await self._summarize_names(unit, names)
        return _Summary(text=_render_indicators(unit, groups), by_llm=True)

    def _source_hash(self, indicators_coverage: str) -> str:
        """What the summary was derived from: the submitted list and the effective config."""
        return hashlib.sha256(
            f"{self._fingerprint}\0{indicators_coverage}".encode("utf-8")
        ).hexdigest()

    async def _summarize_names(self, unit: str | None, names: list[str]) -> list[str]:
        """Collapse the names into groups: one call per chunk, then one merging them."""
        chunks = self._chunk(unit, names)
        groups: list[str] = []
        for chunk in chunks:
            groups.extend(await self._ask(self._chunk_prompt, unit, indicators="\n".join(chunk)))
        if len(chunks) > 1:
            groups = await self._ask(self._merge_prompt, unit, groups="\n".join(groups))
        return groups

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
        self, prompt: schemas.SystemUserPrompt, unit: str | None, **inputs: str
    ) -> list[str]:
        chain = prompt.get_template() | self._structured_llm()
        response = cast(
            _IndicatorGroups,
            await chain.ainvoke(
                {
                    "unit": unit or "",
                    "units_instruction": _units_instruction(unit),
                    "max_groups": self._config.max_groups,
                    "max_group_chars": self._config.max_group_chars,
                    **inputs,
                }
            ),
        )
        # ';' separates the groups of the published list, so a group must not contain one.
        return [normalize_whitespace(group.replace(";", ",")) for group in response.groups]

    def _structured_llm(self) -> Runnable:
        if self._llm is None:
            self._llm = (
                get_chat_model(
                    api_key=self._auth_context.api_key,
                    model_config=self._config.llm_model_config,
                    timeout=httpx.Timeout(self._timeout_seconds, connect=_CONNECT_TIMEOUT_SECONDS),
                    max_retries=_HTTP_RETRIES,
                )
                .with_structured_output(self._output_model, method="json_schema", strict=True)
                .with_retry(
                    retry_if_exception_type=(OutputParserException, ValidationError),
                    # An answer past the limits is not a rate limit: there is nothing to wait out.
                    wait_exponential_jitter=False,
                    stop_after_attempt=self._validation_retries + 1,
                )
            )
        return self._llm
