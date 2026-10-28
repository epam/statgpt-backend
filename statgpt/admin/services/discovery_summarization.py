"""Deriving the indicator list a discovery document is published with.

A record's `indicators_coverage` is what its submitter authored: every indicator, each with its
units of measure. Indexed as it is, a list of a few hundred indicators would drown the
description a question is meant to match, so a run derives `indicators_summary` from it:

- the list is parsed, and a units of measure every indicator shares is stated once, as a
  `Unit: ...` prefix, instead of after every name;
- a list that is still longer than the channel's threshold is collapsed by an LLM into
  indicator groups. A list too long for one call is split between indicators, each part is
  summarized on its own, and the parts' groups are merged by one more call.

The prefix is added here, never by the model, so the unit a document states is the one the
submitter wrote.
"""

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, cast

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, StringConstraints, ValidationError, create_model

from statgpt.admin.settings.discovery import DiscoverySummarizeSettings
from statgpt.common import models, schemas
from statgpt.common.default_prompts import discovery_indexing_default_prompts
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils import (
    async_utils,
    compact_units,
    format_exception_reason,
    get_ts_utcnow,
    split_cell,
)
from statgpt.common.utils.misc import normalize_whitespace
from statgpt.common.utils.models import get_chat_model

_log = logging.getLogger(__name__)

_SETTINGS = DiscoverySummarizeSettings()

_SEPARATOR = "; "
_UNIT_PREFIX = "Unit: "

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
        groups=(
            list[group],  # type: ignore[valid-type]
            Field(
                min_length=1,
                max_length=config.max_groups,
                description="The indicator groups, each naming its subject and breakdowns.",
            ),
        ),
    )


def _effective_config(config: schemas.DiscoveryIndexingConfig) -> schemas.DiscoveryIndexingConfig:
    """The config with every unset prompt replaced by its default.

    What a summary depends on, so it is also what the source hash covers: a changed default
    prompt re-derives the summaries made with the old one.
    """
    prompts = schemas.DiscoveryIndexingPrompts(
        chunk_summary_prompt=(
            config.prompts.chunk_summary_prompt
            or discovery_indexing_default_prompts.chunk_summary_prompt
        ),
        merge_summary_prompt=(
            config.prompts.merge_summary_prompt
            or discovery_indexing_default_prompts.merge_summary_prompt
        ),
    )
    return config.model_copy(update={"prompts": prompts})


def render_indicators(unit: str | None, items: Sequence[str]) -> str:
    """Join indicators or groups into the published form, the shared unit leading."""
    joined = _SEPARATOR.join(items)
    return f"{_UNIT_PREFIX}{unit}{_SEPARATOR}{joined}" if unit else joined


class _Result(StrEnum):
    SUMMARIZED = "summarized"
    COPIED = "copied"
    UNCHANGED = "unchanged"
    FAILED = "failed"


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

    One instance per run. Mutates the records it is given and leaves persisting them to the
    caller.
    """

    def __init__(
        self,
        config: schemas.DiscoveryIndexingConfig,
        *,
        force: bool = False,
        concurrency: int | None = None,
        timeout_seconds: float | None = None,
        validation_retries: int | None = None,
    ) -> None:
        """`force` re-derives every valid record's summary, even if its source hash matches.

        The concurrency, timeout and retries default to the configured settings.
        """
        self._config = _effective_config(config)
        self._fingerprint = self._config.model_dump_json()
        self._output_model = _output_model(self._config)
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

        A failure on one record is recorded on that record and the run continues.
        """
        valid = [
            record
            for record in records
            if record.validation_status is schemas.DiscoveryValidationStatus.VALID
        ]
        results: list[_Result] = await async_utils.gather_with_concurrency(
            self._concurrency, *(self._summarize_safely(record) for record in valid)
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

    async def _summarize_safely(self, record: models.DiscoveryDataset) -> _Result:
        """One record's turn, which must not raise.

        The turns share a task group, so an exception escaping here would cancel every other
        record's turn.

        A failure leaves the summary and its hash as they were, so the next run tries again,
        and marks the record FAILED, which keeps its live document from being touched.
        """
        previous = (record.indicators_summary, record.missing_indicators)
        record.missing_indicators = ""
        try:
            result = await self._summarize(record)
        except Exception as e:
            _log.exception(
                f"Failed to summarize the indicators of discovery record {record.id}"
                f" (agency={record.agency!r} dataset_id={record.dataset_id!r})"
            )
            record.indexing_status = schemas.DiscoveryIndexingStatus.FAILED
            record.index_error = (
                f"Summarizing the indicator list failed: {format_exception_reason(e)}"
            )
            return _Result.FAILED

        changed = (record.indicators_summary, record.missing_indicators) != previous
        if changed and record.indexing_status is schemas.DiscoveryIndexingStatus.INDEXED:
            record.indexing_status = schemas.DiscoveryIndexingStatus.OUTDATED
        return result

    async def _summarize(self, record: models.DiscoveryDataset) -> _Result:
        source_hash = self._source_hash(record.indicators_coverage)
        if not self._force and record.indicators_summary_source_hash == source_hash:
            return _Result.UNCHANGED

        unit, names = compact_units(split_cell(record.indicators_coverage))
        compacted = render_indicators(unit, names)
        if len(compacted) <= self._config.summary_threshold_chars:
            record.indicators_summary = compacted
            record.indicators_summarized_at = None
            result = _Result.COPIED
        else:
            groups = await self._summarize_names(unit, names)
            record.indicators_summary = render_indicators(unit, groups)
            record.indicators_summarized_at = get_ts_utcnow()
            result = _Result.SUMMARIZED

        record.indicators_summary_source_hash = source_hash
        return result

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
            groups.extend(
                await self._ask(
                    self._config.prompts.chunk_summary_prompt, unit, indicators="\n".join(chunk)
                )
            )
        if len(chunks) > 1:
            groups = await self._ask(
                self._config.prompts.merge_summary_prompt, unit, groups="\n".join(groups)
            )
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
        self, prompt: schemas.SystemUserPrompt | None, unit: str | None, **inputs: str
    ) -> list[str]:
        assert prompt is not None, "every prompt falls back to its default"
        chain = prompt.get_template() | self._structured_llm()
        response = cast(
            _IndicatorGroups,
            await chain.ainvoke(
                {
                    "unit": unit or "",
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
                    api_key=dial_settings.api_key.get_secret_value(),
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
