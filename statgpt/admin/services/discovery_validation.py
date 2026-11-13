"""Semantic validation of discovery dataset records.

Run by an indexing job, never on a write path. These checks are expected to become slow and
expensive - LLM-assisted normalization, jurisdiction and remit checks - and a request that
uploads thousands of rows cannot absorb that. The checks below are cheap today; the seam
exists because the ones that follow will not be.

Deferring them also means the check set can change - new checks, corrected vocabularies -
without stored verdicts going stale relative to the checks that produced them: the next run
re-derives all of them.

The criteria encode the metadata guidelines, which are a separate deliverable still being
developed, so the check set is a registry rather than a hardcoded sequence.
"""

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from statgpt.common.schemas import DiscoveryIndicator, DiscoveryValidationIssue, read_indicators
from statgpt.common.utils import FREQUENCY_VOCABULARY, is_known_frequency, split_cell

_log = logging.getLogger(__name__)

_ALLOWED_URL_SCHEMES = ("http", "https")

_MAX_NAMED_INDICATORS = 5
"""How many indicators an issue names before it only counts the rest."""

_MAX_INDICATOR_NAME_CHARS = 60
"""How much of an indicator's name an issue quotes, so a long one does not swamp the message."""


class DiscoveryRecord(Protocol):
    """The submitted half of a discovery dataset record.

    What a submitter wrote, and so the only thing a check may read: the columns the indexing
    run derives are filled in after validation has already had its say.

    A narrow read-only view, so a check can be run against a stored row, an unsaved
    schema, or a test stub alike. The members are properties rather than attributes so the
    protocol stays covariant: a mutable attribute would be invariant, and a mapped column
    typed `Mapped[str]` would not satisfy it.
    """

    @property
    def reference_area(self) -> str: ...

    @property
    def regional_coverage(self) -> str: ...

    @property
    def agency(self) -> str: ...

    @property
    def dataset_id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def url(self) -> str: ...

    @property
    def time_coverage(self) -> str: ...

    @property
    def frequency_coverage(self) -> str: ...

    @property
    def indicators_coverage(self) -> Sequence[DiscoveryIndicator | Mapping[str, Any]]:
        """Models on a schema, the JSON they were stored as on a stored row."""
        ...


_CheckFn = Callable[[DiscoveryRecord], Iterable[DiscoveryValidationIssue]]


@dataclass(frozen=True)
class DiscoveryCheck:
    name: str
    run: _CheckFn


def _check_frequency_coverage(record: DiscoveryRecord) -> Iterable[DiscoveryValidationIssue]:
    """Column I must name at least one frequency, each from the template's vocabulary."""
    values = split_cell(record.frequency_coverage)
    if not values:
        yield DiscoveryValidationIssue(
            field="frequency_coverage",
            message=(
                "At least one frequency is required, from: "
                f"{_allowed_frequencies()}. Search narrows by it."
            ),
        )
        return

    for value in values:
        if not is_known_frequency(value):
            yield DiscoveryValidationIssue(
                field="frequency_coverage",
                message=f"{value!r} is not one of: {_allowed_frequencies()}.",
            )


def _allowed_frequencies() -> str:
    return "; ".join(FREQUENCY_VOCABULARY)


def _check_reference_area(record: DiscoveryRecord) -> Iterable[DiscoveryValidationIssue]:
    """Column A must name at least one reference area."""
    if not split_cell(record.reference_area):
        yield DiscoveryValidationIssue(
            field="reference_area",
            message=(
                "At least one reference area is required: search narrows by it."
                " A group such as 'Euro area' or 'World' counts as one."
            ),
        )


def _check_url(record: DiscoveryRecord) -> Iterable[DiscoveryValidationIssue]:
    """Column G must be a web address, since referring a user to it is the record's purpose.

    ``http`` is accepted alongside ``https``: an invalid record is not published, so
    demanding https would delist an agency that publishes over plain http - a warning
    nobody could act on. An empty cell is not an issue.
    """
    if not record.url:
        return

    parts = urlsplit(record.url)
    if parts.scheme.casefold() not in _ALLOWED_URL_SCHEMES or not parts.netloc:
        yield DiscoveryValidationIssue(
            field="url",
            message=f"{record.url!r} is not an http or https web address.",
        )


def _check_indicators_coverage(record: DiscoveryRecord) -> Iterable[DiscoveryValidationIssue]:
    """The dataset's indicator list must name at least one indicator.

    The only place emptiness is caught. No write path refuses it, so this one check covers a
    dataset whose indicators sheet is missing or holds only its header, a CSV without the
    column, a JSON write that omits it, and a row stored empty under the old template.
    """
    if not record.indicators_coverage:
        yield DiscoveryValidationIssue(
            field="indicators_coverage",
            message=(
                "At least one indicator is required: search narrows by it. List the"
                " dataset's indicators on its own sheet, named with the dataset's ID."
            ),
        )


def _shorten(name: str) -> str:
    if len(name) <= _MAX_INDICATOR_NAME_CHARS:
        return name
    return name[: _MAX_INDICATOR_NAME_CHARS - 1].rstrip() + "…"


def _check_indicator_units(record: DiscoveryRecord) -> Iterable[DiscoveryValidationIssue]:
    """Every indicator must give its units of measure.

    The units are part of what a dataset is matched on - the same indicator in percent and in
    currency are different series - and the published list states them, so an indicator
    without them is one a referral cannot describe. Caught here rather than on upload, like
    every other gap in a record, so the rest of the file is still stored.
    """
    missing = [item.name for item in read_indicators(record.indicators_coverage) if not item.units]
    if not missing:
        return

    named = ", ".join(repr(_shorten(name)) for name in missing[:_MAX_NAMED_INDICATORS])
    rest = len(missing) - _MAX_NAMED_INDICATORS
    yield DiscoveryValidationIssue(
        field="indicators_coverage",
        message=(
            f"Units of measure are required for every indicator. Missing for: {named}"
            f"{f' and {rest} more' if rest > 0 else ''}."
        ),
    )


def _check_description(record: DiscoveryRecord) -> Iterable[DiscoveryValidationIssue]:
    """Column F must say something, because it is the whole of the published document."""
    if not record.description:
        yield DiscoveryValidationIssue(
            field="description",
            message="A description is required: it is the content of the published document.",
        )


DEFAULT_CHECKS: tuple[DiscoveryCheck, ...] = (
    DiscoveryCheck(name="description", run=_check_description),
    DiscoveryCheck(name="reference_area", run=_check_reference_area),
    DiscoveryCheck(name="frequency_coverage", run=_check_frequency_coverage),
    DiscoveryCheck(name="indicators_coverage", run=_check_indicators_coverage),
    DiscoveryCheck(name="indicator_units", run=_check_indicator_units),
    DiscoveryCheck(name="url", run=_check_url),
)
"""The check set as it stands."""


class DiscoveryValidator:
    """Runs the check set over one record at a time."""

    def __init__(self, checks: Sequence[DiscoveryCheck] = DEFAULT_CHECKS) -> None:
        self._checks = tuple(checks)

    def validate(self, record: DiscoveryRecord) -> list[DiscoveryValidationIssue]:
        """Return every issue found. An empty list means the record may be published.

        A check that raises is reported as an issue rather than aborting the run: one
        broken check must not keep a whole channel from being indexed.
        """
        issues: list[DiscoveryValidationIssue] = []
        for check in self._checks:
            try:
                issues.extend(check.run(record))
            except Exception:
                _log.exception(
                    f"Discovery check {check.name!r} failed on record"
                    f" agency={record.agency!r} dataset_id={record.dataset_id!r}"
                )
                issues.append(
                    DiscoveryValidationIssue(
                        field=check.name,
                        message=(
                            f"The {check.name!r} check could not be evaluated for this record."
                        ),
                    )
                )
        return issues
