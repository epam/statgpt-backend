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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

from statgpt.common.schemas import DiscoveryValidationIssue

_log = logging.getLogger(__name__)

_FREQUENCY_VOCABULARY: tuple[str, ...] = (
    "Daily",
    "Business daily",
    "Weekly",
    "Monthly",
    "Quarterly",
    "Semi-annual",
    "Annual",
    "Irregular",
)
"""The frequencies the discovery template tells submitters to choose from (column J)."""

_FREQUENCY_LOOKUP = {value.casefold(): value for value in _FREQUENCY_VOCABULARY}
_ALLOWED_URL_SCHEMES = ("http", "https")


class DiscoveryRecord(Protocol):
    """The descriptive half of a discovery dataset record.

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
    def excluded_regional_values(self) -> str: ...

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
    def indicators_coverage(self) -> str: ...

    @property
    def missing_indicators(self) -> str: ...


_CheckFn = Callable[[DiscoveryRecord], Iterable[DiscoveryValidationIssue]]


@dataclass(frozen=True)
class DiscoveryCheck:
    name: str
    run: _CheckFn


def _check_frequency_coverage(record: DiscoveryRecord) -> Iterable[DiscoveryValidationIssue]:
    """Every ';'-separated token of column J must come from the template's vocabulary.

    An empty cell is not an issue: absent information does not make a record unfit to
    refer to, a wrong value does.
    """
    if not record.frequency_coverage:
        return

    allowed = "; ".join(_FREQUENCY_VOCABULARY)
    for token in record.frequency_coverage.split(";"):
        value = token.strip()
        if value and value.casefold() not in _FREQUENCY_LOOKUP:
            yield DiscoveryValidationIssue(
                field="frequency_coverage",
                message=f"{value!r} is not one of: {allowed}.",
            )


def _check_url(record: DiscoveryRecord) -> Iterable[DiscoveryValidationIssue]:
    """Column H must be a web address, since referring a user to it is the record's purpose.

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


DEFAULT_CHECKS: tuple[DiscoveryCheck, ...] = (
    DiscoveryCheck(name="frequency_coverage", run=_check_frequency_coverage),
    DiscoveryCheck(name="url", run=_check_url),
)
"""The check set as it stands.

Deliberately minimal. There is no severity axis, so a check must only emit an issue when
the record genuinely should not be indexed - an advisory nitpick has nowhere to live and
would silently make records unindexable. `reference_area` is therefore not checked against
ISO codes: the template explicitly allows group labels such as 'Euro area' or 'World'.
"""


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
