"""The reference-area filter axis shared by the discovery write and read paths.

Column A of the discovery workbook is one free-text cell, which a document filter cannot match:
equality against `'Indonesia (IDN); Malaysia (MYS)'` never matches a question about Malaysia, so
every multi-country dataset would be unreachable by a country filter. The write path therefore
publishes the cell's entries as a list of discrete values alongside the verbatim cell, and the
read path grounds a user's countries against the values the channel reports holding.

Both halves live here because they have to agree on one vocabulary: a value the publisher writes
and a value the search grounds to are the same string, or the filter silently matches nothing.

Splitting a delimited cell is not deriving content. The workbook prescribes the format - "English
country name + ISO 3166-1 alpha-3 code, separated by ';'" - so this parses what a submitter was
told to write, and adds nothing of its own.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from statgpt.common.utils import normalize_whitespace

SENTINEL = "*ANY*"
"""Stands for "scope not pinned to countries", and is always included in a country filter.

A record whose cell holds a group label such as 'Euro area' or 'World', holds nothing parseable,
or is empty covers countries this module cannot enumerate. Excluding it from a country filter
would silently drop a euro-area dataset from a question about Germany, so it carries this value
and every country filter unions it in.

Spelled with characters no country name or ISO code contains, so it cannot collide with a real
value.
"""

_PARTNER_MARKER = "partner countries"
"""Introduces the counterparty half of a bilateral record's cell.

The workbook's convention is "reporting countries first, then 'partner countries:' + the list".
Only the reporting side is a filter value: a question about Japan's exports is a question for a
record that reports Japan, not for every record that happens to name Japan as a partner.
"""

_ISO3 = re.compile(r"\(\s*([A-Za-z]{3})\s*\)\s*$")
"""The trailing '(IDN)' of a country entry. An entry without one is not a country."""


def _fold(value: str) -> str:
    """Fold a value or an entity for comparison, the way the rest of discovery folds keys."""
    return normalize_whitespace(value).lower()


def parse_reference_area(cell: str) -> list[str]:
    """Split column A into the discrete values a document filter can match.

    Every entry before the partner marker becomes a value, verbatim apart from whitespace
    normalization, so the published vocabulary reads the way the workbook was filled in and a
    reader of `/channel/metadata` sees country names rather than codes.

    `SENTINEL` is appended when any entry carries no ISO 3166-1 alpha-3 code, and when the cell
    yields no entries at all. A group label is kept as a value *and* triggers the sentinel: the
    label is what a question about the euro area should filter to, and the sentinel is what keeps
    the record reachable from a question about one of its member countries.

    Order is stable and duplicates are dropped, so an unchanged cell always renders the same list
    and a record is not republished for a reordering that means nothing.
    """
    values: list[str] = []
    needs_sentinel = False

    for entry in cell.split(";"):
        entry = normalize_whitespace(entry)
        if not entry:
            continue
        if _fold(entry).startswith(_PARTNER_MARKER):
            # Everything from the marker on is the counterparty list, including this entry, whose
            # text is 'partner countries: China'.
            break
        values.append(entry)
        if not _ISO3.search(entry):
            needs_sentinel = True

    if not values:
        needs_sentinel = True

    if needs_sentinel:
        values.append(SENTINEL)

    return list(dict.fromkeys(values))


def value_aliases(value: str) -> set[str]:
    """The folded spellings a user's country entity may use for one published value.

    `'Indonesia (IDN)'` is matched by "Indonesia" and by "IDN" as well as by the whole string,
    which is why the publisher keeps the name and the code together: the value grounds itself,
    with no country-name table - and this repository has none.

    `SENTINEL` has no aliases. It is never something a user names; it is unioned in
    unconditionally by `ground_reference_areas`.
    """
    if value == SENTINEL:
        return set()

    folded = _fold(value)
    aliases = {folded}
    if match := _ISO3.search(value):
        aliases.add(_fold(match.group(1)))
        aliases.add(_fold(value[: match.start()]))
    return {alias for alias in aliases if alias}


@dataclass(frozen=True)
class GroundedAreas:
    """What grounding a request's countries against a channel's values produced."""

    values: list[str] = field(default_factory=list)
    """Published values to filter on, `SENTINEL` included when the channel holds it.

    Empty means no country filter should be applied at all: an unfiltered search is a precision
    problem the relevance judge absorbs, while an over-narrow filter loses records irrecoverably.
    """

    unmatched: list[str] = field(default_factory=list)
    """Entities that matched no published value, for logging.

    A country the channel has no records for lands here, and so does an alias this module cannot
    resolve - "Holland", "Tu:rkiye". The two are worth telling apart in a log, which is why they
    are reported rather than silently dropped.
    """

    @property
    def matched_any(self) -> bool:
        """Whether at least one entity resolved to a published value.

        The sentinel alone does not count: it is not evidence that the request's countries are in
        the channel, so a filter of nothing but the sentinel would narrow a search to the records
        with no country scope - the opposite of what the request asked for.
        """
        return any(value != SENTINEL for value in self.values)


def ground_reference_areas(entities: Iterable[str], available: Iterable[str]) -> GroundedAreas:
    """Resolve a request's country entities to values the channel actually holds.

    Grounding is mandatory rather than defensive. The RAG service types a filterable field as a
    `Literal` of the values present in the channel, so a value no document carries fails the whole
    retrieval request instead of matching nothing - and an unknown field name fails it the same
    way. Every value returned here comes from `available`.

    The sentinel is included whenever the channel holds it and at least one entity matched, so a
    country filter never excludes a record whose scope could not be pinned to countries. When
    nothing matched, the result is empty: `GroundedAreas.values` is then a filter not worth
    applying, and the caller searches unfiltered.
    """
    available = list(dict.fromkeys(available))
    by_alias: dict[str, list[str]] = {}
    for value in available:
        for alias in value_aliases(value):
            by_alias.setdefault(alias, []).append(value)

    values: list[str] = []
    unmatched: list[str] = []
    for entity in entities:
        if matches := by_alias.get(_fold(entity)):
            values.extend(matches)
        elif normalize_whitespace(entity):
            unmatched.append(normalize_whitespace(entity))

    values = list(dict.fromkeys(values))
    if values and SENTINEL in available:
        values.append(SENTINEL)

    return GroundedAreas(values=values, unmatched=list(dict.fromkeys(unmatched)))
