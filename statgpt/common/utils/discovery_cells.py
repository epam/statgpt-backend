"""Parsing the ';'-separated cells of a discovery dataset record.

Several workbook columns hold a list inside one cell. Published verbatim they are searchable
text and nothing more: a pre-filter needs the individual values, so the publisher derives an
array beside each such cell and the derived array is what a retrieval request filters on.

Deliberately mechanical - split, strip, drop empties, recognize one label. No LLM and no
canonicalization of reference areas: whatever vocabulary a submitter used is the vocabulary a
query is matched against, and silently rewriting a value here would make the filter disagree
with the cell a user reads.

Frequencies are the exception, and only because they have a closed vocabulary that validation
already enforces: a token is folded onto its vocabulary spelling so `annual` and `Annual` are
one filter value rather than two.
"""

from statgpt.common.utils.misc import normalize_whitespace

FREQUENCY_VOCABULARY: tuple[str, ...] = (
    "Daily",
    "Business daily",
    "Weekly",
    "Monthly",
    "Quarterly",
    "Semi-annual",
    "Annual",
    "Irregular",
)
"""The frequencies the discovery template tells submitters to choose from (column J).

Lives here rather than in the validator that rejects everything else, because the chat-time
pre-filter offers the same list to a model: one definition, so the values a query can be
narrowed by cannot drift from the values a record may carry.
"""

_FREQUENCY_LOOKUP = {value.casefold(): value for value in FREQUENCY_VOCABULARY}

_PARTNER_LABEL = "partner countries:"
"""Marks the rest of a reference-area cell as partner countries rather than subjects.

The template lets one cell carry both roles, so the label is a divider: it applies to the
token carrying it and to every token after it, not just to the one it precedes.
"""


def split_cell(cell: str) -> list[str]:
    """The values of a ';'-separated cell, in order, whitespace-normalized and de-duplicated.

    Empty tokens are dropped rather than preserved as empty strings: a trailing ';' and a
    ';;' typo are formatting, not a value. Duplicates go too - a filter value repeated in one
    cell would add nothing but would show up twice in the channel's dimensions.
    """
    values: list[str] = []
    seen: set[str] = set()
    for token in cell.split(";"):
        value = normalize_whitespace(token)
        if not value:
            continue
        folded = value.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        values.append(value)
    return values


def parse_reference_areas(cell: str) -> tuple[list[str], list[str]]:
    """Split a reference-area cell into its subject areas and its partner areas.

    Returns `(areas, partner_areas)`. Everything before the `partner countries:` label is a
    subject area; the label's own token, once the label is stripped off it, and every token
    after it are partner areas. A cell without the label has no partner areas.

    Values are kept as submitted - `World` and `Euro area` are values in their own right, not
    shorthand for the countries inside them, so nothing is expanded or rewritten.
    """
    areas: list[str] = []
    partners: list[str] = []
    target = areas

    for value in split_cell(cell):
        if target is areas and (remainder := _strip_partner_label(value)) is not None:
            target = partners
            if remainder:
                partners.append(remainder)
            continue
        target.append(value)

    return areas, partners


def _strip_partner_label(value: str) -> str | None:
    """The value with the partner label removed, or `None` if it does not carry one.

    An empty string is a meaningful answer: the label was the whole token, so the tokens
    after it are partners while this one contributes no value of its own.
    """
    if not value.casefold().startswith(_PARTNER_LABEL):
        return None
    return value[len(_PARTNER_LABEL) :].strip()


def parse_frequencies(cell: str) -> list[str]:
    """The frequencies a frequency-coverage cell names, folded onto the vocabulary's spelling.

    A token outside the vocabulary is kept as submitted rather than dropped. Such a record
    fails validation and is never published, so this only decides what a broken cell looks
    like in a debug stage - and a silently vanishing value would read as a parser bug.
    """
    return [_FREQUENCY_LOOKUP.get(value.casefold(), value) for value in split_cell(cell)]


def is_known_frequency(value: str) -> bool:
    """Whether a single token is one of the template's frequencies, whatever its casing."""
    return value.casefold() in _FREQUENCY_LOOKUP
