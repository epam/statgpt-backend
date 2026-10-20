"""Repairing the citation marker tags the Supreme Agent writes into its answer.

A tool response may carry `<cit data-id="..."></cit>` tags, each claimed by an annotation that is
relayed to the choice (see `dial_annotations`). The agent copies the tags into its answer, and the
client resolves a tag only if it parses as an element: a tag the agent mistyped, such as
`<cit data-id="9f81df22e8b442e8a491d163198b7d4a></cit>` with its closing quote dropped, is shown
as literal text instead of a citation pill, although its annotation reached the client.
`CitationTagRepairer` rewrites every such tag of the answer into the canonical form as it streams.
"""

import re

_TAG_START = '<cit'

# A whole tag, leniently: the attributes, then either a self-closing end or a closing tag. The `>`
# before the closing tag is optional, since the agent drops it as easily as it drops a quote.
_TAG = re.compile(r'<cit(?P<attrs>[\s/][^<>]*?)?(?:/>|>?\s*</cit\s*>)')

# An opening tag that no closing tag follows. Repaired too, or the client would read the text
# after it as the content of the tag.
_OPENING_TAG = re.compile(r'<cit(?P<attrs>[\s/][^<>]*)>')

# The text a whole tag still grows from as more of the stream arrives, e.g. `<cit data-id="9f8`.
_TAG_PREFIX = re.compile(r'<cit(?:[\s/][^<>]*)?(?:>\s*)?(?:<(?:/(?:c(?:i(?:t\s*)?)?)?)?)?')

# The `data-id` value, quoted with either quote, unquoted, or with one of its quotes missing.
_DATA_ID = re.compile(r'''\bdata-id\s*=\s*["']?(?P<id>[^"'\s<>/]+)''')

# Text that starts like a tag but is still not one at this length is let through as it is.
# A canonical tag with a 32-character id is 54 characters long.
_MAX_TAG_LENGTH = 256


class CitationTagRepairer:
    """Rewrites the citation tags of a stream into `<cit data-id="..."></cit>`, chunk by chunk.

    Text that may be the start of a tag is held back until it is known whether it is one, so a
    tag split across chunks is repaired as a whole. All other text is let through at once.
    """

    def __init__(self) -> None:
        self._pending = ''

    def feed(self, text: str) -> str:
        """Take the next chunk of the stream; return the text that is ready to be sent."""
        self._pending += text
        return self._drain(final=False)

    def flush(self) -> str:
        """Return the text still held back, at the end of the stream."""
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        ready: list[str] = []
        rest = self._pending
        while (start := rest.find('<')) != -1:
            ready.append(rest[:start])
            rest = rest[start:]
            if match := _TAG.match(rest):
                ready.append(_repair(match))
                rest = rest[match.end() :]
            elif not final and _may_become_tag(rest):
                self._pending = rest
                return ''.join(ready)
            elif match := _OPENING_TAG.match(rest):
                ready.append(_repair(match))
                rest = rest[match.end() :]
            else:
                ready.append('<')
                rest = rest[1:]
        ready.append(rest)
        self._pending = ''
        return ''.join(ready)


def _may_become_tag(text: str) -> bool:
    if len(text) > _MAX_TAG_LENGTH:
        return False
    return _TAG_START.startswith(text) or _TAG_PREFIX.fullmatch(text) is not None


def _repair(match: re.Match[str]) -> str:
    """The canonical form of a matched tag, or the tag as it is if it names no id."""
    id_match = _DATA_ID.search(match['attrs'] or '')
    if id_match is None:
        return match[0]
    return f'<cit data-id="{id_match["id"]}"></cit>'
