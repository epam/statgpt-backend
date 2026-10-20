"""The citation tags of a sub-deployment's response, the sources they cite, and their ids.

A sub-deployment writes a `<cit data-id="...">` tag at each cited position and claims it with an
annotation (see `dial_annotations`) that names the cited source, and may quote the cited passage.
The client draws a pill from the annotation in place of the tag, but the agent reads only the tag,
so `OpenAiToDialStreamer` lists the source and the quote of every claimed tag for the tool
response to name them to the agent.

The ids a sub-deployment picks are its own: a long hash such as
`94c80f43b36f40ffbd9063db0cdc35c6`, or a counter such as `cit-1` that the next tool call starts
again. The agent has to copy the tags of a tool response into its answer verbatim for the client
to resolve them, so `OpenAiToDialStreamer` renumbers every claimed id into the id space of the
response: `citation001-k3f9`, `citation002-k3f9`, ... A short id is copied without typos, is easy
to match against the sources note of a tool response, and never names two sources in one
conversation, whichever tool cited them.
"""

import re
import secrets
import string
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

# The `data-id` attribute of a marker tag, e.g. `<cit data-id="citation001">`, in either quote and
# with optional whitespace around `=`. An unquoted value is not recognised: the streamer logs every
# claimed tag it could not renumber. Groups: everything up to the opening quote, the quote, the id.
_TAG_ID_ATTRIBUTE = re.compile(r'''(<\w[^>]*?\sdata-id\s*=\s*(["']))(.*?)\2''')

# A whole, empty citation tag, `<cit data-id="..."></cit>` or `<cit data-id="..." />`, its id
# quoted as `_TAG_ID_ATTRIBUTE` reads it, and the spaces before it on its line.
_CITATION_TAG = re.compile(
    r'''(?P<space>[ \t]*)<cit\b[^>]*?\sdata-id\s*=\s*(["'])(?P<id>.*?)\2[^>]*?(?:/>|>\s*</cit\s*>)'''
)

# A quote longer than this is cut short in the note: it is often a long page excerpt, and the
# start of it is enough for the agent to tell the cited passage.
_MAX_QUOTE_LENGTH = 500

_SCOPE_ALPHABET = string.ascii_lowercase + string.digits
_SCOPE_LENGTH = 4


class Citation(BaseModel):
    """A citation tag of a tool response and the source it cites, as the agent is told it."""

    id: str
    # None if no annotation claiming the tag titles its citation card.
    title: str | None
    # The cited passage, e.g. the filters of a dataset query; None if the annotation quotes none.
    quote: str | None = None


class CitationIdSpace:
    """Hands out the citation ids of a response, in the order they are first claimed.

    One per response, shared by every `OpenAiToDialStreamer` of that response. Its random scope
    keeps the ids apart from those of every other response of the conversation, which the history
    still shows the agent and the client still resolves: `citation001-k3f9`. Nothing has to be
    carried over from the earlier responses, so a response that was stopped or failed before
    saving its state cannot make the next one reuse its ids.

    `next_id` deliberately contains no `await`, so concurrent tool calls cannot interleave
    inside it and it needs no lock.
    """

    def __init__(self, scope: str | None = None) -> None:
        """`scope` is random unless given, which only a test should do."""
        if scope is None:
            scope = ''.join(secrets.choice(_SCOPE_ALPHABET) for _ in range(_SCOPE_LENGTH))
        self._scope = scope
        self._count = 0

    def next_id(self) -> str:
        self._count += 1
        return f'citation{self._count:03d}-{self._scope}'


def get_tag_id(annotation: dict[str, Any]) -> str | None:
    """The id of the marker tag the annotation claims, or None if it claims no tag."""
    target = annotation.get('target')
    selector = target.get('selector') if isinstance(target, dict) else None
    if not isinstance(selector, dict) or selector.get('type') != 'html_tag':
        return None
    tag_id = selector.get('id')
    return tag_id if isinstance(tag_id, str) else None


def get_source_title(annotation: dict[str, Any]) -> str | None:
    """The title of the citation card (`body.title`), or None.

    The title of the cited file (the label of the pill) and its url are not used: they name the
    file, not the cited passage.
    """
    body = annotation.get('body')
    title = body.get('title') if isinstance(body, dict) else None
    return title if isinstance(title, str) and title else None


def get_source_quote(annotation: dict[str, Any]) -> str | None:
    """The passage of the source the annotation quotes, cut to `_MAX_QUOTE_LENGTH`, or None."""
    body = annotation.get('body')
    quote = body.get('quote') if isinstance(body, dict) else None
    if not isinstance(quote, str) or not quote:
        return None
    if len(quote) > _MAX_QUOTE_LENGTH:
        return quote[:_MAX_QUOTE_LENGTH].rstrip() + '…'
    return quote


def with_tag_id(annotation: dict[str, Any], tag_id: str) -> dict[str, Any]:
    """A copy of an annotation that claims a tag (see `get_tag_id`), claiming `tag_id` instead."""
    target = annotation['target']
    return {**annotation, 'target': {**target, 'selector': {**target['selector'], 'id': tag_id}}}


def find_tag_ids(content: str) -> set[str]:
    """The id of every marker tag in `content`."""
    return {match[3] for match in _TAG_ID_ATTRIBUTE.finditer(content)}


def replace_tag_ids(content: str, tag_ids: Mapping[str, str]) -> str:
    """Replace the id of every marker tag in `content` that `tag_ids` maps; leave the rest."""
    if not tag_ids:
        return content
    return _TAG_ID_ATTRIBUTE.sub(lambda m: m[1] + tag_ids.get(m[3], m[3]) + m[2], content)


def to_plain_text_references(content: str, citations: Sequence[Citation]) -> str:
    """`content` with plain-text references in place of its citation tags, for a client that
    gets no annotations to resolve the tags with, such as an MCP client.

    Each cited tag becomes `[1]`, `[2]`, ..., in the order the tags were first claimed, and the
    references are listed after the content. A tag that cites nothing is dropped.
    """
    numbers = {tag_id: n for n, tag_id in enumerate(dict.fromkeys(c.id for c in citations), 1)}
    text = _CITATION_TAG.sub(
        lambda m: f'{m["space"]}[{numbers[m["id"]]}]' if m['id'] in numbers else '', content
    )
    if not numbers:
        return text
    references = '\n'.join(_format_reference(numbers[c.id], c) for c in citations)
    return f'{text}\n\n### References:\n\n{references}'


def _format_reference(number: int, citation: Citation) -> str:
    """One item of the references list: `- [1] Title: "quote"`, leaving out what is missing."""
    # The quote on one line, so that each reference is one item of the list.
    quote = f'"{" ".join(citation.quote.split())}"' if citation.quote else None
    source = ': '.join(part for part in (citation.title, quote) if part)
    return f'- [{number}] {source or "(the source is not named)"}'
