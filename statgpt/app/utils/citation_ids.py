"""The citation tags of a sub-deployment's response, the sources they cite, and their ids.

A sub-deployment writes a `<cit data-id="...">` tag at each cited position and claims it with an
annotation (see `dial_annotations`) that names the cited source, and may quote the cited passage.
The client draws a pill from the annotation in place of the tag, but the agent reads only the tag,
so `OpenAiToDialStreamer` lists the source of every claimed tag, and if asked the quote, for the
tool response to name them to the agent.

The ids a sub-deployment picks are its own: a long hash such as
`94c80f43b36f40ffbd9063db0cdc35c6`, or a counter such as `cit-1` that the next tool call starts
again. The agent has to copy the tags of a tool response into its answer verbatim for the client
to resolve them, so `OpenAiToDialStreamer` renumbers every claimed id into the id space of the
response: `citation001-k3f9`, `citation002-k3f9`, ... A short id is copied without typos, is easy
to match against the sources note of a tool response, and never names two sources in one
conversation, whichever tool cited them.
"""

import itertools
import re
import secrets
import string
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from pydantic import BaseModel

# The `data-id` attribute of a marker tag, e.g. `<cit data-id="citation001">`, in either quote and
# with optional whitespace around `=`. An unquoted value is not recognised: the streamer logs every
# claimed tag it could not renumber. Groups: everything up to the opening quote, the quote, the id.
_TAG_ID_ATTRIBUTE = re.compile(r'''(<\w[^>]*?\sdata-id\s*=\s*(["']))(.*?)\2''')

# A whole, empty citation tag, `<cit data-id="..."></cit>` or `<cit data-id="..." />`, its id
# quoted as `_TAG_ID_ATTRIBUTE` reads it, and the spaces before it on its line. The id can't hold a
# quote or an angle bracket: an id that could would grow past a tag that is not empty, such as
# `<cit data-id="a">text</cit>`, up to the quote of the next tag, and swallow the text between.
_CITATION_TAG = re.compile(
    r'''(?P<space>[ \t]*)<cit\b[^>]*?\sdata-id\s*=\s*(["'])(?P<id>[^"'<>]*)\2[^>]*?'''
    r'''(?:/>|>\s*</cit\s*>)'''
)

# A quote longer than this is cut short in a note or a reference: it is often a long page excerpt,
# and the start of it is enough to tell the cited passage.
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
    file, not the cited passage. Only a plain-text reference, which no pill comes with, falls back
    to them (see `to_plain_text_references`).
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


class _Reference(NamedTuple):
    """A source of a citation tag, as a plain-text reference names it."""

    tag_id: str
    title: str | None
    url: str | None
    quote: str | None


def to_plain_text_references(content: str, annotations: Sequence[dict[str, Any]]) -> str:
    """`content` with plain-text references in place of its citation tags, for a client that
    gets no annotations to resolve the tags with, such as an MCP client.

    `annotations` are the annotations relayed with `content`. Each cited tag becomes `[1]`, `[2]`,
    ..., in the order the tags first appear in `content`, and the references are listed after the
    content by number, each distinct source of a tag once. A tag that cites nothing is dropped.

    No pill names the cited file to such a client, so a reference falls back to the title of the
    file if the citation card has none, and gives the url of the file if the client can open it.
    """
    references = list(
        dict.fromkeys(
            _Reference(
                tag_id=tag_id,
                title=get_source_title(annotation) or _get_file_title(annotation),
                url=_get_file_link(annotation),
                quote=get_source_quote(annotation),
            )
            for annotation in annotations
            if (tag_id := get_tag_id(annotation)) is not None
        )
    )
    claimed = {r.tag_id for r in references}
    # Numbered as the text shows the tags, so that it reads `[1]`, `[2]`, ...: the order of the
    # annotations need not match it. A claimed tag the text does not show comes last.
    in_text = (m['id'] for m in _CITATION_TAG.finditer(content) if m['id'] in claimed)
    order = dict.fromkeys(itertools.chain(in_text, (r.tag_id for r in references)))
    numbers = {tag_id: n for n, tag_id in enumerate(order, 1)}
    text = _CITATION_TAG.sub(
        lambda m: f'{m["space"]}[{numbers[m["id"]]}]' if m['id'] in numbers else '', content
    )
    if not numbers:
        return text
    # The sources of one tag together, in the order they were claimed: the sort is stable.
    lines = '\n'.join(
        _format_reference(numbers[r.tag_id], r)
        for r in sorted(references, key=lambda r: numbers[r.tag_id])
    )
    return f'{text}\n\n### References:\n\n{lines}'


def _get_cited_file(annotation: dict[str, Any]) -> dict[str, Any]:
    """The file the annotation cites (`body.source.attachment`), or an empty dict."""
    value: Any = annotation
    for key in ('body', 'source', 'attachment'):
        value = value.get(key) if isinstance(value, dict) else None
    return value if isinstance(value, dict) else {}


def _get_file_title(annotation: dict[str, Any]) -> str | None:
    title = _get_cited_file(annotation).get('title')
    return title if isinstance(title, str) and title else None


def _get_file_link(annotation: dict[str, Any]) -> str | None:
    """The url of the cited file if a client outside DIAL can open it, or None.

    A DIAL file url, `files/...`, is relative to DIAL Core: it is left out unless a rewrite rule
    of the tool made it public.
    """
    url = _get_cited_file(annotation).get('url')
    return url if isinstance(url, str) and url.startswith(('https://', 'http://')) else None


def _format_reference(number: int, reference: _Reference) -> str:
    """One item of the references list: `- [1] Title (url): "quote"`, leaving out what is
    missing."""
    name: str | None
    if reference.title and reference.url:
        name = f'{reference.title} ({reference.url})'
    else:
        name = reference.title or reference.url
    # The quote on one line, so that each reference is one item of the list.
    quote = f'"{" ".join(reference.quote.split())}"' if reference.quote else None
    source = ': '.join(part for part in (name, quote) if part)
    return f'- [{number}] {source or "(the source is not named)"}'
