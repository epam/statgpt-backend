"""The citation tag ids of a conversation: `citation001`, `citation002`, ...

A sub-deployment writes a `<cit data-id="...">` tag at each cited position and claims it with an
annotation (see `dial_annotations`). The ids it picks are its own: a long hash such as
`94c80f43b36f40ffbd9063db0cdc35c6`, or a counter such as `cit-1` that the next tool call starts
again. The agent has to copy the tags of a tool response into its answer verbatim for the client
to resolve them, so `OpenAiToDialStreamer` renumbers every claimed id into the one id space of
the conversation. A short id is copied without typos, is easy to match against the sources note
of a tool response, and never names two sources in one conversation, whichever tool cited them.
"""

import re
import secrets
import string
from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel

# The `data-id` attribute of a marker tag, e.g. `<cit data-id="citation001">`, in either quote and
# with optional whitespace around `=`. An unquoted value is not recognised: the streamer logs every
# claimed tag it could not renumber. Groups: everything up to the opening quote, the quote, the id.
_TAG_ID_ATTRIBUTE = re.compile(r'''(<\w[^>]*?\sdata-id\s*=\s*(["']))(.*?)\2''')

# Where an annotation names its source, best first: the label of the pill, then the title and the
# url of the file it cites. The same fields the rewrite rules know as `body_title`,
# `attachment_title` and `url`.
_SOURCE_TITLE_PATHS = (
    ('body', 'title'),
    ('body', 'source', 'attachment', 'title'),
    ('body', 'source', 'attachment', 'url'),
)

_SCOPE_ALPHABET = string.ascii_lowercase + string.digits
_SCOPE_LENGTH = 4


class Citation(BaseModel):
    """A citation tag of a tool response and the source it cites, as the agent is told it."""

    id: str
    # None if no annotation claiming the tag names its source.
    title: str | None


class CitationIdSpace:
    """Hands out the citation ids of a conversation, in the order they are first claimed.

    One per response, shared by every `OpenAiToDialStreamer` of that response, and started at
    the `count` the previous response persisted, so that an id stays unique across turns.

    `next_id` deliberately contains no `await`, so concurrent tool calls cannot interleave
    inside it and it needs no lock.
    """

    def __init__(self, count: int = 0, scope: str | None = None) -> None:
        self._count = count
        self._suffix = f'-{scope}' if scope else ''

    @classmethod
    def scoped(cls) -> Self:
        """A space for a call that cannot continue the ids of the calls before it.

        An MCP tool call gets no state back from its client, so a plain space would start every
        call at `citation001`, and a client that called the tool twice would read one id naming
        two sources. A random scope keeps the ids of each call apart: `citation001-k3f9`.
        """
        return cls(scope=''.join(secrets.choice(_SCOPE_ALPHABET) for _ in range(_SCOPE_LENGTH)))

    @property
    def count(self) -> int:
        """How many ids the conversation has handed out so far."""
        return self._count

    def next_id(self) -> str:
        self._count += 1
        return f'citation{self._count:03d}{self._suffix}'


def get_tag_id(annotation: dict[str, Any]) -> str | None:
    """The id of the marker tag the annotation claims, or None if it claims no tag."""
    target = annotation.get('target')
    selector = target.get('selector') if isinstance(target, dict) else None
    if not isinstance(selector, dict) or selector.get('type') != 'html_tag':
        return None
    tag_id = selector.get('id')
    return tag_id if isinstance(tag_id, str) else None


def get_source_title(annotation: dict[str, Any]) -> str | None:
    """The name of the source the annotation cites (see `_SOURCE_TITLE_PATHS`), or None."""
    for path in _SOURCE_TITLE_PATHS:
        value: Any = annotation
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, str) and value:
            return value
    return None


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
