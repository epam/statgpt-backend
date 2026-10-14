"""The citation tags of a sub-deployment's response and the sources they cite.

A sub-deployment writes a `<cit data-id="...">` tag at each cited position and claims it with an
annotation (see `dial_annotations`) that names the cited source, and may quote the cited passage.
The client draws a pill from the annotation in place of the tag, but the agent reads only the tag,
so `OpenAiToDialStreamer` lists the source and the quote of every claimed tag for the tool
response to name them to the agent.
"""

from typing import Any

from pydantic import BaseModel

# Where an annotation names its source, best first: the label of the pill, then the title and the
# url of the file it cites. The same fields the rewrite rules know as `body_title`,
# `attachment_title` and `url`.
_SOURCE_TITLE_PATHS = (
    ('body', 'title'),
    ('body', 'source', 'attachment', 'title'),
    ('body', 'source', 'attachment', 'url'),
)


class Citation(BaseModel):
    """A citation tag of a tool response and the source it cites, as the agent is told it."""

    id: str
    # None if no annotation claiming the tag names its source.
    title: str | None
    # The cited passage, e.g. the filters of a dataset query; None if the annotation quotes none.
    quote: str | None = None


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


def get_source_quote(annotation: dict[str, Any]) -> str | None:
    """The passage of the source the annotation quotes, or None."""
    body = annotation.get('body')
    quote = body.get('quote') if isinstance(body, dict) else None
    return quote if isinstance(quote, str) and quote else None
