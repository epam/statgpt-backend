"""Rendering the datasets the relevance judge kept.

Templates come from channel configuration, so a placeholder the config author typed and this
code does not provide must not break a chat turn: it renders empty instead. `str.format_map`
over a defaulting mapping is what buys that, and it is also why the templates are plain
`str.format` and not Jinja - there is nothing here worth a template engine.
"""

from statgpt.app.schemas.discovery_datasets import DiscoveryCandidate
from statgpt.common.schemas.discovery_datasets_tool import DiscoveryDatasetsTemplates

_ITEM_SEPARATOR = "\n"

_ITEMS_PLACEHOLDER = "items"


class _BlankDefaultingContext(dict):
    """A `format_map` mapping whose unknown keys render as empty strings."""

    def __missing__(self, key: str) -> str:
        return ""


def render_item(template: str, candidate: DiscoveryCandidate, reason: str = "") -> str:
    return template.format_map(_BlankDefaultingContext(candidate.template_context(reason)))


def render_block(
    templates: DiscoveryDatasetsTemplates,
    selected: list[tuple[DiscoveryCandidate, str]],
) -> str | None:
    """The block to append to the data query response, or `None` when there is nothing to say.

    `selected` is `(candidate, reason)` in the order the items should appear - the caller keeps
    rank order. An empty selection renders nothing at all rather than a header with no rows
    under it.
    """
    if not selected:
        return None

    items = _ITEM_SEPARATOR.join(
        render_item(templates.item, candidate, reason).rstrip() for candidate, reason in selected
    )
    block = templates.wrapper.format_map(_BlankDefaultingContext({_ITEMS_PLACEHOLDER: items}))
    return block.strip() or None
