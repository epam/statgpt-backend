"""Rendering the datasets the relevance judge kept.

Templates come from channel configuration, so a placeholder the config author typed and this
code does not provide must not break a chat turn: it renders empty instead. `str.format_map`
over a defaulting mapping is what buys that, and it is also why the templates are plain
`str.format` and not Jinja - there is nothing here worth a template engine.
"""

from typing import Any

from statgpt.app.schemas.discovery_datasets import DiscoveryCandidate, SelectedDiscoveryDataset
from statgpt.common.schemas.discovery_datasets_tool import DiscoveryDatasetsTemplates

_ITEM_SEPARATOR = "\n"

_ITEMS_PLACEHOLDER = "items"
"""The wrapper's own placeholder, and this module's alone to define.

Dropped from the item context so an item template cannot recurse into it.
"""


class _BlankDefaultingContext(dict[str, Any]):
    """A `format_map` mapping whose unknown keys render as empty strings."""

    def __missing__(self, key: str) -> str:
        return ""


def _render_item(template: str, candidate: DiscoveryCandidate, reason: str = "") -> str:
    context = candidate.template_context(reason)
    context.pop(_ITEMS_PLACEHOLDER, None)
    return template.format_map(_BlankDefaultingContext(context))


def render_block(
    templates: DiscoveryDatasetsTemplates,
    selected: list[SelectedDiscoveryDataset],
) -> str | None:
    """The block to append to the data query response, or `None` when there is nothing to say.

    `selected` is in the order the items should appear - the caller keeps rank order. An empty
    selection renders nothing at all rather than a header with no rows under it.
    """
    if not selected:
        return None

    items = _ITEM_SEPARATOR.join(
        _render_item(templates.item, item.candidate, item.reason).rstrip() for item in selected
    )
    block = templates.wrapper.format_map(_BlankDefaultingContext({_ITEMS_PLACEHOLDER: items}))
    return block.strip() or None
