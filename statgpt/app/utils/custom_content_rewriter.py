from collections.abc import Sequence
from typing import Any

from statgpt.common.schemas import CustomContentKind, CustomContentRewriteRule


class CustomContentRewriter:
    """Applies the configured rewrite rules to annotations and attachments of a sub-deployment.

    Rules are checked in order and only the first matching rule is applied. Items are returned as
    rewritten copies; an item no rule matches is returned unchanged."""

    def __init__(self, rules: Sequence[CustomContentRewriteRule]):
        self._rules = rules

    def rewrite_annotation(self, annotation: dict[str, Any]) -> dict[str, Any]:
        return self._rewrite(annotation, CustomContentKind.annotation)

    def rewrite_attachment(self, attachment: dict[str, Any]) -> dict[str, Any]:
        return self._rewrite(attachment, CustomContentKind.attachment)

    def _rewrite(self, item: dict[str, Any], kind: CustomContentKind) -> dict[str, Any]:
        for rule in self._rules:
            if rule.selector.matches(item, kind):
                return rule.apply(item, kind)
        return item
