import copy
import re
from typing import Any

from pydantic import Field, model_validator

from statgpt.common.config import utils as config_utils

from .base import BaseYamlModel
from .enums import CustomContentKind, RewriteAppliesTo, RewriteSelectorField, RewriteTargetField

_ANNOTATION_ATTACHMENT = ('body', 'source', 'attachment')

# Where each logical field lives in each kind of item. A field missing for a kind cannot be used
# by a rule that applies to that kind. Extend this map (and `RewriteTargetField`) to make more
# fields rewritable.
_FIELD_PATHS: dict[CustomContentKind, dict[str, tuple[str, ...]]] = {
    CustomContentKind.annotation: {
        'type': (*_ANNOTATION_ATTACHMENT, 'type'),
        'url': (*_ANNOTATION_ATTACHMENT, 'url'),
        'attachment_title': (*_ANNOTATION_ATTACHMENT, 'title'),
        'body_title': ('body', 'title'),
    },
    CustomContentKind.attachment: {
        'type': ('type',),
        'url': ('url',),
        'reference_url': ('reference_url',),
        'attachment_title': ('title',),
    },
}


def _field_path(kind: CustomContentKind, field: str) -> tuple[str, ...] | None:
    return _FIELD_PATHS[kind].get(field)


def _check_applicable(field: str, applies_to: RewriteAppliesTo) -> None:
    for kind in applies_to.kinds:
        if _field_path(kind, field) is None:
            raise ValueError(f"Field `{field}` is not available for `{kind}` items.")


def _get_value(item: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = item
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _set_value(item: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    parent = _get_value(item, path[:-1])
    parent[path[-1]] = value


class RewriteSelector(BaseYamlModel):
    applies_to: RewriteAppliesTo = Field(
        description="Which kinds of items the rule applies to: annotations, attachments, or both."
    )
    type: re.Pattern[str] | None = Field(
        default=None, description="Regex searched in the MIME type of the (source) attachment."
    )
    url: re.Pattern[str] | None = Field(
        default=None, description="Regex searched in the url of the (source) attachment."
    )
    reference_url: re.Pattern[str] | None = Field(
        default=None,
        description="Regex searched in the `reference_url` of the attachment. Attachments only.",
    )
    attachment_title: re.Pattern[str] | None = Field(
        default=None, description="Regex searched in the title of the (source) attachment."
    )
    body_title: re.Pattern[str] | None = Field(
        default=None,
        description="Regex searched in the title of the annotation body. Annotations only.",
    )

    def _get_conditions(self) -> dict[str, re.Pattern[str]]:
        return {
            field.value: pattern
            for field in RewriteSelectorField
            if (pattern := getattr(self, field.value)) is not None
        }

    @model_validator(mode='after')
    def _validate_conditions(self) -> "RewriteSelector":
        conditions = self._get_conditions()
        if not conditions:
            raise ValueError(
                "Rewrite selector must specify at least one of: "
                + ", ".join(f"`{field}`" for field in RewriteSelectorField)
            )
        for field in conditions:
            _check_applicable(field, self.applies_to)
        return self

    def matches(self, item: dict[str, Any], kind: CustomContentKind) -> bool:
        """Whether every specified regex is found in its field. A missing field never matches."""
        if kind not in self.applies_to.kinds:
            return False
        for field, pattern in self._get_conditions().items():
            path = _field_path(kind, field)
            value = _get_value(item, path) if path is not None else None
            if not isinstance(value, str) or pattern.search(value) is None:
                return False
        return True


class FieldRewrite(BaseYamlModel):
    field: RewriteTargetField = Field(description="The field to modify.")
    pattern: re.Pattern[str] = Field(description="Regex of the part of the field to replace.")
    replacement: str = Field(
        description=(
            "The new value of the matched part. Supports regex backreferences (e.g. `\\1`)"
            " and $env:{VAR} syntax."
        )
    )

    def get_replacement(self) -> str:
        return config_utils.replace_env(self.replacement)

    def apply(self, item: dict[str, Any], kind: CustomContentKind) -> None:
        """Rewrite the field of `item` in place. A missing field is skipped."""
        path = _field_path(kind, self.field)
        if path is None:
            return
        value = _get_value(item, path)
        if isinstance(value, str):
            _set_value(item, path, self.pattern.sub(self.get_replacement(), value))


class CustomContentRewriteRule(BaseYamlModel):
    selector: RewriteSelector = Field(description="Which items the rule applies to.")
    rewrites: list[FieldRewrite] = Field(
        min_length=1, description="The modifications applied to a selected item, in order."
    )

    @model_validator(mode='after')
    def _validate_rewrites(self) -> "CustomContentRewriteRule":
        for rewrite in self.rewrites:
            _check_applicable(rewrite.field, self.selector.applies_to)
        return self

    def apply(self, item: dict[str, Any], kind: CustomContentKind) -> dict[str, Any]:
        """Return a rewritten copy of `item`; the input is not mutated."""
        result = copy.deepcopy(item)
        for rewrite in self.rewrites:
            rewrite.apply(result, kind)
        return result
