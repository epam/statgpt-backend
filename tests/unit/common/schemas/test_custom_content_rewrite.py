"""Validation of the Deep Research `rewrite_rules` config."""

import pytest
from pydantic import ValidationError

from statgpt.common.schemas import CustomContentRewriteRule
from statgpt.common.schemas.tool_details import DeepResearchDetails

_REWRITE = {"field": "url", "pattern": "^files/", "replacement": "https://public/"}


def _rule(selector: dict, rewrites: list[dict] | None = None) -> CustomContentRewriteRule:
    return CustomContentRewriteRule.model_validate(
        {"selector": selector, "rewrites": rewrites if rewrites is not None else [_REWRITE]}
    )


def test_rewrite_rules_default_to_empty() -> None:
    assert DeepResearchDetails.model_validate({"deploymentId": "dr"}).rewrite_rules == []


def test_rewrite_rules_parsed_from_camel_case() -> None:
    details = DeepResearchDetails.model_validate(
        {
            "deploymentId": "dr",
            "rewriteRules": [
                {
                    "selector": {"appliesTo": "attachment", "referenceUrl": "^files/"},
                    "rewrites": [{**_REWRITE, "field": "reference_url"}],
                }
            ],
        }
    )
    reference_url = details.rewrite_rules[0].selector.reference_url
    assert reference_url is not None and reference_url.pattern == "^files/"


def test_selector_requires_a_condition() -> None:
    with pytest.raises(ValidationError, match="at least one of"):
        _rule({"applies_to": "both"})


def test_invalid_regex_rejected() -> None:
    with pytest.raises(ValidationError):
        _rule({"applies_to": "both", "url": "("})


@pytest.mark.parametrize(
    ("field", "applies_to"),
    [
        ("reference_url", "annotation"),
        ("reference_url", "both"),
        ("body_title", "attachment"),
        ("body_title", "both"),
    ],
)
def test_selector_field_not_available_for_kind(field: str, applies_to: str) -> None:
    with pytest.raises(ValidationError, match=f"`{field}` is not available"):
        _rule({"applies_to": applies_to, field: "x"})


@pytest.mark.parametrize("applies_to", ["annotation", "both"])
def test_reference_url_rewrite_requires_attachments_only(applies_to: str) -> None:
    with pytest.raises(ValidationError, match="`reference_url` is not available"):
        _rule({"applies_to": applies_to, "url": "x"}, [{**_REWRITE, "field": "reference_url"}])


def test_rewrites_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        _rule({"applies_to": "both", "url": "x"}, [])


def test_replacement_resolves_env_var(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_FILES_URL", "https://public")
    rule = _rule(
        {"applies_to": "both", "url": "x"},
        [{**_REWRITE, "replacement": "$env:{PUBLIC_FILES_URL}/"}],
    )
    assert rule.rewrites[0].get_replacement() == "https://public/"
