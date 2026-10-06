"""Tests for `DeepResearchDetails`: `get_access_claim_value`, the accessor that resolves the Deep
Research access role — including its `$env:{VAR}` support — before it is checked against the
caller's DIAL roles, and the `query_check` config of the check run before a session is started.
"""

import pytest
from pydantic import ValidationError

from statgpt.common.schemas.tool_details import DeepResearchDetails


def _details(**overrides: object) -> DeepResearchDetails:
    return DeepResearchDetails.model_validate({"deploymentId": "dr-app", **overrides})


def test_access_claim_value_defaults_to_none() -> None:
    assert _details().get_access_claim_value() is None


def test_access_claim_value_literal_value() -> None:
    assert _details(access_claim_value="dr_access").get_access_claim_value() == "dr_access"


def test_access_claim_value_accepts_camel_case_alias() -> None:
    assert _details(accessClaimValue="dr_access").get_access_claim_value() == "dr_access"


def test_access_claim_value_blank_resolves_to_none() -> None:
    assert _details(access_claim_value="   ").get_access_claim_value() is None


def test_access_claim_value_resolves_env_var(monkeypatch) -> None:
    monkeypatch.setenv("DR_ROLE", "dr_access")
    assert _details(access_claim_value="$env:{DR_ROLE}").get_access_claim_value() == "dr_access"


def test_access_claim_value_env_var_default_used_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("DR_ROLE", raising=False)
    assert (
        _details(access_claim_value="$env:{DR_ROLE|dr_default}").get_access_claim_value()
        == "dr_default"
    )


def test_access_claim_value_empty_default_opens_access_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("DR_ROLE", raising=False)
    assert _details(access_claim_value="$env:{DR_ROLE|}").get_access_claim_value() is None


def test_query_check_defaults() -> None:
    query_check = _details().query_check
    assert query_check.enabled is True
    assert query_check.timeout_seconds == 10
    assert query_check.excluded_topics is None  # the default topics are used
    assert query_check.skipped_message


def test_query_check_accepts_camel_case_aliases() -> None:
    query_check = _details(
        queryCheck={
            "enabled": False,
            "timeoutSeconds": 30,
            "excludedTopics": ["Weather"],
            "skippedMessage": "",
            "llmModelConfig": {"temperature": 0.5},
        }
    ).query_check
    assert query_check.enabled is False
    assert query_check.timeout_seconds == 30
    assert query_check.excluded_topics == ["Weather"]
    assert query_check.skipped_message == ""
    assert query_check.llm_model_config.temperature == 0.5


@pytest.mark.parametrize("timeout_seconds", [0, -1])
def test_query_check_timeout_must_be_positive(timeout_seconds: float) -> None:
    with pytest.raises(ValidationError):
        _details(queryCheck={"timeoutSeconds": timeout_seconds})
