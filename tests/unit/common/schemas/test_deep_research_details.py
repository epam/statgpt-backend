"""Tests for the `DeepResearchDetails` accessors that resolve the Deep Research access gate —
`get_access_claim` (the claim name) and `get_access_claim_value` (the required value) — including
their `$env:{VAR}` support, before the claim is checked against a caller's token.
"""

from statgpt.common.schemas.tool_details import DeepResearchDetails


def _details(**overrides: object) -> DeepResearchDetails:
    return DeepResearchDetails.model_validate({"deploymentId": "dr-app", **overrides})


def test_access_claim_defaults_to_none() -> None:
    assert _details().get_access_claim() is None


def test_access_claim_literal_value() -> None:
    assert _details(access_claim="dr_access").get_access_claim() == "dr_access"


def test_access_claim_accepts_camel_case_alias() -> None:
    assert _details(accessClaim="dr_access").get_access_claim() == "dr_access"


def test_access_claim_blank_resolves_to_none() -> None:
    assert _details(access_claim="   ").get_access_claim() is None


def test_access_claim_resolves_env_var(monkeypatch) -> None:
    monkeypatch.setenv("DR_CLAIM", "dr_access")
    assert _details(access_claim="$env:{DR_CLAIM}").get_access_claim() == "dr_access"


def test_access_claim_env_var_default_used_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("DR_CLAIM", raising=False)
    assert _details(access_claim="$env:{DR_CLAIM|dr_default}").get_access_claim() == "dr_default"


def test_access_claim_missing_env_without_default_fails_closed(monkeypatch) -> None:
    # No env var and no default: the template is left unresolved rather than silently opening
    # access. It becomes a claim name no token will carry, so gating denies access (fail closed).
    # Use an empty default (`$env:{VAR|}`) instead if "unset means open" is intended.
    monkeypatch.delenv("DR_CLAIM", raising=False)
    assert _details(access_claim="$env:{DR_CLAIM}").get_access_claim() == "$env:{DR_CLAIM}"


def test_access_claim_empty_default_opens_access_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("DR_CLAIM", raising=False)
    assert _details(access_claim="$env:{DR_CLAIM|}").get_access_claim() is None


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
