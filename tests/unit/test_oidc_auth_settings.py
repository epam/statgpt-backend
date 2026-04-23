import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

# The module creates a singleton OidcAuthSettings() at import time, which runs
# validation and requires OIDC env vars.  We must set them BEFORE importing.
_MODULE_LEVEL_ENV = {
    "OIDC_AUTH_ENABLED": "false",
}

with patch.dict(os.environ, _MODULE_LEVEL_ENV):
    from statgpt.admin.settings.oidc_auth import OidcAuthSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All required OIDC fields when auth is enabled
_FULL_OIDC_ENV = {
    "OIDC_AUTH_ENABLED": "true",
    "OIDC_CONFIGURATION_ENDPOINT": "https://login.example.com/.well-known/openid-configuration",
    "OIDC_CLIENT_ID": "my-client-id",
    "OIDC_ISSUER": "https://login.example.com",
    "OIDC_USERNAME_CLAIM": "preferred_username",
    "ADMIN_ROLES_CLAIM": "roles",
    "ADMIN_ROLES_VALUES": "admin,superadmin",
    "ADMIN_SCOPE_CLAIM": "scp",
    "ADMIN_SCOPE_VALUE": "admin_access",
    "ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED": "true",
}


def _make_settings(**overrides) -> OidcAuthSettings:
    """Create OidcAuthSettings from a dict of env-like values merged with defaults."""
    env = {**_FULL_OIDC_ENV, **overrides}
    return OidcAuthSettings(**env)


# ---------------------------------------------------------------------------
# Validation: OIDC disabled – no required fields enforced
# ---------------------------------------------------------------------------


class TestOidcDisabled:
    def test_no_required_fields_when_disabled(self):
        """When OIDC auth is disabled, missing required fields should not raise."""
        settings = OidcAuthSettings(OIDC_AUTH_ENABLED=False)
        assert settings.oidc_auth_enabled is False
        assert settings.oidc_configuration_endpoint is None
        assert settings.oidc_client_id is None

    def test_disabled_ignores_scope_fields(self):
        """Scope validation should be skipped entirely when OIDC is disabled."""
        settings = OidcAuthSettings(
            OIDC_AUTH_ENABLED=False,
            ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED=True,
        )
        assert settings.admin_scope_claim is None
        assert settings.admin_scope_value is None


# ---------------------------------------------------------------------------
# Validation: OIDC enabled – missing required fields
# ---------------------------------------------------------------------------


class TestOidcEnabledValidation:
    def test_missing_all_required_fields_raises(self):
        """Enabling OIDC without any required fields must raise ValueError."""
        with pytest.raises(ValidationError, match="OIDC configuration is incomplete"):
            OidcAuthSettings(OIDC_AUTH_ENABLED=True)

    @pytest.mark.parametrize(
        "field_to_remove",
        [
            "OIDC_CONFIGURATION_ENDPOINT",
            "OIDC_CLIENT_ID",
            "OIDC_ISSUER",
            "OIDC_USERNAME_CLAIM",
            "ADMIN_ROLES_CLAIM",
            "ADMIN_ROLES_VALUES",
        ],
    )
    def test_missing_single_required_field_raises(self, field_to_remove):
        """Removing any single required field while OIDC is enabled must raise."""
        env = {**_FULL_OIDC_ENV}
        env.pop(field_to_remove)
        with pytest.raises(ValidationError, match=field_to_remove.upper()):
            OidcAuthSettings(**env)

    def test_missing_scope_fields_when_scope_validation_enabled(self):
        """Scope fields are required when scope claim validation is enabled."""
        env = {**_FULL_OIDC_ENV}
        env.pop("ADMIN_SCOPE_CLAIM")
        env.pop("ADMIN_SCOPE_VALUE")
        with pytest.raises(ValidationError, match="OIDC scope configuration is incomplete"):
            OidcAuthSettings(**env)

    def test_scope_fields_not_required_when_scope_validation_disabled(self):
        """Scope fields should not be required when scope validation is disabled."""
        settings = _make_settings(
            ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED="false",
            ADMIN_SCOPE_CLAIM=None,
            ADMIN_SCOPE_VALUE=None,
        )
        assert settings.admin_scope_claim_validation_enabled is False


# ---------------------------------------------------------------------------
# Validation: OIDC enabled – all fields provided
# ---------------------------------------------------------------------------


class TestOidcEnabledFullConfig:
    def test_full_config_succeeds(self):
        """All required fields provided should create a valid settings instance."""
        settings = _make_settings()
        assert settings.oidc_auth_enabled is True
        assert settings.oidc_configuration_endpoint == (
            "https://login.example.com/.well-known/openid-configuration"
        )
        assert settings.oidc_client_id == "my-client-id"
        assert settings.oidc_issuer == "https://login.example.com"
        assert settings.oidc_username_claim == "preferred_username"
        assert settings.admin_roles_claim == "roles"
        assert settings.admin_roles_values == "admin,superadmin"
        assert settings.admin_scope_claim == "scp"
        assert settings.admin_scope_value == "admin_access"

    def test_default_audit_user_id_claim(self):
        """Default audit user id claim should be 'oid,sub'."""
        settings = _make_settings()
        assert settings.oidc_audit_user_id_claim == "oid,sub"

    def test_default_audit_performed_by_name_claim(self):
        """Default audit performed_by_name claim should be 'unique_name,email'."""
        settings = _make_settings()
        assert settings.oidc_audit_performed_by_name_claim == "unique_name,email"


# ---------------------------------------------------------------------------
# _parse_audit_claims static method
# ---------------------------------------------------------------------------


class TestParseAuditClaims:
    @pytest.mark.parametrize(
        "input_value, expected",
        [
            ("oid,sub", ["oid", "sub"]),
            ("oid", ["oid"]),
            ("oid, sub, email", ["oid", "sub", "email"]),
            ("  oid  ,  sub  ", ["oid", "sub"]),
            ("", []),
            (",,,", []),
            ("a,,b", ["a", "b"]),
        ],
    )
    def test_parse_audit_claims(self, input_value, expected):
        result = OidcAuthSettings._parse_audit_claims(input_value)
        assert result == expected


# ---------------------------------------------------------------------------
# Properties that use _parse_audit_claims
# ---------------------------------------------------------------------------


class TestAuditClaimProperties:
    def test_oidc_audit_user_id_claims_property(self):
        """Property should parse the default 'oid,sub' into a list."""
        settings = _make_settings()
        assert settings.oidc_audit_user_id_claims == ["oid", "sub"]

    def test_oidc_audit_performed_by_name_claims_property(self):
        """Property should parse the default 'unique_name,email' into a list."""
        settings = _make_settings()
        assert settings.oidc_audit_performed_by_name_claims == ["unique_name", "email"]

    def test_custom_audit_claims(self):
        """Custom audit claim values should be parsed correctly."""
        settings = _make_settings(
            OIDC_AUDIT_USER_ID_CLAIM="custom_id",
            OIDC_AUDIT_PERFORMED_BY_NAME_CLAIM="display_name, upn, email",
        )
        assert settings.oidc_audit_user_id_claims == ["custom_id"]
        assert settings.oidc_audit_performed_by_name_claims == [
            "display_name",
            "upn",
            "email",
        ]
