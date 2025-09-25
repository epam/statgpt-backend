from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OidcAuthSettings(BaseSettings):
    """
    Settings for OIDC authentication
    """

    model_config = SettingsConfigDict(env_prefix="")

    # Main OIDC settings
    oidc_auth_enabled: bool = Field(
        default=True,
        alias="OIDC_AUTH_ENABLED",
        description="If enabled, all admin endpoints require OIDC authentication",
    )

    oidc_configuration_endpoint: Optional[str] = Field(
        default=None, alias="OIDC_CONFIGURATION_ENDPOINT", description="OIDC Configuration Endpoint"
    )

    oidc_client_id: Optional[str] = Field(
        default=None, alias="OIDC_CLIENT_ID", description="OIDC Client ID"
    )

    oidc_issuer: Optional[str] = Field(default=None, alias="OIDC_ISSUER", description="OIDC Issuer")

    oidc_username_claim: Optional[str] = Field(
        default=None, alias="OIDC_USERNAME_CLAIM", description="OIDC Username Claim"
    )

    # Admin roles settings
    admin_roles_claim: Optional[str] = Field(
        default=None, alias="ADMIN_ROLES_CLAIM", description="OIDC Admin Roles Claim"
    )

    admin_roles_values: Optional[str] = Field(
        default=None,
        alias="ADMIN_ROLES_VALUES",
        description="OIDC Admin Roles Values (comma-separated)",
    )

    # Admin scope settings
    admin_scope_claim_validation_enabled: bool = Field(
        default=True,
        alias="ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED",
        description="If enabled, the admin portal will check for scopes in the OIDC token",
    )

    admin_scope_claim: Optional[str] = Field(
        default=None,
        alias="ADMIN_SCOPE_CLAIM",
        description="The name of the custom access token field that contains scope information",
    )

    admin_scope_value: Optional[str] = Field(
        default=None,
        alias="ADMIN_SCOPE_VALUE",
        description="Required scope claim value to get access to admin portal",
    )

    @model_validator(mode="after")
    def validate_required_fields(self):
        """Validate that required fields are present when OIDC is enabled"""
        if self.oidc_auth_enabled:
            required_fields = {
                "oidc_configuration_endpoint": self.oidc_configuration_endpoint,
                "oidc_client_id": self.oidc_client_id,
                "oidc_issuer": self.oidc_issuer,
                "oidc_username_claim": self.oidc_username_claim,
                "admin_roles_claim": self.admin_roles_claim,
                "admin_roles_values": self.admin_roles_values,
            }

            missing_fields = [name for name, value in required_fields.items() if value is None]
            if missing_fields:
                raise ValueError(
                    f"OIDC configuration is incomplete, missing required environment variables: "
                    f"{', '.join(f'{name.upper()}' for name in missing_fields)}"
                )

            if self.admin_scope_claim_validation_enabled:
                scope_fields = {
                    "admin_scope_claim": self.admin_scope_claim,
                    "admin_scope_value": self.admin_scope_value,
                }
                missing_scope_fields = [
                    name for name, value in scope_fields.items() if value is None
                ]
                if missing_scope_fields:
                    raise ValueError(
                        f"OIDC scope configuration is incomplete, missing required environment variables: "
                        f"{', '.join(f'{name.upper()}' for name in missing_scope_fields)}"
                    )

        return self


# Create a singleton instance
oidc_auth_settings = OidcAuthSettings()
