import os

from common.utils import str2bool


class OidcAuthConfig:
    ENABLED = str2bool(os.getenv("OIDC_AUTH_ENABLED", "true"))
    CONFIGURATION_ENDPOINT = os.getenv("OIDC_CONFIGURATION_ENDPOINT")
    CLIENT_ID = os.getenv("OIDC_CLIENT_ID")
    ISSUER = os.getenv("OIDC_ISSUER")
    USERNAME_CLAIM = os.getenv("OIDC_USERNAME_CLAIM")
    ADMIN_ROLES_CLAIM = os.getenv("ADMIN_ROLES_CLAIM")
    ADMIN_ROLES_VALUES = os.getenv("ADMIN_ROLES_VALUES")
    ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED = str2bool(
        os.getenv("ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED", 'true')
    )
    ADMIN_SCOPE_CLAIM = os.getenv("ADMIN_SCOPE_CLAIM")
    ADMIN_SCOPE_VALUE = os.getenv("ADMIN_SCOPE_VALUE")


if OidcAuthConfig.ENABLED:
    if not all(
        [
            OidcAuthConfig.CONFIGURATION_ENDPOINT,
            OidcAuthConfig.CLIENT_ID,
            OidcAuthConfig.ISSUER,
            OidcAuthConfig.USERNAME_CLAIM,
            OidcAuthConfig.ADMIN_ROLES_CLAIM,
            OidcAuthConfig.ADMIN_ROLES_VALUES,
        ]
    ):
        raise ValueError(
            "OIDC configuration is incomplete, one of the required environment variables is missing: "
            "OIDC_CONFIGURATION_ENDPOINT, OIDC_CLIENT_ID, OIDC_ISSUER, OIDC_USERNAME_CLAIM, ADMIN_ROLES_CLAIM, "
            "ADMIN_ROLES_VALUES",
        )

    if OidcAuthConfig.ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED:
        if not all([OidcAuthConfig.ADMIN_SCOPE_CLAIM, OidcAuthConfig.ADMIN_SCOPE_VALUE]):
            raise ValueError(
                "OIDC configuration is incomplete, one of the required environment variables is missing: "
                "ADMIN_SCOPE_CLAIM, ADMIN_SCOPE_VALUE",
            )
