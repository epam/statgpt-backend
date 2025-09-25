import typing as t
from dataclasses import dataclass

import jwt
import requests
from aidial_sdk.exceptions import InvalidRequestError

from admin_portal.settings.oidc_auth import oidc_auth_settings


@dataclass
class SigningInfo:
    key: str
    algorithms: list[str]


class TokenPayload:
    def __init__(self, payload: dict):
        self._payload = payload

    @property
    def raw(self) -> dict:
        return self._payload

    @property
    def username(self):
        username = self._payload.get(oidc_auth_settings.oidc_username_claim, None)
        if not username:
            raise InvalidRequestError(
                f"Username claim {oidc_auth_settings.oidc_username_claim} not found in token"
            )
        return username


class Jwks:
    def __init__(self, openid_configuration_endpoint: str):
        self.openid_configuration_endpoint = openid_configuration_endpoint

        self._oidc_config: dict | None = None
        self._pyjwk_client: jwt.PyJWKClient | None = None

    def _fetch_oidc_config(self) -> dict:
        response = requests.get(self.openid_configuration_endpoint, timeout=20)  # ToDo: make async
        response.raise_for_status()
        oidc_config = response.json()
        return oidc_config

    @property
    def oidc_config(self) -> dict:
        if not self._oidc_config:
            self._oidc_config = self._fetch_oidc_config()
        return self._oidc_config

    @property
    def _jwk_client(self) -> jwt.PyJWKClient:
        if not self._pyjwk_client:
            self._pyjwk_client = jwt.PyJWKClient(self.oidc_config["jwks_uri"])
        return self._pyjwk_client

    def fetch_signing_info(self, token: str) -> SigningInfo:
        signing_algos = self.oidc_config["id_token_signing_alg_values_supported"]
        signing_key = self._jwk_client.get_signing_key_from_jwt(token)
        return SigningInfo(signing_key.key, signing_algos)


class TokenValidationError(BaseException):
    pass


class TokenPayloadValidator:
    def validate(self, token_payload: dict):
        """
        @raise TokenValidationError: if token validation failed
        """
        raise NotImplementedError("not implemented")


class AdminGroupClaimValidationError(TokenValidationError):
    def __init__(self, reason: str):
        self.reason = reason

    def __str__(self):
        return f"group claim validation failed: {self.reason}"


class AdminGroupsClaimValidator(TokenPayloadValidator):
    def __init__(self, admin_group_claim: str, admin_groups_values: t.Iterable[str]):
        self.admin_group_claim = admin_group_claim
        self.admin_groups_values = admin_groups_values

    def validate(self, token_payload: dict):
        token_admin_groups = token_payload.get(self.admin_group_claim, None)
        if token_admin_groups is None:
            raise AdminGroupClaimValidationError("missing group claim")
        if isinstance(token_admin_groups, str):
            # make set from one string value
            token_admin_groups = {token_admin_groups}
        else:
            # assume it is a list or set
            token_admin_groups = set(token_admin_groups)
        if not token_admin_groups.intersection(set(self.admin_groups_values)):
            raise AdminGroupClaimValidationError("missing admin group")


class ScopeClaimValidationError(TokenValidationError):
    def __init__(self, reason: str):
        self.reason = reason

    def __str__(self):
        return f"scope claim validation failed: {self.reason}"


class ScopeClaimValidator(TokenPayloadValidator):
    def __init__(self, scope_claim: str, required_value: str):
        self.scope_claim = scope_claim
        self.required_value = required_value

    def validate(self, token_payload: dict):
        if self.scope_claim not in token_payload.keys():
            raise ScopeClaimValidationError("missing scope claim")
        if self.required_value not in token_payload.get(self.scope_claim, ""):
            raise ScopeClaimValidationError("missing required scope")


class TokenValidator(TokenPayloadValidator):
    def __init__(self, validators: t.Iterable[TokenPayloadValidator]):
        self.validators = validators

    def validate(self, token_payload: dict):
        for validator in self.validators:
            validator.validate(token_payload)

    @classmethod
    def from_config(cls) -> "TokenValidator":
        if (
            oidc_auth_settings.admin_roles_values is None
            or oidc_auth_settings.admin_roles_values.strip() == ""
        ):
            raise ValueError("ADMIN_ROLES_VALUES must be set in OidcAuthConfig")
        if (
            oidc_auth_settings.admin_roles_claim is None
            or oidc_auth_settings.admin_roles_claim.strip() == ""
        ):
            raise ValueError("ADMIN_ROLES_CLAIM must be set in OidcAuthConfig")
        admin_roles_values: list[str] = oidc_auth_settings.admin_roles_values.split(",")
        admin_roles_claim: str = oidc_auth_settings.admin_roles_claim
        validators: list[TokenPayloadValidator] = []
        role_validator = AdminGroupsClaimValidator(admin_roles_claim, admin_roles_values)
        validators.append(role_validator)
        if oidc_auth_settings.admin_scope_claim_validation_enabled:
            if (
                oidc_auth_settings.admin_scope_value is None
                or oidc_auth_settings.admin_scope_value.strip() == ""
            ):
                raise ValueError(
                    "ADMIN_SCOPE_VALUE must be set in OidcAuthConfig if ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED is True"
                )
            if (
                oidc_auth_settings.admin_scope_claim is None
                or oidc_auth_settings.admin_scope_claim.strip() == ""
            ):
                raise ValueError(
                    "ADMIN_SCOPE_CLAIM must be set in OidcAuthConfig if ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED is True"
                )
            admin_scope_claim: str = oidc_auth_settings.admin_scope_claim
            admin_scope_value: str = oidc_auth_settings.admin_scope_value
            scope_validator = ScopeClaimValidator(admin_scope_claim, admin_scope_value)
            validators.append(scope_validator)
        return cls(validators=validators)


class JwtTokenVerifier:
    def __init__(
        self,
        jwks: Jwks,
        issuer: t.Optional[str] = None,
        audience: t.Optional[str] = None,
    ):
        self.jwks = jwks
        self.issuer = issuer
        self.audience = audience

    def verify(self, token: str) -> TokenPayload:
        signing_info = self.jwks.fetch_signing_info(token)
        data = jwt.decode(
            token,
            key=signing_info.key,
            algorithms=signing_info.algorithms,
            audience=self.audience,
            issuer=self.issuer,
        )

        return TokenPayload(data)

    @classmethod
    def create(cls):
        return cls(
            Jwks(oidc_auth_settings.oidc_configuration_endpoint),
            issuer=oidc_auth_settings.oidc_issuer,
            audience=oidc_auth_settings.oidc_client_id,
        )
