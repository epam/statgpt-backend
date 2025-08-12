import typing as t
from dataclasses import dataclass

import jwt
import requests

from admin_portal.config.oidc_auth import OidcAuthConfig


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
        username = self._payload.get(OidcAuthConfig.USERNAME_CLAIM, None)
        assert username, f"Username claim {OidcAuthConfig.USERNAME_CLAIM} not found in token"
        return username


class Jwks:
    def __init__(self, openid_configuration_endpoint: str):
        self.openid_configuration_endpoint = openid_configuration_endpoint

        self._oidc_config: dict | None = None
        self._pyjwk_client: jwt.PyJWKClient | None = None

    def _fetch_oidc_config(self) -> dict:
        response = requests.get(self.openid_configuration_endpoint)
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
    def from_config(cls, config: type[OidcAuthConfig]) -> "TokenValidator":
        if config.ADMIN_ROLES_VALUES is None:
            raise ValueError("ADMIN_ROLES_VALUES must be set in OidcAuthConfig")
        admin_roles_values = config.ADMIN_ROLES_VALUES.split(",")
        validators = []
        role_validator = AdminGroupsClaimValidator(config.ADMIN_ROLES_CLAIM, admin_roles_values)
        validators.append(role_validator)
        if config.ADMIN_SCOPE_CLAIM_VALIDATION_ENABLED:
            scope_validator = ScopeClaimValidator(
                config.ADMIN_SCOPE_CLAIM, config.ADMIN_SCOPE_VALUE
            )
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
    def create(cls, config):
        return cls(
            Jwks(config.CONFIGURATION_ENDPOINT),
            issuer=config.ISSUER,
            audience=config.CLIENT_ID,
        )
