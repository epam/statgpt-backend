import logging
from abc import ABC, abstractmethod
from typing import Any

import jwt
from pydantic import TypeAdapter, ValidationError

_log = logging.getLogger(__name__)

bool_validator = TypeAdapter(bool)


class AuthContext(ABC):
    """Authentication context for data access."""

    @property
    @abstractmethod
    def is_system(self) -> bool:
        """Indicates if the context is for a system user."""

    @property
    @abstractmethod
    def dial_access_token(self) -> str | None:
        pass

    @property
    @abstractmethod
    def api_key(self) -> str:
        """DIAL API key for the request."""

    def get_token_claims(self) -> dict[str, Any] | None:
        """Decode the caller's DIAL access token (JWT) and return its claims.

        The token is already validated by DIAL Core upstream, so the signature is not
        re-verified here — only the claim payload is read. Returns ``None`` when no token
        is present or it cannot be decoded, so callers can fail closed.
        """
        token = self.dial_access_token
        if not token:
            return None
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except jwt.PyJWTError as e:
            _log.warning(f"Failed to decode DIAL access token claims: {e}")
            return None

    def has_claim_value(self, claim: str, value: str | None = None) -> bool:
        """Whether the caller's token satisfies the ``claim`` gate.

        - ``value`` is ``None``: the ``claim`` must parse to a truthy boolean, so string
          claims like ``"false"``/``"0"`` are correctly treated as falsy.
        - ``value`` is set: the token's claim must equal it (scalar claim) or contain it
          (list-valued claim such as ``roles``, which may carry many values).

        Comparison is done on string form so YAML/env-configured values match numeric or
        boolean claims. Fails closed: a missing/undecodable token, an absent claim, a
        non-boolean-coercible claim, or a value mismatch all return False.
        """
        claims = self.get_token_claims()
        if claims is None or claim not in claims:
            return False
        actual = claims[claim]
        if value is None:
            try:
                return bool_validator.validate_python(actual)
            except ValidationError:
                return False
        if isinstance(actual, (list, tuple, set)):
            return any(value == str(item) for item in actual)
        return value == str(actual)
