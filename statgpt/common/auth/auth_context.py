import logging
from abc import ABC, abstractmethod
from typing import Any

import jwt

_log = logging.getLogger(__name__)


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

    def has_truthy_claim(self, claim: str) -> bool:
        """Whether ``claim`` is present and truthy in the caller's token.

        Fails closed: a missing/undecodable token or an absent/falsy claim returns False.
        """
        claims = self.get_token_claims()
        if claims is None:
            return False
        return bool(claims.get(claim))
