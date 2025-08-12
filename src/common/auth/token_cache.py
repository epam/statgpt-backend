import time

from common.auth.token import TokenResponseI
from common.config.auth import TOKEN_EXPIRATION_BUFFER


class TokenCache:
    """
    saves access token and its expiration date
    """

    def __init__(self) -> None:
        self._cache: dict[str, TokenResponseI] = dict()

    def add(self, key: str, token_info: TokenResponseI) -> None:
        self._cache[key] = token_info

    def get(self, key: str) -> TokenResponseI | None:
        return self._cache.get(key)

    def get_not_expired(self, key) -> TokenResponseI | None:
        token_info = self._cache.get(key)
        if token_info:
            expires_at = token_info.expires_at - TOKEN_EXPIRATION_BUFFER
            if time.time() < expires_at:  # not expired
                return token_info
        return None


class GlobalTokenCache:
    _instance = None

    @classmethod
    def get_or_create(cls) -> TokenCache:
        if not cls._instance:
            cls._instance = TokenCache()
        return cls._instance
