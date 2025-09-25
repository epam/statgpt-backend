import time

from common.auth.token import TokenResponseI


class OAuthTokenResponse(TokenResponseI):
    def __init__(self, access_token: str, expires_in: int, refresh_token: str | None):
        self._access_token = access_token
        self._expires_at = int(time.time()) + expires_in
        self._refresh_token = refresh_token

    @property
    def access_token(self) -> str:
        return self._access_token

    @property
    def expires_at(self) -> int:
        return self._expires_at

    @property
    def refresh_token(self) -> str | None:
        return self._refresh_token
