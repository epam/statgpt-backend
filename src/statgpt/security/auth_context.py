from functools import cached_property

from aidial_sdk.chat_completion import Request

from common.auth.auth_context import AuthContext
from common.config import DialConfig
from statgpt.config import DialAppConfig, DialAuthMode


class UserAuthContext(AuthContext):
    _api_key: str
    _request: Request

    def __init__(self, request: Request):
        self._request = request

    @cached_property
    def api_key(self) -> str:
        if DialAppConfig.AUTH_MODE == DialAuthMode.USER_TOKEN:
            if self._request.api_key is None:
                raise ValueError("API key is not provided in the `request`.")
            else:
                return self._request.api_key
        elif DialAppConfig.AUTH_MODE == DialAuthMode.API_KEY:
            return DialConfig.get_api_key().get_secret_value()
        else:
            raise ValueError(f"Unsupported DIAL auth mode: {DialAppConfig.AUTH_MODE}")

    @property
    def is_system(self) -> bool:
        return False

    @property
    def dial_access_token(self) -> str | None:
        token = self._request.jwt
        if token is not None and token.startswith("Bearer "):
            token = token[7:]
        return token
