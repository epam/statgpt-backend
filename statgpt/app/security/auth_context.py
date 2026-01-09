from functools import cached_property
from typing import Protocol

from statgpt.app.settings.dial_app import DialAuthMode, dial_app_settings
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils import dial_core_factory


class RequestProtocol(Protocol):
    @property
    def api_key(self) -> str | None: ...
    @property
    def bearer_token(self) -> str | None: ...


class UserAuthContext(AuthContext):
    _request: RequestProtocol

    def __init__(self, request: RequestProtocol):
        self._request = request

    @cached_property
    def api_key(self) -> str:
        if dial_app_settings.dial_auth_mode == DialAuthMode.USER_TOKEN:
            if self._request.api_key is None:
                raise ValueError("API key is not provided in the `request`.")
            else:
                return self._request.api_key
        elif dial_app_settings.dial_auth_mode == DialAuthMode.API_KEY:
            return dial_settings.api_key.get_secret_value()
        else:
            raise ValueError(f"Unsupported DIAL auth mode: {dial_app_settings.dial_auth_mode}")

    @property
    def is_system(self) -> bool:
        return False

    @property
    def dial_access_token(self) -> str | None:
        return self._request.bearer_token


class EvalAuthContext(AuthContext):
    """Authentication context for evaluation"""

    def __init__(self, request: RequestProtocol):
        self._request = request

    @property
    def api_key(self) -> str:
        if dial_app_settings.dial_auth_mode == DialAuthMode.USER_TOKEN:
            if self._request.api_key is None:
                raise ValueError("API key is not provided in the `request`.")
            else:
                return self._request.api_key
        elif dial_app_settings.dial_auth_mode == DialAuthMode.API_KEY:
            return dial_settings.api_key.get_secret_value()
        else:
            raise ValueError(f"Unsupported DIAL auth mode: {dial_app_settings.dial_auth_mode}")

    @property
    def is_system(self) -> bool:
        # TODO: We need to implement a proper check for evaluation context and make this property return False
        return True

    @property
    def dial_access_token(self) -> str | None:
        return None


async def create_auth_context(request: RequestProtocol) -> AuthContext:
    """Create an authentication context based on the request."""

    if request.bearer_token is not None:
        return UserAuthContext(request)

    if role := dial_app_settings.eval_dial_role:
        if await _check_role(request, role):
            return EvalAuthContext(request)

    raise ValueError("Request does not contain a valid JWT token for user authentication.")


async def _check_role(request: RequestProtocol, role: str) -> bool:
    """Check if the request has the specified role."""

    if request.api_key is None:
        return False

    async with dial_core_factory(base_url=dial_settings.url, api_key=request.api_key) as dial_core:
        response = await dial_core.get_user_info()
        return role in response.get("roles", [])
