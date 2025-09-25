from functools import cached_property

from aidial_sdk.chat_completion import Request

from common.auth.auth_context import AuthContext
from common.settings.dial import dial_settings
from common.utils import dial_core_factory
from statgpt.settings.dial_app import DialAuthMode, dial_app_settings


class UserAuthContext(AuthContext):
    _request: Request

    def __init__(self, request: Request):
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
        token = self._request.jwt
        if token is not None and token.startswith("Bearer "):
            token = token[7:]
        return token


class EvalAuthContext(AuthContext):
    """Authentication context for evaluation"""

    def __init__(self, request: Request):
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


async def create_auth_context(request: Request) -> AuthContext:
    """Create an authentication context based on the request."""

    if request.jwt is not None:
        return UserAuthContext(request)

    if role := dial_app_settings.eval_dial_role:
        if await _check_role(request, role):
            return EvalAuthContext(request)

    raise ValueError("Request does not contain a valid JWT token for user authentication.")


async def _check_role(request: Request, role: str) -> bool:
    """Check if the request has the specified role."""

    async with dial_core_factory(base_url=dial_settings.url, api_key=request.api_key) as dial_core:
        response = await dial_core.get_user_info()
        return role in response.get("roles", [])
