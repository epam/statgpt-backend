from functools import cached_property

from aidial_sdk.chat_completion import Request

from statgpt.app.security.exceptions import InsufficientRoleError, MissingApiKeyError
from statgpt.app.settings.dial_app import DialAuthMode, dial_app_settings
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils import dial_core_factory


def _resolve_api_key(request: Request) -> str:
    """Resolve API key based on the configured authentication mode."""
    if dial_app_settings.dial_auth_mode == DialAuthMode.USER_TOKEN:
        if request.api_key is None:
            raise MissingApiKeyError()
        return request.api_key
    elif dial_app_settings.dial_auth_mode == DialAuthMode.API_KEY:
        return dial_settings.api_key.get_secret_value()
    else:
        raise ValueError(f"Unsupported DIAL auth mode: {dial_app_settings.dial_auth_mode}")


class UserAuthContext(AuthContext):
    _request: Request

    def __init__(self, request: Request):
        self._request = request

    @cached_property
    def api_key(self) -> str:
        return _resolve_api_key(self._request)

    @property
    def is_system(self) -> bool:
        return False

    @property
    def dial_access_token(self) -> str | None:
        token = self._request.jwt
        if token is not None and token.startswith("Bearer "):
            token = token[7:]
        return token


class SystemUserAuthContext(AuthContext):
    """
    Authentication context for system users.

    Used when no JWT is present but the user has a role that is allowed
    to access channels requiring JWT forwarding.
    """

    def __init__(self, request: Request):
        self._request = request

    @cached_property
    def api_key(self) -> str:
        return _resolve_api_key(self._request)

    @property
    def is_system(self) -> bool:
        return True

    @property
    def dial_access_token(self) -> str | None:
        return None


async def create_auth_context(request: Request, bearer_token_required: bool = False) -> AuthContext:
    """Create authentication context based on request and channel requirements."""
    if request.jwt is not None:
        return UserAuthContext(request)

    if bearer_token_required:
        allowed_roles = dial_app_settings.system_user_context_roles_set
        if allowed_roles and await _check_roles(request, allowed_roles):
            return SystemUserAuthContext(request)
        raise InsufficientRoleError()

    return UserAuthContext(request)


async def _check_roles(request: Request, allowed_roles: set[str]) -> bool:
    """Check if the request has any of the specified roles."""
    async with dial_core_factory(base_url=dial_settings.url, api_key=request.api_key) as dial_core:
        response = await dial_core.get_user_info()
        user_roles = set(response.get("roles", []))
        return bool(user_roles & allowed_roles)
