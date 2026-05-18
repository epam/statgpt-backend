from functools import cached_property

from statgpt.app.security.exceptions import InsufficientRoleError, MissingApiKeyError
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils import dial_core_factory

from .credentials import DialAuthCredentialsI


def _resolve_api_key(request: DialAuthCredentialsI) -> str:
    """Resolve API key from the request."""
    if request.api_key is None:
        raise MissingApiKeyError()
    return request.api_key


class UserAuthContext(AuthContext):
    _request: DialAuthCredentialsI

    def __init__(self, request: DialAuthCredentialsI):
        self._request = request

    @cached_property
    def api_key(self) -> str:
        return _resolve_api_key(self._request)

    @property
    def is_system(self) -> bool:
        return False

    @property
    def dial_access_token(self) -> str | None:
        return self._request.bearer_token


class SystemUserAuthContext(AuthContext):
    """
    Authentication context for system users.

    Used when no JWT is present but the user has a role that is allowed
    to access channels requiring JWT forwarding.
    """

    def __init__(self, request: DialAuthCredentialsI):
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


async def create_auth_context(
    request: DialAuthCredentialsI, bearer_token_required: bool = False
) -> AuthContext:
    """Create an authentication context based on the request."""

    if request.bearer_token is not None:
        return UserAuthContext(request)

    if bearer_token_required:
        allowed_roles = dial_app_settings.system_user_context_roles_set
        if allowed_roles and await _check_roles(request, allowed_roles):
            return SystemUserAuthContext(request)
        raise InsufficientRoleError()

    return UserAuthContext(request)


async def _check_roles(request: DialAuthCredentialsI, allowed_roles: set[str]) -> bool:
    """Check if the request has the specified role."""

    if request.api_key is None:
        return False

    async with dial_core_factory(base_url=dial_settings.url, api_key=request.api_key) as dial_core:
        response = await dial_core.get_user_info()
        user_roles = set(response.get("roles", []))
        return bool(user_roles & allowed_roles)
