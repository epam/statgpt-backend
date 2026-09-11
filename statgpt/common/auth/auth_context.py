import logging
from abc import ABC, abstractmethod

from aidial_client import DialException

from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils.dial import dial_client_factory

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

    async def get_roles(self) -> list[str]:
        """Authorization roles DIAL resolves for the caller.

        Sends the caller's access token to DIAL's user-info endpoint and returns the roles it
        reports, rather than decoding the JWT locally. Fails closed: without an access token, or
        when the lookup fails, an empty list is returned so callers deny access by default.
        """
        token = self.dial_access_token
        if not token:
            return []
        try:
            async with dial_client_factory(base_url=dial_settings.url, bearer_token=token) as dial:
                user_info = await dial.user.info()
                return list(user_info.roles)
        except DialException as e:
            _log.warning(f"Failed to resolve caller roles from DIAL: {e}")
            return []

    async def has_role(self, role: str) -> bool:
        """Whether DIAL reports ``role`` among the caller's roles. Fails closed."""
        return role in await self.get_roles()
