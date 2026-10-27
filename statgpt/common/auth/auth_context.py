import logging
from abc import ABC, abstractmethod

from aidial_client import DialException

from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils.dial import dial_client_factory

_log = logging.getLogger(__name__)


class AuthContext(ABC):
    """Authentication context for data access."""

    # Request-scoped cache of the roles DIAL resolves for the caller. `get_roles` can be called
    # several times per request (e.g. Deep Research availability is checked while building the
    # configuration, routing the agent, and emitting the toggle form schema), and each miss is a
    # DIAL user-info round-trip, so the result is memoized on the instance. An auth context is
    # built per request and pins a single token, so the cache never goes stale within its lifetime.
    # `None` means "not resolved yet"; an empty list is a valid (fail-closed) resolved value. Kept
    # as a class-level default so subclasses need not call `super().__init__()`.
    _roles_cache: list[str] | None = None

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
        """Authorization roles DIAL resolves for the caller, memoized for the request.

        Sends the caller's access token to DIAL's user-info endpoint and returns the roles it
        reports, rather than decoding the JWT locally. Fails closed: without an access token, or
        when the lookup fails, an empty list is returned so callers deny access by default. The
        result (including a fail-closed empty list) is cached on the instance, so repeated checks
        within the same request reuse the first lookup instead of re-querying DIAL.
        """
        if self._roles_cache is None:
            self._roles_cache = await self._resolve_roles()
        return self._roles_cache

    async def _resolve_roles(self) -> list[str]:
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
