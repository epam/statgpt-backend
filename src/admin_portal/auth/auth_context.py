from common.auth.auth_context import AuthContext
from common.config import DialConfig


class SystemUserAuthContext(AuthContext):
    """All requests in the admin portal are executed on behalf of the system user."""

    @property
    def is_system(self) -> bool:
        return True

    @property
    def dial_access_token(self) -> None:
        return None

    @property
    def api_key(self) -> str:
        return DialConfig.get_api_key().get_secret_value()
