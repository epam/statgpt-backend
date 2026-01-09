from aidial_sdk.exceptions import HTTPException as DIALException


class AuthenticationError(DIALException):
    """Base class for authentication errors (401)."""

    def __init__(self, message: str, code: str = "unauthorized"):
        super().__init__(status_code=401, code=code, message=message)


class AuthorizationError(DIALException):
    """Base class for authorization errors (403)."""

    def __init__(self, message: str, code: str = "forbidden"):
        super().__init__(status_code=403, code=code, message=message)


class MissingApiKeyError(AuthenticationError):
    """Raised when API key is not provided in the request."""

    def __init__(self):
        super().__init__(
            message="API key is not provided in the request.",
            code="missing_api_key",
        )


class InsufficientRoleError(AuthorizationError):
    """Raised when user lacks required role for system context access."""

    def __init__(self):
        super().__init__(
            message="User does not have a role that allows system user context access.",
            code="insufficient_role",
        )
