from typing import Protocol, Self

from starlette.datastructures import Headers


class DialAuthCredentialsI(Protocol):
    """Credentials passed by DIAL Core with each proxied request."""

    @property
    def api_key(self) -> str | None: ...

    @property
    def bearer_token(self) -> str | None: ...


class DialAuthCredentials(DialAuthCredentialsI):
    """Simple DIAL auth credentials container."""

    def __init__(self, api_key: str | None, bearer_token: str | None):
        self._api_key = api_key
        self._bearer_token = bearer_token

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def bearer_token(self) -> str | None:
        return self._bearer_token

    @classmethod
    def from_headers(cls, headers: Headers) -> Self:
        """Extract DIAL auth credentials from HTTP headers.

        Use when ``create_auth_context`` is needed outside the DIAL SDK request
        lifecycle (plain FastAPI endpoints, MCP handlers, etc.).
        """
        api_key = headers.get("api-key") or headers.get("x-api-key")
        token = headers.get("authorization")
        bearer_token = (
            token[7:] if token is not None and token.lower().startswith("bearer ") else None
        )
        return cls(api_key=api_key, bearer_token=bearer_token)
