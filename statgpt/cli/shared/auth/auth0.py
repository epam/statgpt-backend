"""Auth0 authentication provider using Authorization Code Flow with PKCE."""

import base64
import hashlib
import secrets
import socket
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from statgpt.cli.shared.auth.base import (
    AuthConfigError,
    AuthenticationError,
    AuthProvider,
    AuthResult,
)

if TYPE_CHECKING:
    from statgpt.cli.settings import CLISettings

DEFAULT_EXPIRES_IN = 3600
CALLBACK_TIMEOUT = 300
DEFAULT_SCOPE = "openid profile email offline_access"


def _generate_code_verifier(length: int = 64) -> str:
    """Generate a cryptographically random code verifier for PKCE.

    Args:
        length: Number of random bytes (43-128 characters per RFC 7636)

    Returns:
        URL-safe base64-encoded random string
    """
    random_bytes = secrets.token_bytes(length)
    return base64.urlsafe_b64encode(random_bytes).rstrip(b"=").decode("ascii")


def _generate_code_challenge(verifier: str) -> str:
    """Generate S256 code challenge from verifier.

    Args:
        verifier: The code verifier string

    Returns:
        URL-safe base64-encoded SHA256 hash of verifier
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _find_available_port() -> int:
    """Find an available port for the callback server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _AuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    authorization_code: str | None = None
    error: str | None = None
    state_received: str | None = None

    def log_message(self, format: str, *args: object) -> None:
        """Suppress HTTP server logging."""

    def do_GET(self) -> None:
        """Handle GET request from OAuth callback."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "code" in params:
            _AuthCallbackHandler.authorization_code = params["code"][0]
            _AuthCallbackHandler.state_received = params.get("state", [None])[0]
            self._send_success_response()
        elif "error" in params:
            error_desc = params.get("error_description", params["error"])[0]
            _AuthCallbackHandler.error = error_desc
            self._send_error_response(error_desc)
        else:
            _AuthCallbackHandler.error = "Missing authorization code"
            self._send_error_response("Missing authorization code")

    def _send_success_response(self) -> None:
        """Send success HTML response to browser."""
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Authentication Successful</title></head>
        <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
            <h1 style="color: #28a745;">Authentication Successful</h1>
            <p>You can close this window and return to the CLI.</p>
        </body>
        </html>
        """
        self.wfile.write(html.encode())

    def _send_error_response(self, error: str) -> None:
        """Send error HTML response to browser."""
        self.send_response(400)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Authentication Failed</title></head>
        <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
            <h1 style="color: #dc3545;">Authentication Failed</h1>
            <p>{error}</p>
        </body>
        </html>
        """
        self.wfile.write(html.encode())


class Auth0Provider(AuthProvider):
    """Authentication provider for Auth0 using Authorization Code Flow with PKCE.

    Required environment variables for interactive login:
        - STATGPT_CLI_AUTH_AUTH0_DOMAIN
        - STATGPT_CLI_AUTH_AUTH0_CLIENT_ID
        - STATGPT_CLI_AUTH_AUTH0_AUDIENCE

    Additional variables for system user login:
        - STATGPT_CLI_AUTH_AUTH0_USERNAME
        - STATGPT_CLI_AUTH_AUTH0_PASSWORD

    Optional:
        - STATGPT_CLI_AUTH_AUTH0_SCOPE (default: 'openid profile email offline_access')
        - STATGPT_CLI_AUTH_AUTH0_CLIENT_SECRET (for confidential clients)
    """

    @property
    def name(self) -> str:
        return "auth0"

    def _validate_base_config(self, settings: "CLISettings") -> None:
        """Validate base Auth0 configuration required for all operations."""
        missing = []
        if not settings.auth_auth0_domain:
            missing.append("STATGPT_CLI_AUTH_AUTH0_DOMAIN")
        if not settings.auth_auth0_client_id:
            missing.append("STATGPT_CLI_AUTH_AUTH0_CLIENT_ID")
        if not settings.auth_auth0_audience:
            missing.append("STATGPT_CLI_AUTH_AUTH0_AUDIENCE")
        if missing:
            raise AuthConfigError(self.name, missing)

    def validate_config(self, settings: "CLISettings", interactive: bool) -> None:
        """Validate Auth0 configuration."""
        self._validate_base_config(settings)

        if not interactive:
            missing = []
            if not settings.auth_auth0_username:
                missing.append("STATGPT_CLI_AUTH_AUTH0_USERNAME")
            if not settings.auth_auth0_password:
                missing.append("STATGPT_CLI_AUTH_AUTH0_PASSWORD")
            if missing:
                raise AuthConfigError(self.name, missing)

    def _get_token_url(self, settings: "CLISettings") -> str:
        """Get the Auth0 token endpoint URL."""
        domain = settings.auth_auth0_domain
        if domain and not domain.startswith("https://"):
            domain = f"https://{domain}"
        return f"{domain}/oauth/token"

    def _get_authorize_url(self, settings: "CLISettings") -> str:
        """Get the Auth0 authorization endpoint URL."""
        domain = settings.auth_auth0_domain
        if domain and not domain.startswith("https://"):
            domain = f"https://{domain}"
        return f"{domain}/authorize"

    def _build_auth_url(
        self,
        settings: "CLISettings",
        redirect_uri: str,
        scope: str,
        state: str,
        code_challenge: str,
    ) -> str:
        """Build authorization URL with PKCE parameters."""
        params = {
            "client_id": settings.auth_auth0_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "audience": settings.auth_auth0_audience,
        }
        return f"{self._get_authorize_url(settings)}?{urlencode(params)}"

    def _parse_token_response(self, response: httpx.Response) -> AuthResult:
        """Parse Auth0 token response into AuthResult."""
        if response.status_code != 200:
            try:
                error_data = response.json()
                error_msg = error_data.get(
                    "error_description", error_data.get("error", "Unknown error")
                )
            except Exception:
                error_msg = response.text or f"HTTP {response.status_code}"
            raise AuthenticationError(f"Auth0 authentication failed: {error_msg}")

        data = response.json()
        if not data.get("access_token"):
            raise AuthenticationError("Auth0 authentication failed: No access token received")

        return AuthResult(
            access_token=data["access_token"],
            expires_in=data.get("expires_in", DEFAULT_EXPIRES_IN),
            refresh_token=data.get("refresh_token"),
        )

    def interactive_login(self, settings: "CLISettings") -> AuthResult:
        """Perform interactive Auth0 login via browser using PKCE."""
        self.validate_config(settings, interactive=True)

        code_verifier = _generate_code_verifier()
        code_challenge = _generate_code_challenge(code_verifier)
        state = secrets.token_urlsafe(32)

        port = _find_available_port()
        redirect_uri = f"http://localhost:{port}/callback"

        scope = settings.auth_auth0_scope or DEFAULT_SCOPE
        auth_url = self._build_auth_url(
            settings=settings,
            redirect_uri=redirect_uri,
            scope=scope,
            state=state,
            code_challenge=code_challenge,
        )

        _AuthCallbackHandler.authorization_code = None
        _AuthCallbackHandler.error = None
        _AuthCallbackHandler.state_received = None

        server = HTTPServer(("127.0.0.1", port), _AuthCallbackHandler)
        server.timeout = CALLBACK_TIMEOUT

        webbrowser.open(auth_url)

        try:
            server.handle_request()
        finally:
            server.server_close()

        if _AuthCallbackHandler.error:
            raise AuthenticationError(f"Auth0 authentication failed: {_AuthCallbackHandler.error}")

        if not _AuthCallbackHandler.authorization_code:
            raise AuthenticationError(
                "Auth0 authentication failed: No authorization code received (timeout?)"
            )

        if _AuthCallbackHandler.state_received != state:
            raise AuthenticationError(
                "Auth0 authentication failed: State mismatch (possible CSRF attack)"
            )

        token_data = {
            "grant_type": "authorization_code",
            "client_id": settings.auth_auth0_client_id,
            "code": _AuthCallbackHandler.authorization_code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }
        if settings.auth_auth0_client_secret:
            token_data["client_secret"] = settings.auth_auth0_client_secret

        response = httpx.post(
            self._get_token_url(settings),
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return self._parse_token_response(response)

    def system_user_login(self, settings: "CLISettings") -> AuthResult:
        """Perform Auth0 system user login using Resource Owner Password Grant.

        Note: Resource Owner Password Credentials grant is deprecated in OAuth 2.1
        but may still be needed for service accounts or automation.
        The Auth0 application must have "Password" grant type enabled.
        """
        self.validate_config(settings, interactive=False)

        scope = settings.auth_auth0_scope or DEFAULT_SCOPE
        token_data = {
            "grant_type": "password",
            "client_id": settings.auth_auth0_client_id,
            "username": settings.auth_auth0_username,
            "password": settings.auth_auth0_password,
            "audience": settings.auth_auth0_audience,
            "scope": scope,
        }
        if settings.auth_auth0_client_secret:
            token_data["client_secret"] = settings.auth_auth0_client_secret

        response = httpx.post(
            self._get_token_url(settings),
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return self._parse_token_response(response)

    def refresh_token(self, settings: "CLISettings", refresh_token: str) -> AuthResult:
        """Refresh an access token using a refresh token."""
        self._validate_base_config(settings)

        token_data = {
            "grant_type": "refresh_token",
            "client_id": settings.auth_auth0_client_id,
            "refresh_token": refresh_token,
        }
        if settings.auth_auth0_client_secret:
            token_data["client_secret"] = settings.auth_auth0_client_secret

        response = httpx.post(
            self._get_token_url(settings),
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return self._parse_token_response(response)
