"""Keycloak authentication provider using Authorization Code Flow with PKCE."""

import base64
import hashlib
import secrets
import socket
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlparse

from keycloak import KeycloakOpenID
from keycloak.exceptions import KeycloakAuthenticationError, KeycloakError

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


class KeycloakProvider(AuthProvider):
    """Authentication provider for Keycloak using Authorization Code Flow with PKCE.

    Required environment variables for interactive login:
        - STATGPT_CLI_AUTH_KEYCLOAK_SERVER_URL
        - STATGPT_CLI_AUTH_KEYCLOAK_REALM
        - STATGPT_CLI_AUTH_KEYCLOAK_CLIENT_ID

    Additional variables for system user login:
        - STATGPT_CLI_AUTH_KEYCLOAK_CLIENT_SECRET (optional for public clients)
        - STATGPT_CLI_AUTH_KEYCLOAK_USERNAME
        - STATGPT_CLI_AUTH_KEYCLOAK_PASSWORD

    Optional:
        - STATGPT_CLI_AUTH_KEYCLOAK_SCOPE (default: 'openid')
    """

    @property
    def name(self) -> str:
        return "keycloak"

    def _validate_base_config(self, settings: "CLISettings") -> None:
        """Validate base Keycloak configuration required for all operations."""
        missing = []
        if not settings.auth_keycloak_server_url:
            missing.append("STATGPT_CLI_AUTH_KEYCLOAK_SERVER_URL")
        if not settings.auth_keycloak_realm:
            missing.append("STATGPT_CLI_AUTH_KEYCLOAK_REALM")
        if not settings.auth_keycloak_client_id:
            missing.append("STATGPT_CLI_AUTH_KEYCLOAK_CLIENT_ID")
        if missing:
            raise AuthConfigError(self.name, missing)

    def validate_config(self, settings: "CLISettings", interactive: bool) -> None:
        """Validate Keycloak configuration."""
        self._validate_base_config(settings)

        if not interactive:
            missing = []
            if not settings.auth_keycloak_username:
                missing.append("STATGPT_CLI_AUTH_KEYCLOAK_USERNAME")
            if not settings.auth_keycloak_password:
                missing.append("STATGPT_CLI_AUTH_KEYCLOAK_PASSWORD")
            if missing:
                raise AuthConfigError(self.name, missing)

    def _create_keycloak_client(self, settings: "CLISettings") -> KeycloakOpenID:
        """Create KeycloakOpenID client instance."""
        return KeycloakOpenID(
            server_url=settings.auth_keycloak_server_url,
            realm_name=settings.auth_keycloak_realm,
            client_id=settings.auth_keycloak_client_id,
            client_secret_key=settings.auth_keycloak_client_secret,
        )

    def _build_auth_url(
        self,
        settings: "CLISettings",
        redirect_uri: str,
        scope: str,
        state: str,
        code_challenge: str,
    ) -> str:
        """Build authorization URL with PKCE parameters.

        KeycloakOpenID.auth_url() doesn't support PKCE, so we build it manually.
        """
        base_url = (
            f"{settings.auth_keycloak_server_url}/realms/"
            f"{settings.auth_keycloak_realm}/protocol/openid-connect/auth"
        )
        params = {
            "client_id": settings.auth_keycloak_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return f"{base_url}?{urlencode(params)}"

    def _parse_token_response(self, token_response: dict) -> AuthResult:
        """Parse Keycloak token response into AuthResult."""
        if not token_response.get("access_token"):
            raise AuthenticationError("Keycloak authentication failed: No access token received")

        return AuthResult(
            access_token=token_response["access_token"],
            expires_in=token_response.get("expires_in", DEFAULT_EXPIRES_IN),
            refresh_token=token_response.get("refresh_token"),
        )

    def interactive_login(self, settings: "CLISettings") -> AuthResult:
        """Perform interactive Keycloak login via browser using PKCE."""
        self.validate_config(settings, interactive=True)

        code_verifier = _generate_code_verifier()
        code_challenge = _generate_code_challenge(code_verifier)
        state = secrets.token_urlsafe(32)

        port = _find_available_port()
        redirect_uri = f"http://localhost:{port}/callback"

        keycloak_client = self._create_keycloak_client(settings)

        scope = settings.auth_keycloak_scope or "openid"
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
            raise AuthenticationError(
                f"Keycloak authentication failed: {_AuthCallbackHandler.error}"
            )

        if not _AuthCallbackHandler.authorization_code:
            raise AuthenticationError(
                "Keycloak authentication failed: No authorization code received (timeout?)"
            )

        if _AuthCallbackHandler.state_received != state:
            raise AuthenticationError(
                "Keycloak authentication failed: State mismatch (possible CSRF attack)"
            )

        try:
            token_response = keycloak_client.token(
                grant_type="authorization_code",
                code=_AuthCallbackHandler.authorization_code,
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
            )
            return self._parse_token_response(token_response)
        except KeycloakAuthenticationError as e:
            raise AuthenticationError(f"Keycloak token exchange failed: {e}") from e
        except KeycloakError as e:
            raise AuthenticationError(f"Keycloak error: {e}") from e

    def system_user_login(self, settings: "CLISettings") -> AuthResult:
        """Perform Keycloak system user login using Direct Grant.

        Note: Direct Grant (Resource Owner Password Credentials) is deprecated
        in OAuth 2.1 but may still be needed for service accounts or automation.
        """
        self.validate_config(settings, interactive=False)

        keycloak_client = self._create_keycloak_client(settings)
        scope = settings.auth_keycloak_scope or "openid"

        try:
            token_response = keycloak_client.token(
                username=settings.auth_keycloak_username,
                password=settings.auth_keycloak_password,
                grant_type="password",
                scope=scope,
            )
            return self._parse_token_response(token_response)
        except KeycloakAuthenticationError as e:
            raise AuthenticationError(f"Keycloak system user authentication failed: {e}") from e
        except KeycloakError as e:
            raise AuthenticationError(f"Keycloak error: {e}") from e

    def refresh_token(self, settings: "CLISettings", refresh_token: str) -> AuthResult:
        """Refresh an access token using a refresh token."""
        self._validate_base_config(settings)
        keycloak_client = self._create_keycloak_client(settings)

        try:
            token_response = keycloak_client.refresh_token(refresh_token)
            return self._parse_token_response(token_response)
        except KeycloakAuthenticationError as e:
            raise AuthenticationError(f"Keycloak token refresh failed: {e}") from e
        except KeycloakError as e:
            raise AuthenticationError(f"Keycloak error: {e}") from e
