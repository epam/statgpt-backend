"""Shared OAuth utilities for PKCE flow and callback handling."""

import base64
import hashlib
import secrets
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from statgpt.cli.settings import CLISettings

CALLBACK_TIMEOUT = 300


def generate_code_verifier(length: int = 64) -> str:
    """Generate a cryptographically random code verifier for PKCE."""
    random_bytes = secrets.token_bytes(length)
    return base64.urlsafe_b64encode(random_bytes).rstrip(b"=").decode("ascii")


def generate_code_challenge(verifier: str) -> str:
    """Generate S256 code challenge from verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def find_available_port() -> int:
    """Find an available port for the callback server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_callback_port(settings: "CLISettings") -> int:
    """Get callback port - use configured port or find available one.

    Use configured port when the OAuth provider requires exact redirect URI matching
    (e.g., Auth0). Falls back to dynamic port allocation for providers that allow
    wildcard ports (e.g., Keycloak).
    """
    if settings.auth_callback_port:
        return settings.auth_callback_port
    return find_available_port()


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    authorization_code: str | None = None
    error: str | None = None
    state_received: str | None = None

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress HTTP server logging."""
        del format, args

    def do_GET(self) -> None:
        """Handle GET request from OAuth callback."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "code" in params:
            OAuthCallbackHandler.authorization_code = params["code"][0]
            OAuthCallbackHandler.state_received = params.get("state", [None])[0]
            self._send_success_response()
        elif "error" in params:
            error_desc = params.get("error_description", params["error"])[0]
            OAuthCallbackHandler.error = error_desc
            self._send_error_response(error_desc)
        else:
            OAuthCallbackHandler.error = "Missing authorization code"
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

    @classmethod
    def reset(cls) -> None:
        """Reset handler state before starting a new OAuth flow."""
        cls.authorization_code = None
        cls.error = None
        cls.state_received = None


def run_oauth_callback_server(port: int) -> None:
    """Run the OAuth callback server and wait for a single request."""
    server = HTTPServer(("127.0.0.1", port), OAuthCallbackHandler)
    server.timeout = CALLBACK_TIMEOUT
    try:
        server.handle_request()
    finally:
        server.server_close()
