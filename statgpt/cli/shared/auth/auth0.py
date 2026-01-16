"""Auth0 authentication provider using Authorization Code Flow with PKCE."""

import secrets
import webbrowser
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import httpx

from statgpt.cli.shared.auth.base import (
    AuthConfigError,
    AuthenticationError,
    AuthProvider,
    AuthResult,
)
from statgpt.cli.shared.auth.oauth import (
    OAuthCallbackHandler,
    generate_code_challenge,
    generate_code_verifier,
    get_callback_port,
    run_oauth_callback_server,
)

if TYPE_CHECKING:
    from statgpt.cli.settings import CLISettings

DEFAULT_EXPIRES_IN = 3600
DEFAULT_SCOPE = "openid profile email offline_access"


class Auth0Provider(AuthProvider):
    """Authentication provider for Auth0.

    Supports two authentication flows:

    1. Interactive login (for developers):
       - Authorization Code Flow with PKCE
       - Public client (no client_secret required)
       - Opens browser for user authentication

    2. System user login (for CI/CD - Machine-to-Machine):
       - Client Credentials Grant
       - Requires client_secret
       - No browser interaction

    Required settings for interactive login:
        - auth_auth0_domain
        - auth_auth0_client_id
        - auth_auth0_audience

    Additional settings for M2M (system_user) login:
        - auth_auth0_client_secret

    Optional:
        - auth_auth0_scope (default: 'openid profile email offline_access')
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

        if interactive:
            if not settings.auth_callback_port:
                raise AuthConfigError(self.name, ["STATGPT_CLI_AUTH_CALLBACK_PORT"])
        else:
            if not settings.auth_auth0_client_secret:
                raise AuthConfigError(self.name, ["STATGPT_CLI_AUTH_AUTH0_CLIENT_SECRET"])

    def _get_base_url(self, settings: "CLISettings") -> str:
        """Get the Auth0 base URL with https scheme."""
        domain = settings.auth_auth0_domain or ""
        if not domain.startswith("https://"):
            domain = f"https://{domain}"
        return domain

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
        return f"{self._get_base_url(settings)}/authorize?{urlencode(params)}"

    def _exchange_code_for_token(
        self,
        settings: "CLISettings",
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> AuthResult:
        """Exchange authorization code for tokens."""
        token_data = {
            "grant_type": "authorization_code",
            "client_id": settings.auth_auth0_client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }
        response = httpx.post(
            f"{self._get_base_url(settings)}/oauth/token",
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return self._parse_token_response(response)

    def interactive_login(self, settings: "CLISettings") -> AuthResult:
        """Perform interactive Auth0 login via browser using PKCE."""
        self.validate_config(settings, interactive=True)

        code_verifier = generate_code_verifier()
        code_challenge = generate_code_challenge(code_verifier)
        state = secrets.token_urlsafe(32)

        port = get_callback_port(settings)
        redirect_uri = f"http://localhost:{port}/callback"

        scope = settings.auth_auth0_scope or DEFAULT_SCOPE
        auth_url = self._build_auth_url(
            settings=settings,
            redirect_uri=redirect_uri,
            scope=scope,
            state=state,
            code_challenge=code_challenge,
        )

        OAuthCallbackHandler.reset()
        webbrowser.open(auth_url)
        run_oauth_callback_server(port)

        if OAuthCallbackHandler.error:
            raise AuthenticationError(f"Auth0 authentication failed: {OAuthCallbackHandler.error}")

        if not OAuthCallbackHandler.authorization_code:
            raise AuthenticationError(
                "Auth0 authentication failed: No authorization code received (timeout?)"
            )

        if OAuthCallbackHandler.state_received != state:
            raise AuthenticationError(
                "Auth0 authentication failed: State mismatch (possible CSRF attack)"
            )

        return self._exchange_code_for_token(
            settings=settings,
            code=OAuthCallbackHandler.authorization_code,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
        )

    def system_user_login(self, settings: "CLISettings") -> AuthResult:
        """Perform Auth0 M2M login using Client Credentials Grant."""
        self.validate_config(settings, interactive=False)

        token_data = {
            "grant_type": "client_credentials",
            "client_id": settings.auth_auth0_client_id,
            "client_secret": settings.auth_auth0_client_secret,
            "audience": settings.auth_auth0_audience,
        }
        response = httpx.post(
            f"{self._get_base_url(settings)}/oauth/token",
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
        response = httpx.post(
            f"{self._get_base_url(settings)}/oauth/token",
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return self._parse_token_response(response)
