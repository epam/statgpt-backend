"""Keycloak authentication provider."""

import secrets
import webbrowser
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import httpx
from keycloak import KeycloakOpenID
from keycloak.exceptions import KeycloakAuthenticationError, KeycloakError

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


class KeycloakProvider(AuthProvider):
    """Authentication provider for Keycloak.

    Supports two authentication flows:

    1. Interactive login (for developers):
       - Authorization Code Flow with PKCE
       - Public client (no client_secret required)
       - Opens browser for user authentication

    2. System user login (for CI/CD - Machine-to-Machine):
       - Client Credentials Grant
       - Requires client_secret (confidential client with service account)
       - No browser interaction

    Required settings for interactive login:
        - auth_keycloak_server_url
        - auth_keycloak_realm
        - auth_keycloak_client_id

    Additional settings for M2M (system_user) login:
        - auth_keycloak_client_secret

    Optional:
        - auth_keycloak_scope (default: 'openid')
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
            if not settings.auth_keycloak_client_secret:
                raise AuthConfigError(self.name, ["STATGPT_CLI_AUTH_KEYCLOAK_CLIENT_SECRET"])

    def _create_keycloak_client(self, settings: "CLISettings") -> KeycloakOpenID:
        """Create KeycloakOpenID client instance."""
        return KeycloakOpenID(
            server_url=settings.auth_keycloak_server_url,
            realm_name=settings.auth_keycloak_realm,
            client_id=settings.auth_keycloak_client_id,
            client_secret_key=settings.auth_keycloak_client_secret,
        )

    def _get_openid_endpoint(self, settings: "CLISettings", endpoint: str) -> str:
        """Build OpenID Connect endpoint URL."""
        return (
            f"{settings.auth_keycloak_server_url}/realms/"
            f"{settings.auth_keycloak_realm}/protocol/openid-connect/{endpoint}"
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
            "client_id": settings.auth_keycloak_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return f"{self._get_openid_endpoint(settings, 'auth')}?{urlencode(params)}"

    def _parse_token_response(self, token_response: dict) -> AuthResult:
        """Parse Keycloak token response into AuthResult."""
        if not token_response.get("access_token"):
            raise AuthenticationError("Keycloak authentication failed: No access token received")

        return AuthResult(
            access_token=token_response["access_token"],
            expires_in=token_response.get("expires_in", DEFAULT_EXPIRES_IN),
            refresh_token=token_response.get("refresh_token"),
        )

    def _parse_http_response(self, response: httpx.Response, error_prefix: str) -> AuthResult:
        """Parse HTTP response, raising AuthenticationError on failure."""
        if response.status_code != 200:
            try:
                error_data = response.json()
                error_msg = error_data.get(
                    "error_description", error_data.get("error", "Unknown error")
                )
            except Exception:
                error_msg = response.text or f"HTTP {response.status_code}"
            raise AuthenticationError(f"{error_prefix}: {error_msg}")

        return self._parse_token_response(response.json())

    def interactive_login(self, settings: "CLISettings") -> AuthResult:
        """Perform interactive Keycloak login via browser using PKCE."""
        self.validate_config(settings, interactive=True)

        code_verifier = generate_code_verifier()
        code_challenge = generate_code_challenge(code_verifier)
        state = secrets.token_urlsafe(32)

        port = get_callback_port(settings)
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

        OAuthCallbackHandler.reset()
        webbrowser.open(auth_url)
        run_oauth_callback_server(port)

        if OAuthCallbackHandler.error:
            raise AuthenticationError(
                f"Keycloak authentication failed: {OAuthCallbackHandler.error}"
            )

        if not OAuthCallbackHandler.authorization_code:
            raise AuthenticationError(
                "Keycloak authentication failed: No authorization code received (timeout?)"
            )

        if OAuthCallbackHandler.state_received != state:
            raise AuthenticationError(
                "Keycloak authentication failed: State mismatch (possible CSRF attack)"
            )

        try:
            token_response = keycloak_client.token(
                grant_type="authorization_code",
                code=OAuthCallbackHandler.authorization_code,
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
            )
            return self._parse_token_response(token_response)
        except KeycloakAuthenticationError as e:
            raise AuthenticationError(f"Keycloak token exchange failed: {e}") from e
        except KeycloakError as e:
            raise AuthenticationError(f"Keycloak error: {e}") from e

    def system_user_login(self, settings: "CLISettings") -> AuthResult:
        """Perform Keycloak M2M login using Client Credentials Grant.

        Requires a confidential client with "Service accounts roles" enabled.
        """
        self.validate_config(settings, interactive=False)

        token_data = {
            "grant_type": "client_credentials",
            "client_id": settings.auth_keycloak_client_id,
            "client_secret": settings.auth_keycloak_client_secret,
            "scope": settings.auth_keycloak_scope or "openid",
        }

        try:
            response = httpx.post(
                self._get_openid_endpoint(settings, "token"),
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return self._parse_http_response(response, "Keycloak M2M authentication failed")
        except httpx.RequestError as e:
            raise AuthenticationError(f"Keycloak connection error: {e}") from e

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
