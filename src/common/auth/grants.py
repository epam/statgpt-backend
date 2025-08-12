from urllib.parse import urlencode

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from common.auth.base import ClientCredentialsI, OboFlowI, RopcGrantI, TokenRefreshI
from common.auth.oauth_token import OAuthTokenResponse


class AuthConfig(BaseModel):
    scope: str


class ConfidentialClientApplicationConfig(AuthConfig):
    """Abstract class"""

    oauth2_token_endpoint_url: str
    client_id: str
    client_secret: SecretStr

    def get_client_secret(self) -> str:
        return self.client_secret.get_secret_value()


class MsiConfig(AuthConfig):
    pass


class OboFlowConfig(ConfidentialClientApplicationConfig):
    pass


class ClientCredentialsConfig(ConfidentialClientApplicationConfig):
    pass


class RopcGrantConfig(ConfidentialClientApplicationConfig):
    username: str
    password: str


class AuthorizationError(Exception):
    def __init__(self, error: str, error_description: str):
        self.error = error
        self.error_description = error_description

    def __str__(self):
        return f'{self.error}: {self.error_description}'

    @classmethod
    def check(cls, data: dict):
        if data.get("error", ""):
            raise cls(data.get("error", ""), data.get("error_description", ""))


class TokenData(BaseModel):
    model_config = ConfigDict(extra='ignore')

    access_token: str
    expires_in: int
    refresh_token: str | None = Field(default=None)
    # add other fields when needed


class TokenGrantFlowMixin:
    GRANT_TYPE: str = ""

    @property
    def config(self) -> ConfidentialClientApplicationConfig:
        raise NotImplementedError()

    def create_params(self, *args, **kwargs) -> dict:
        raise NotImplementedError()

    async def _authorize(self, *args, **kwargs) -> OAuthTokenResponse:
        params = self.create_params(*args, **kwargs)
        body = urlencode(params)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.oauth2_token_endpoint_url,
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as response:
                token_data = await response.json()
                AuthorizationError.check(token_data)
                token_data = TokenData(**token_data)
                return OAuthTokenResponse(
                    access_token=token_data.access_token,
                    expires_in=token_data.expires_in,
                    refresh_token=token_data.refresh_token,
                )


class OboFlow(OboFlowI, TokenGrantFlowMixin):
    GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
    REQUESTED_TOKEN_USE = "on_behalf_of"

    def __init__(self, config: OboFlowConfig):
        self._config = config

    @property
    def config(self) -> OboFlowConfig:
        return self._config

    def create_params(self, token: str) -> dict:
        return {
            "grant_type": self.GRANT_TYPE,
            "client_id": self.config.client_id,
            "client_secret": self.config.get_client_secret(),
            "assertion": token,
            "scope": f"{self.config.scope} openid offline_access",
            "requested_token_use": self.REQUESTED_TOKEN_USE,
        }

    async def authorize(self, token: str) -> OAuthTokenResponse:
        return await self._authorize(token)


class RopcGrant(RopcGrantI, TokenGrantFlowMixin):
    GRANT_TYPE = "password"

    def __init__(self, config: RopcGrantConfig):
        self._config = config

    @property
    def config(self) -> RopcGrantConfig:
        return self._config

    def create_params(self) -> dict:
        return {
            "grant_type": self.GRANT_TYPE,
            "client_id": self.config.client_id,
            "client_secret": self.config.get_client_secret(),
            "username": self.config.username,
            "password": self.config.password,
            "scope": f"{self.config.scope} openid offline_access",
        }

    async def authorize(self) -> OAuthTokenResponse:
        return await self._authorize()


class TokenRefresh(TokenRefreshI, TokenGrantFlowMixin):
    GRANT_TYPE = "refresh_token"

    def __init__(self, config: ConfidentialClientApplicationConfig):
        self._config = config

    @property
    def config(self) -> ConfidentialClientApplicationConfig:
        return self._config

    def create_params(self, refresh_token: str) -> dict:
        return {
            "grant_type": self.GRANT_TYPE,
            "client_id": self.config.client_id,
            "client_secret": self.config.get_client_secret(),
            "refresh_token": refresh_token,
        }

    async def authorize(self, refresh_token: str) -> OAuthTokenResponse:
        return await self._authorize(refresh_token)


class ClientCredentialsGrant(ClientCredentialsI, TokenGrantFlowMixin):
    GRANT_TYPE = "client_credentials"

    def __init__(self, config: ConfidentialClientApplicationConfig):
        self._config = config

    @property
    def config(self) -> ConfidentialClientApplicationConfig:
        return self._config

    def create_params(self) -> dict:
        return {
            "grant_type": self.GRANT_TYPE,
            "client_id": self.config.client_id,
            "client_secret": self.config.get_client_secret(),
            "scope": f"{self.config.scope}",
        }

    async def authorize(self) -> OAuthTokenResponse:
        return await self._authorize()
