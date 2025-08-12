from azure.core.credentials import AccessToken
from azure.identity.aio import ManagedIdentityCredential
from pydantic import BaseModel

from common.auth.authorizer import SystemUserAuthorizeConfig, SystemUserAuthorizer
from common.auth.base import MsiGrantI
from common.auth.token import TokenResponseI
from common.auth.token_cache import TokenCache
from common.config import logger


class MsiTokenResponse(TokenResponseI):
    def __init__(self, msi_token: AccessToken):
        self._msi_token = msi_token

    @property
    def expires_at(self) -> int:
        return self._msi_token.expires_on

    @property
    def access_token(self) -> str:
        return self._msi_token.token

    @property
    def refresh_token(self) -> str | None:
        logger.warning("MSI token does not support refreshing")
        return None


class Config(BaseModel):
    scope: str


class MsiGrant(MsiGrantI):
    def __init__(self, config: Config):
        self._config = config

    async def authorize(self) -> MsiTokenResponse:
        async with ManagedIdentityCredential() as credential:
            token = await credential.get_token(self._config.scope)
        return MsiTokenResponse(token)


# # todo s.sych: use TokenRefreshDecorator
class CachedMsiAuthorizer(SystemUserAuthorizer):
    def __init__(
        self, msi_grant: MsiGrant, token_cache: TokenCache, token_cache_key: str = "msi_token"
    ):
        super().__init__(msi_grant)
        self._token_cache = token_cache
        self._token_cache_key = token_cache_key

    async def authorize(self, config: SystemUserAuthorizeConfig) -> TokenResponseI:
        token = self._token_cache.get_not_expired(self._token_cache_key)  # type: ignore
        if not token:
            token = await super().authorize(config)
            logger.info("MSI token has been granted")
            self._token_cache.add(self._token_cache_key, token)
        else:
            logger.info("MSI token has been taken from cache")
        return token
