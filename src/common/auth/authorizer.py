import typing as t

from pydantic import BaseModel

from common.auth.base import AuthGrantI, OboFlowI, TokenRefreshI, TokenResponseI
from common.auth.token_cache import TokenCache
from common.config import logger as logger


class AuthorizeConfig(BaseModel):
    pass


T = t.TypeVar('T', bound=AuthorizeConfig)


class AuthorizerI(t.Generic[T]):
    async def authorize(self, config: T) -> TokenResponseI:
        raise NotImplementedError()


class DialUserAuthorizerConfig(AuthorizeConfig):
    dial_token: str


class SystemUserAuthorizeConfig(AuthorizeConfig):
    pass


class DialUserAuthorizerI(AuthorizerI[DialUserAuthorizerConfig]):
    async def authorize(self, config: DialUserAuthorizerConfig) -> TokenResponseI:
        raise NotImplementedError()


class SystemUserAuthorizerI(AuthorizerI[SystemUserAuthorizeConfig]):
    async def authorize(self, config: SystemUserAuthorizeConfig) -> TokenResponseI:
        raise NotImplementedError()


class DialUserAuthorizer(DialUserAuthorizerI):
    def __init__(self, chat_obo_flow: OboFlowI, qh_obo_flow: OboFlowI):
        self.ttyd_chat_obo_flow = chat_obo_flow
        self.quanthub_obo_flow = qh_obo_flow

    async def authorize(self, config: DialUserAuthorizerConfig) -> TokenResponseI:
        logger.info("Requesting OBO flow access token exchange for dial user")
        chat_token = await self.ttyd_chat_obo_flow.authorize(config.dial_token)
        logger.debug(f"ServicesChat OBO-flow access token: {chat_token.access_token}")
        qh_token_result = await self.quanthub_obo_flow.authorize(chat_token.access_token)
        logger.debug(f"Quanthub OBO-flow access token: {qh_token_result.access_token}")
        return qh_token_result


class SystemUserAuthorizer(SystemUserAuthorizerI):
    def __init__(self, grant: AuthGrantI):
        self.grant = grant

    async def authorize(self, config: SystemUserAuthorizeConfig) -> TokenResponseI:
        logger.info(f"Requesting system user token with {type(self.grant)}")
        return await self.grant.authorize()


class TokenRefreshDecorator(AuthorizerI[T]):
    def __init__(
        self,
        authorizer: AuthorizerI,
        token_refresh: TokenRefreshI,
        token_cache=TokenCache(),
    ):
        self._authorizer = authorizer
        self._token_cache = token_cache
        self._token_refresh = token_refresh

    def _get_cache_key(self, config: T) -> str:
        raise NotImplementedError()

    async def authorize(self, config: T) -> TokenResponseI:
        cache_key = self._get_cache_key(config)
        token = self._token_cache.get_not_expired(cache_key)
        if token:
            logger.info("Token has been taken from cache")
            return token
        else:
            expired_token = self._token_cache.get(cache_key)
            if expired_token:
                if expired_token.refresh_token:
                    logger.info("Refresh token based on expired token")
                    token = await self._token_refresh.authorize(expired_token.refresh_token)
                else:
                    logger.info("Refresh token is absent - request new token instead")
                    token = await self._authorizer.authorize(config)
            else:
                logger.info("Token is not in cache - request new")
                token = await self._authorizer.authorize(config)
            self._token_cache.add(cache_key, token)
        return token


class DialUserTokenRefreshDecorator(
    TokenRefreshDecorator[DialUserAuthorizerConfig], DialUserAuthorizerI
):
    def _get_cache_key(self, config: DialUserAuthorizerConfig) -> str:
        return config.dial_token


class SystemUserTokenRefreshDecorator(
    TokenRefreshDecorator[SystemUserAuthorizeConfig], SystemUserAuthorizerI
):
    def __init__(
        self,
        authorizer: SystemUserAuthorizerI,
        token_refresh: TokenRefreshI,
        token_cache=TokenCache(),
        cache_key: str = "system_user",
    ):
        super().__init__(authorizer, token_refresh, token_cache)
        self._cache_key = cache_key

    def _get_cache_key(self, config: SystemUserAuthorizeConfig) -> str:
        return self._cache_key
