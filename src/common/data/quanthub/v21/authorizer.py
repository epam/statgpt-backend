from common.auth import msi
from common.auth.auth_context import AuthContext
from common.auth.authorizer import (
    DialUserAuthorizer,
    DialUserAuthorizerConfig,
    DialUserAuthorizerI,
    DialUserTokenRefreshDecorator,
    SystemUserAuthorizeConfig,
    SystemUserAuthorizer,
    SystemUserAuthorizerI,
    SystemUserTokenRefreshDecorator,
)
from common.auth.grants import (
    AuthorizationError,
    ClientCredentialsConfig,
    ClientCredentialsGrant,
    MsiConfig,
    OboFlow,
    OboFlowConfig,
    RopcGrant,
    RopcGrantConfig,
    TokenRefresh,
)
from common.auth.msi import CachedMsiAuthorizer
from common.auth.token_cache import GlobalTokenCache
from common.config import logger as logger
from common.config.auth import ClientsSpaChatConfig, TTYDChatConfig
from common.data.quanthub.config import AuthConfig, AuthGrantType
from common.data.sdmx.common.authorizer import IAuthorizer


class QuanthubAuthorizer(IAuthorizer):
    def __init__(
        self,
        auth_context: AuthContext,
        system_user_authorizer: SystemUserAuthorizerI,
        dial_user_authorizer: DialUserAuthorizerI,
        forward_dial_token: bool = False,
    ):
        self._auth_context = auth_context
        self._system_user_authorizer = system_user_authorizer
        self._dial_user_authorizer = dial_user_authorizer
        self.forward_dial_token = forward_dial_token

    async def get_authorization_headers(self) -> dict[str, str]:
        """Get authorization headers for the request."""
        access_token = await self._authorize()
        return {"Authorization": f"Bearer {access_token}"}

    async def _authorize(self) -> str:
        if self._auth_context.is_system:
            logger.info('Authorizing as system user')
            system_user_token = await self._system_user_authorizer.authorize(
                SystemUserAuthorizeConfig()
            )
            return system_user_token.access_token
        else:
            dial_token = self._auth_context.dial_access_token
            if not dial_token:
                raise AuthorizationError(
                    'AuthorizationError', 'DIAL access token is required for user authorization'
                )

            if self.forward_dial_token:
                logger.info("Forward DIAL token.")
                return dial_token
            try:
                logger.info("Try to authorize with dial access token.")
                logger.debug(f"DIAL access token: {dial_token}")
                config = DialUserAuthorizerConfig(dial_token=dial_token)
                token = await self._dial_user_authorizer.authorize(config)
                logger.info("Authorized with dial user token")
                return token.access_token
            except AuthorizationError as e:
                logger.exception("Failed to authorize with dial token")
                raise e


class QuanthubAuthorizerFactory:
    @staticmethod
    def _create_dial_user_authorizer(auth_config: AuthConfig) -> DialUserAuthorizerI:
        ttyd_chat_config = OboFlowConfig(
            client_id=ClientsSpaChatConfig.CLIENT_ID,
            client_secret=ClientsSpaChatConfig.CLIENT_SECRET,  # type: ignore
            scope=TTYDChatConfig.SCOPE,
            oauth2_token_endpoint_url=ClientsSpaChatConfig.OAUTH2_TOKEN_ENDPOINT_URL,
        )

        quanthub_config = OboFlowConfig(
            client_id=TTYDChatConfig.CLIENT_ID,
            client_secret=TTYDChatConfig.CLIENT_SECRET,  # type: ignore
            scope=auth_config.obo_flow.get_target_scope(),
            oauth2_token_endpoint_url=TTYDChatConfig.OAUTH2_TOKEN_ENDPOINT_URL,
        )

        ttyd_chat_obo_flow = OboFlow(ttyd_chat_config)
        qh_obo_flow = OboFlow(quanthub_config)

        return DialUserTokenRefreshDecorator(
            DialUserAuthorizer(ttyd_chat_obo_flow, qh_obo_flow),
            TokenRefresh(quanthub_config),
            GlobalTokenCache.get_or_create(),
        )

    @staticmethod
    def create_system_user_authorizer(auth_config_model: AuthConfig) -> SystemUserAuthorizerI:
        grant_type = auth_config_model.get_grant_type()
        if grant_type == AuthGrantType.ROPC:
            logger.info("ROPC grant selected for system user")
            ropc_config = RopcGrantConfig(
                username=auth_config_model.get_ropc_config().system_user_credentials.get_username(),
                password=auth_config_model.get_ropc_config().system_user_credentials.get_password(),
                client_id=TTYDChatConfig.CLIENT_ID,
                client_secret=TTYDChatConfig.CLIENT_SECRET,  # type: ignore
                scope=auth_config_model.get_ropc_config().get_target_scope(),
                oauth2_token_endpoint_url=TTYDChatConfig.OAUTH2_TOKEN_ENDPOINT_URL,
            )
            system_user_grant = RopcGrant(ropc_config)
            return SystemUserTokenRefreshDecorator(
                SystemUserAuthorizer(system_user_grant),
                TokenRefresh(ropc_config),
                GlobalTokenCache.get_or_create(),
                cache_key="ropc_user",
            )
        elif grant_type == AuthGrantType.CLIENT_CREDENTIALS:
            logger.info("Client Credentials grant selected for system user")
            cc_config = ClientCredentialsConfig(
                client_id=TTYDChatConfig.CLIENT_ID,
                client_secret=TTYDChatConfig.CLIENT_SECRET,  # type: ignore
                scope=auth_config_model.get_client_credentials_config().get_target_scope(),
                oauth2_token_endpoint_url=TTYDChatConfig.OAUTH2_TOKEN_ENDPOINT_URL,
            )
            return SystemUserTokenRefreshDecorator(
                SystemUserAuthorizer(ClientCredentialsGrant(cc_config)),
                TokenRefresh(cc_config),
                GlobalTokenCache.get_or_create(),
                cache_key="client_credentials_user",
            )
        elif grant_type == AuthGrantType.MSI:
            logger.info("MSI grant selected for system user")
            msi_config = MsiConfig(scope=auth_config_model.get_msi_config().get_target_scope())
            return CachedMsiAuthorizer(
                msi.MsiGrant(msi.Config(scope=msi_config.scope)),
                GlobalTokenCache.get_or_create(),
                token_cache_key="msi_user",
            )
        else:
            raise RuntimeError(
                f"Unexpected value for auth_config.type: {grant_type}. "
                f"Should be one of AuthGrantType values"
            )

    def create(self, auth_context: AuthContext, auth_config: AuthConfig) -> QuanthubAuthorizer:
        return QuanthubAuthorizer(
            auth_context,
            self.create_system_user_authorizer(auth_config),
            self._create_dial_user_authorizer(auth_config),
            auth_config.forward_dial_token,
        )
