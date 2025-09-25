from common.auth.token import TokenResponseI


class OboFlowI:
    """
    https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-on-behalf-of-flow
    """

    async def authorize(self, token: str) -> TokenResponseI:
        raise NotImplementedError()


class AuthGrantI:
    async def authorize(self) -> TokenResponseI:
        raise NotImplementedError()


class RopcGrantI(AuthGrantI):
    """
    https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth-ropc
    """


class ClientCredentialsI(AuthGrantI):
    """
    https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-client-creds-grant-flow#get-a-token
    """


class MsiGrantI(AuthGrantI):
    """
    https://learn.microsoft.com/ru-ru/python/api/azure-identity/azure.identity.managedidentitycredential
    """

    async def authorize(self) -> TokenResponseI:
        raise NotImplementedError()


class TokenRefreshI:
    async def authorize(self, refresh_token: str) -> TokenResponseI:
        raise NotImplementedError()
