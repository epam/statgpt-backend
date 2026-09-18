from openai import AsyncAzureOpenAI
from pydantic import SecretStr

from statgpt.common.settings.dial import dial_settings
from statgpt.common.settings.langchain import langchain_settings


def get_async_client(
    api_key: str | SecretStr,
    azure_endpoint=dial_settings.url,
    api_version: str = langchain_settings.default_api_version,
    access_token: str | None = None,
) -> AsyncAzureOpenAI:
    """Client for a DIAL deployment, authenticated with `api_key`.

    When the caller's `access_token` is passed it is sent as a bearer header alongside the api key:
    DIAL Core authenticates on the api key and relays the token to the deployment when its config
    sets ``forwardAuthToken``, so a nested deployment can authorize downstream calls as the user
    (e.g. QuantHub token forwarding or the OBO flow).
    """
    if isinstance(api_key, SecretStr):
        api_key = api_key.get_secret_value()

    default_headers = (
        {"Authorization": f"Bearer {access_token}"} if access_token is not None else None
    )
    return AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        api_key=api_key,
        default_headers=default_headers,
    )
