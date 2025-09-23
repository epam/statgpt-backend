from openai import AsyncAzureOpenAI
from pydantic import SecretStr

from common.settings.dial import dial_settings
from common.settings.langchain import langchain_settings


def get_async_client(
    api_key: str | SecretStr,
    azure_endpoint=dial_settings.url,
    api_version: str = langchain_settings.default_api_version,
) -> AsyncAzureOpenAI:
    if isinstance(api_key, SecretStr):
        api_key = api_key.get_secret_value()
    return AsyncAzureOpenAI(azure_endpoint=azure_endpoint, api_version=api_version, api_key=api_key)
