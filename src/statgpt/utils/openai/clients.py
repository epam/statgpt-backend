from openai import AsyncAzureOpenAI
from pydantic import SecretStr

from common.config import DialConfig, LangChainConfig


def get_async_client(
    api_key: str | SecretStr,
    azure_endpoint=DialConfig.get_url(),
    api_version: str = LangChainConfig.DEFAULT_API_VERSION,
) -> AsyncAzureOpenAI:
    if isinstance(api_key, SecretStr):
        api_key = api_key.get_secret_value()
    return AsyncAzureOpenAI(azure_endpoint=azure_endpoint, api_version=api_version, api_key=api_key)
