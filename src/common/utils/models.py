import httpx
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import SecretStr

from common.config import DialConfig, LangChainConfig
from common.config.logging import multiline_logger as logger


def get_chat_model(
    api_key: str | SecretStr,
    model: str = LangChainConfig.DEFAULT_MODEL,
    temperature: float = LangChainConfig.DEFAULT_TEMPERATURE,
    azure_endpoint: str = DialConfig.get_url(),
    api_version: str = LangChainConfig.DEFAULT_API_VERSION,
    seed: int | None = LangChainConfig.DEFAULT_SEED,
    **kwargs,
) -> AzureChatOpenAI:
    # default params
    if not isinstance(api_key, SecretStr):
        api_key = SecretStr(api_key)
    params = dict(
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_deployment=model,
        temperature=temperature,
        seed=seed,
        max_retries=10,
        api_key=api_key,  # since we use SecretStr, it won't be logged
        timeout=httpx.Timeout(60, connect=4),  # timeouts are crucial!
    )
    params.update(kwargs)  # update default params
    api_key_log = f'{api_key.get_secret_value()[:3]}*****{api_key.get_secret_value()[-2:]}'
    logger.info(
        f'creating langchain LLM with the following params: {params}, Api key: {api_key_log}'
    )
    return AzureChatOpenAI.model_validate(params)


def get_embeddings_model(
    api_key: str | SecretStr,
    model: str,
    api_version: str = LangChainConfig.DEFAULT_API_VERSION,
    **kwargs,
) -> AzureOpenAIEmbeddings:
    if not isinstance(api_key, SecretStr):
        api_key = SecretStr(api_key)
    params = dict(
        azure_endpoint=DialConfig.get_url(),
        azure_deployment=model,
        api_version=api_version,
        max_retries=10,
        api_key=api_key,  # since we use SecretStr, it won't be logged
    )
    params.update(kwargs)  # update default params
    api_key_log = f'{api_key.get_secret_value()[:3]}*****{api_key.get_secret_value()[-2:]}'
    logger.info(
        f'creating langchain embeddings with the following params: {params}, Api key: {api_key_log}'
    )
    return AzureOpenAIEmbeddings.model_validate(params)
