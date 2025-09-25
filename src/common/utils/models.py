import httpx
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import SecretStr

from common.config.logging import multiline_logger as logger
from common.schemas import EmbeddingsModelConfig, LLMModelConfig
from common.settings.dial import dial_settings


def get_chat_model(
    api_key: str | SecretStr,
    model_config: LLMModelConfig,
    azure_endpoint: str = dial_settings.url,
    **kwargs,
) -> AzureChatOpenAI:
    # default params
    if not isinstance(api_key, SecretStr):
        api_key = SecretStr(api_key)
    params = dict(
        azure_endpoint=azure_endpoint,
        api_version=model_config.api_version,
        azure_deployment=model_config.deployment.deployment_id,
        temperature=model_config.temperature,
        seed=model_config.seed,
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
    model_config: EmbeddingsModelConfig,
    azure_endpoint: str = dial_settings.url,
    **kwargs,
) -> AzureOpenAIEmbeddings:
    if not isinstance(api_key, SecretStr):
        api_key = SecretStr(api_key)
    params = dict(
        azure_endpoint=azure_endpoint,
        azure_deployment=model_config.deployment.value,
        api_version=model_config.api_version,
        max_retries=10,
        api_key=api_key,  # since we use SecretStr, it won't be logged
    )
    params.update(kwargs)  # update default params
    api_key_log = f'{api_key.get_secret_value()[:3]}*****{api_key.get_secret_value()[-2:]}'
    logger.info(
        f'creating langchain embeddings with the following params: {params}, Api key: {api_key_log}'
    )
    return AzureOpenAIEmbeddings.model_validate(params)
