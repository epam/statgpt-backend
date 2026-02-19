from typing import Any

import httpx
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import SecretStr

from statgpt.common.config.logging import multiline_logger as logger
from statgpt.common.schemas import EmbeddingsModelConfig, LLMModelConfig
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils.callbacks import BrokenResponseInterceptor


def get_chat_model(
    api_key: str | SecretStr,
    model_config: LLMModelConfig,
    azure_endpoint: str = dial_settings.url,
    timeout: httpx.Timeout | None = None,
) -> AzureChatOpenAI:
    if not isinstance(api_key, SecretStr):
        api_key = SecretStr(api_key)
    if not timeout:
        timeout = httpx.Timeout(60, connect=4)
    params: dict[str, Any] = dict(
        azure_endpoint=azure_endpoint,
        api_version=model_config.api_version,
        azure_deployment=model_config.deployment.deployment_id,
        max_retries=10,
        api_key=api_key,  # since we use SecretStr, it won't be logged
        timeout=timeout,  # timeouts are crucial!
    )

    params.update(model_config.model_dump(mode="json", exclude_none=True, exclude={"deployment"}))

    if model_config.deployment.is_gpt_41_family:
        callback = BrokenResponseInterceptor(regex_pattern=r"\s{5,}")
        params.setdefault("callbacks", []).append(callback)

    api_key_log = f"{api_key.get_secret_value()[:3]}*****{api_key.get_secret_value()[-2:]}"
    logger.info(
        f"creating langchain LLM with the following params: {params}, Api key: {api_key_log}"
    )
    return AzureChatOpenAI.model_validate(params)


def get_embeddings_model(
    api_key: str | SecretStr,
    model_config: EmbeddingsModelConfig,
    azure_endpoint: str = dial_settings.url,
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
    api_key_log = f"{api_key.get_secret_value()[:3]}*****{api_key.get_secret_value()[-2:]}"
    logger.info(
        f"creating langchain embeddings with the following params: {params}, Api key: {api_key_log}"
    )
    return AzureOpenAIEmbeddings.model_validate(params)
