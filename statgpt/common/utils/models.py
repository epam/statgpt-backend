import warnings
from typing import Any

import httpx
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import SecretStr

# langchain_openai's _should_stream warns unconditionally when response_format is a Pydantic type,
# even when streaming is disabled. This is a bug in langchain_openai.
warnings.filterwarnings(
    "ignore",
    message="Streaming with Pydantic response_format not yet supported.",
    category=UserWarning,
    module=r"langchain_openai\.chat_models\.base",
)

# langchain_openai serializes openai SDK's generic ParsedChatCompletion[ContentType] after a
# structured-output call. Under pydantic 2.11+ this emits a spurious
# PydanticSerializationUnexpectedValue for the `parsed` field (generic TypeVar vs. a concrete
# BaseModel value), even though `parsed` is already in the `exclude` set and is not serialized.
# Tracked upstream: openai-python PR #2885 (still open as of 2026-04-23).
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:[\s\S]*field_name='parsed'",
    category=UserWarning,
    module=r"pydantic\.main",
)

from statgpt.common.config.logging import multiline_logger as logger
from statgpt.common.schemas import EmbeddingsModelConfig, LLMModelConfig
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils.callbacks import BrokenResponseInterceptor
from statgpt.common.utils.http_pool import get_shared_llm_http_client


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
        http_async_client=get_shared_llm_http_client(),  # shared pool; per-request timeout still applies
    )

    params.update(model_config.model_dump(mode="json", exclude_none=True, exclude={"deployment"}))

    if model_config.deployment.is_gpt_41_family:
        callback = BrokenResponseInterceptor(regex_pattern=r'\s{5,}')
        params.setdefault('callbacks', []).append(callback)

    api_key_log = f'{api_key.get_secret_value()[:3]}*****{api_key.get_secret_value()[-2:]}'
    params_log = {k: v for k, v in params.items() if k not in ('api_key', 'http_async_client')}
    logger.info(
        f'creating langchain LLM with the following params: {params_log}, Api key: {api_key_log}'
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
        http_async_client=get_shared_llm_http_client(),  # shared pool
    )
    api_key_log = f'{api_key.get_secret_value()[:3]}*****{api_key.get_secret_value()[-2:]}'
    params_log = {k: v for k, v in params.items() if k not in ('api_key', 'http_async_client')}
    logger.info(
        f'creating langchain embeddings with the following params: {params_log}, Api key: {api_key_log}'
    )
    return AzureOpenAIEmbeddings.model_validate(params)
