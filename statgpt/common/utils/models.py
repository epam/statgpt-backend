from typing import Any

import httpx
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import SecretStr

from statgpt.common.config import ReasoningEffortEnum
from statgpt.common.config.logging import multiline_logger as logger
from statgpt.common.schemas import EmbeddingsModelConfig, LLMModelConfig
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils.callbacks import BrokenResponseInterceptor


def get_chat_model(
    api_key: str | SecretStr,
    model_config: LLMModelConfig,
    azure_endpoint: str = dial_settings.url,
    timeout: httpx.Timeout | None = None,
    **kwargs,
) -> AzureChatOpenAI:
    # default params
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

    if model_config.deployment.is_gpt_5_family:
        # GPT-5: use reasoning_effort parameter
        if model_config.reasoning_effort is not None:
            params["reasoning_effort"] = model_config.reasoning_effort
            if model_config.reasoning_effort == ReasoningEffortEnum.NONE:
                # reasoning_effort=none: use temperature=0 for deterministic output
                params["temperature"] = 0
            else:
                # NOTE: Temporarily set temperature=1 for reasoning modes (minimal/low/medium/high/xhigh)
                # TODO: Remove this once Azure OpenAI API is upgraded to properly handle reasoning modes without temperature
                params["temperature"] = 1
        params.update({k: v for k, v in kwargs.items() if k not in ("temperature", "seed")})
    else:
        # Legacy models: use temperature and seed
        params["temperature"] = model_config.temperature
        if model_config.seed is not None:
            params["seed"] = model_config.seed
        params.update(kwargs)

    if model_config.deployment.is_gpt_41_family:
        callback = BrokenResponseInterceptor(regex_pattern=r'\s{5,}')
        params.setdefault('callbacks', []).append(callback)

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
