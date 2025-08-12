from typing import Self
from urllib.parse import urljoin

import httpx
from pydantic import SecretStr

from common.auth.auth_context import AuthContext
from common.config import DialConfig
from common.config import multiline_logger as logger
from statgpt.config import DialRagConfig
from statgpt.schemas.file_rags.dial_rag import DialRagMetadataResponse


class DialRagMetadataLoader:
    def __init__(self, dial_rag_metadata_url: str, dial_rag_metadata_api_key: SecretStr):
        self._dial_rag_metadata_url = dial_rag_metadata_url
        self._dial_rag_metadata_api_key = dial_rag_metadata_api_key

    @classmethod
    def create_for_local_or_remote(cls, auth_context: AuthContext) -> Self:
        if non_default_url := DialRagConfig.DIAL_RAG_PGVECTOR_URL:
            logger.info(f"Using non-default DIAL RAG Metadata url: {non_default_url}")
            base_url = non_default_url
            key = DialRagConfig.DIAL_RAG_PGVECTOR_API_KEY or SecretStr('')
        else:
            base_url = DialConfig.get_url()
            key = SecretStr(auth_context.api_key)

        return cls(
            dial_rag_metadata_url=urljoin(base_url, DialRagConfig.METADATA_ENDPOINT),
            dial_rag_metadata_api_key=key,
        )

    async def load(self) -> DialRagMetadataResponse:
        response_dict = await self._load()
        return DialRagMetadataResponse.model_validate(response_dict)

    async def _load(self) -> dict:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(
                self._dial_rag_metadata_url,
                headers={
                    "api-key": self._dial_rag_metadata_api_key.get_secret_value(),
                },
            )

        if not response.is_success:
            logger.error(
                f'Failed to get metadata info from dial-rag. '
                f'Got {response.status_code} code for '
                f'{response.request.method} '
                f'request to {response.url}. '
                f'response content: {response.content}'
            )
        response.raise_for_status()

        response_dict = response.json()
        return response_dict
