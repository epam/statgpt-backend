"""Shared transport for the channel API of a Generic RAG DIAL application.

The write side (`ingestion`) and the read side (`search`) speak to the same `/channel/...`
endpoints of the same application, differing only in which endpoints they call and whose DIAL
key they authenticate with. That common half lives here so neither module has to carry it.
"""

import logging
from typing import Any, Self, TypeVar

import httpx
from pydantic import BaseModel, SecretStr

from statgpt.common.schemas.generic_rag import GenericRagMetadataSchema
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils import ManagedHttpClient

_log = logging.getLogger(__name__)

_HTTP_TIMEOUT = httpx.Timeout(60.0)

_MAX_DETAIL_CHARS = 500
"""Cap on how much of a failure body is quoted.

The reason ends up in `index_error` or `reason_for_failure`, or in a chat-time debug stage, so an
HTML error page must not arrive there in full.
"""

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class GenericRagChannelError(Exception):
    """A request to the Generic RAG channel API failed.

    Carries the operation and, when there was a response, its status: the message is what a
    caller records on a record, a job row or a debug stage, so it has to say which call failed
    and what came back.
    """

    def __init__(self, operation: str, detail: str, status_code: int | None = None) -> None:
        self.operation = operation
        self.detail = detail
        self.status_code = status_code
        code = f" with HTTP {status_code}" if status_code is not None else ""
        super().__init__(f"Generic RAG {operation} failed{code}: {detail}")


class BaseGenericRagChannelClient:
    """The `/channel/...` endpoints of one Generic RAG channel.

    A channel is addressed by the DIAL application that fronts it, so one client instance
    speaks to exactly one channel. Own the instance for the length of a run - it holds an
    HTTP connection pool - and close it, ideally by using it as an async context manager.
    """

    def __init__(self, base_url: str, api_key: SecretStr) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http = ManagedHttpClient(_HTTP_TIMEOUT)

    @property
    def base_url(self) -> str:
        """The channel route this client is bound to, without a trailing slash."""
        return self._base_url

    @staticmethod
    def application_route(application_id: str) -> str:
        """The DIAL route through which an application's channel API is reached."""
        return f"{dial_settings.url.rstrip('/')}/v1/deployments/{application_id}/route"

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ endpoints ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def get_metadata_schema(self) -> GenericRagMetadataSchema:
        """Read the metadata JSON-schema the channel accepts, and its filterable dimensions.

        Both halves need it, for opposite reasons: a publisher checks that the fields search
        relies on are declared filterable, and a search reads `dimensions` to learn which
        values it is allowed to filter by at all.
        """
        response = await self._request("metadata read", "GET", "/metadata")
        return self._parse(GenericRagMetadataSchema, response, "metadata read")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ transport ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def _request(
        self, operation: str, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        url = f"{self._base_url}/channel{path}"
        try:
            response = await self._http.client.request(
                method, url, headers={"Api-Key": self._api_key.get_secret_value()}, **kwargs
            )
        except httpx.HTTPError as e:
            raise GenericRagChannelError(operation, f"{type(e).__name__}: {e}") from e

        if response.is_success:
            return response

        _log.warning(f"Generic RAG {operation} at {url} returned HTTP {response.status_code}")
        raise GenericRagChannelError(operation, self._detail(response.text), response.status_code)

    @staticmethod
    def _parse(model: type[_ModelT], response: httpx.Response, operation: str) -> _ModelT:
        """Validate a response body, reporting a surprise as a failure of that operation.

        A body that does not parse means the caller cannot proceed, and it is far more
        useful to name the call than to surface a bare `ValidationError` from a job log.
        """
        try:
            return model.model_validate(response.json())
        except ValueError as e:
            raise GenericRagChannelError(
                operation, f"unexpected response body: {e}", response.status_code
            ) from e

    @staticmethod
    def _detail(body: str) -> str:
        body = body.strip()
        if len(body) <= _MAX_DETAIL_CHARS:
            return body
        return f"{body[:_MAX_DETAIL_CHARS]}... (truncated)"
