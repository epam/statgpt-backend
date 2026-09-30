import asyncio
import io
import os
import time
from typing import IO, Any

import httpx
import requests
import sdmx
from httpx import Response
from requests import PreparedRequest
from sdmx import Client, Resource
from sdmx.message import DataMessage, Message, StructureMessage
from sdmx.model.v21 import DataStructureDefinition
from sdmx.reader import get_reader
from sdmx.session import ResponseIO

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.sdmx.common.authorizer import IAuthorizer
from statgpt.common.data.sdmx.common.config import SdmxDataSourceConfig
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiter
from statgpt.common.settings.sdmx import sdmx_settings
from statgpt.common.utils import AsyncLoadingCache
from statgpt.common.utils.http_pool import get_shared_sdmx_http_client


def init_sdmx(config: SdmxDataSourceConfig):
    sdmx.add_source(config.sdmx_config.to_sdmx1_dict(), override=True)


class SdmxRequestTimeoutError(Exception):
    """Raised when an SDMX HTTP request times out after exhausting retries.

    The message is safe to surface to callers (e.g. as an offline dataset status, or a
    chat-facing error) so timeouts read as "the data source didn't respond in time"
    instead of a generic failure.
    """


class AsyncSdmxClient:
    """Async client for interacting with the SDMX API."""

    _cache: AsyncLoadingCache[Message] = AsyncLoadingCache(ttl=sdmx_settings.client_cache_ttl)

    @classmethod
    def from_config(
        cls, config: SdmxDataSourceConfig, auth_context: AuthContext, rate_limiter: SdmxRateLimiter
    ) -> "AsyncSdmxClient":
        """Initialize the client from a configuration object."""

        init_sdmx(config)
        sync_client = Client(config.get_id())
        httpx_client = get_shared_sdmx_http_client(config.get_id())

        return cls(sync_client, httpx_client, None, rate_limiter)

    def __init__(
        self,
        sync_client: Client,
        httpx_client: httpx.AsyncClient,
        authorizer: IAuthorizer | None,
        rate_limiter: SdmxRateLimiter,
        static_headers: dict[str, str] | None = None,
    ):
        self._sync_client = sync_client
        self._httpx_client = httpx_client
        self._authorizer = authorizer
        self._rate_limiter = rate_limiter
        # Applied per request (not on the shared httpx client) so a reconfigured source
        # (e.g. a rotated API key) keeps using the same pooled connections.
        self._static_headers = static_headers or {}

    async def dataflow(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        params: dict[str, Any],
        use_cache: bool = False,
    ) -> StructureMessage:
        return await self._get_structure(  # type: ignore[return-value]
            resource_type=Resource.dataflow,
            agency_id=agency_id,
            resource_id=resource_id,
            version=version,
            params=params,
            use_cache=use_cache,
        )

    async def agencyscheme(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        use_cache: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> StructureMessage:
        """Fetch an agency scheme from the SDMX API."""
        return await self._get_structure(  # type: ignore[return-value]
            resource_type=Resource.agencyscheme,
            agency_id=agency_id,
            resource_id=resource_id,
            version=version,
            use_cache=use_cache,
            extra_headers=extra_headers,
        )

    async def categoryscheme(
        self, *, agency_id: str, resource_id: str, version: str, use_cache: bool = False
    ) -> StructureMessage:
        """Fetch a category scheme from the SDMX API with parents and siblings references."""
        return await self._get_structure(  # type: ignore[return-value]
            resource_type=Resource.categoryscheme,
            agency_id=agency_id,
            resource_id=resource_id,
            version=version,
            params={'references': 'parentsandsiblings'},
            use_cache=use_cache,
        )

    async def conceptscheme(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        use_cache: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> StructureMessage:
        return await self._get_structure(  # type: ignore[return-value]
            resource_type=Resource.conceptscheme,
            agency_id=agency_id,
            resource_id=resource_id,
            version=version,
            use_cache=use_cache,
            extra_headers=extra_headers,
        )

    async def codelist(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        use_cache: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> StructureMessage:
        return await self._get_structure(  # type: ignore[return-value]
            resource_type=Resource.codelist,
            agency_id=agency_id,
            resource_id=resource_id,
            version=version,
            use_cache=use_cache,
            extra_headers=extra_headers,
        )

    async def hierarchicalcodelist(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        params: dict[str, str] | None = None,
        use_cache: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> StructureMessage:
        return await self._get_structure(  # type: ignore[return-value]
            resource_type=Resource.hierarchicalcodelist,
            agency_id=agency_id,
            resource_id=resource_id,
            version=version,
            params=params,
            use_cache=use_cache,
            extra_headers=extra_headers,
        )

    async def availableconstraint(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        use_cache: bool = False,
        key: dict[str, list[str]] | None = None,
        params: dict[str, str] | None = None,
        dsd: DataStructureDefinition | None = None,
    ) -> StructureMessage:
        if key and not dsd:
            raise ValueError("Please provide a DataStructureDefinition (dsd) when using `key`.")

        flow_ref = self._get_flow_ref(resource_id, agency_id, version)
        return await self._get_availability(  # type: ignore[return-value]
            resource_type=Resource.availableconstraint,
            resource_id=flow_ref,
            use_cache=use_cache,
            key=key,
            params=params,
            dsd=dsd,
        )

    async def data(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        key: dict[str, list[str]],
        params: dict[str, str],
        dsd: DataStructureDefinition,
    ) -> DataMessage:
        flow_ref = self._get_flow_ref(resource_id, agency_id, version)

        return await self._get_data(  # type: ignore[return-value]
            resource_type=Resource.data,
            resource_id=flow_ref,
            key=key,
            params=params,
            dsd=dsd,
        )

    async def _get_structure(
        self,
        *,
        resource_type: Resource,
        resource_id: str,
        agency_id: str | None = None,
        version: str | None = None,
        key: dict[str, list[str]] | None = None,
        params: dict[str, str] | None = None,
        dsd: DataStructureDefinition | None = None,
        use_cache: bool = False,
        tofile: os.PathLike | IO | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Message:
        async with self._rate_limiter.structure_limiter():
            return await self._get(
                resource_type=resource_type,
                resource_id=resource_id,
                agency_id=agency_id,
                version=version,
                key=key,
                params=params,
                dsd=dsd,
                use_cache=use_cache,
                tofile=tofile,
                extra_headers=extra_headers,
            )

    async def _get_availability(
        self,
        *,
        resource_type: Resource,
        resource_id: str,
        agency_id: str | None = None,
        version: str | None = None,
        key: dict[str, list[str]] | None = None,
        params: dict[str, str] | None = None,
        dsd: DataStructureDefinition | None = None,
        use_cache: bool = False,
        tofile: os.PathLike | IO | None = None,
    ) -> Message:
        try:
            async with self._rate_limiter.availability_limiter():
                return await self._get(
                    resource_type=resource_type,
                    resource_id=resource_id,
                    agency_id=agency_id,
                    version=version,
                    key=key,
                    params=params,
                    dsd=dsd,
                    use_cache=use_cache,
                    tofile=tofile,
                )
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [400, 404]:
                logger.error(f"Bad request for URL {e.request.url!r}: {e.response.text}")
                return StructureMessage()  # Return empty StructureMessage on bad request
            raise

    async def _get_data(
        self,
        *,
        resource_type: Resource,
        resource_id: str,
        agency_id: str | None = None,
        version: str | None = None,
        key: dict[str, list[str]] | None = None,
        params: dict[str, str] | None = None,
        dsd: DataStructureDefinition | None = None,
        use_cache: bool = False,
        tofile: os.PathLike | IO | None = None,
    ) -> Message:
        async with self._rate_limiter.data_limiter():
            return await self._get(
                resource_type=resource_type,
                resource_id=resource_id,
                agency_id=agency_id,
                version=version,
                key=key,
                params=params,
                dsd=dsd,
                use_cache=use_cache,
                tofile=tofile,
            )

    async def _get(
        self,
        *,
        resource_type: Resource,
        resource_id: str,
        agency_id: str | None = None,
        version: str | None = None,
        key: dict[str, list[str]] | None = None,
        params: dict[str, str] | None = None,
        dsd: DataStructureDefinition | None = None,
        use_cache: bool = False,
        tofile: os.PathLike | IO | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Message:
        params = params or {}
        headers = await self._construct_headers({}, resource_type)
        if extra_headers:
            headers.update(extra_headers)
        req: PreparedRequest = self._sync_client.get(  # type: ignore[assignment]
            resource_type=resource_type,
            resource_id=resource_id,
            dry_run=True,
            headers=headers,
            key=key,
            params=params,
            dsd=dsd,
            **{k: v for k, v in [('agency_id', agency_id), ('version', version)] if v is not None},  # type: ignore[arg-type]
        )

        if use_cache:
            return await self._cache.get(
                key=req.url,  # type: ignore[arg-type]
                loader=lambda: self._fetch(req, tofile=tofile),
            )

        return await self._fetch(req, tofile=tofile)

    async def _construct_headers(
        self, headers: dict[str, str], resource: Resource
    ) -> dict[str, str]:
        if self._authorizer is None:
            auth_headers = {}
        else:
            auth_headers = await self._authorizer.get_authorization_headers()
        default_headers = self._sync_client.source.headers.get(resource.name, {})
        return {**default_headers, **self._static_headers, **auth_headers, **headers}

    async def _fetch(self, req: PreparedRequest, tofile: os.PathLike | IO | None = None) -> Message:
        httpx_response = await self._perform_request(req)
        response = self._convert_response(httpx_response, req)
        return self._parse_response(response, tofile=tofile)

    async def _perform_request(self, req: PreparedRequest, max_retries=3, delay=3) -> Response:
        attempts = 0
        start = time.monotonic()
        request_desc = f"{req.method} {req.url} body={req.body!r}"
        try:
            while True:
                attempts += 1
                try:
                    resp = await self._httpx_client.request(
                        method=req.method,  # type: ignore[arg-type]
                        url=req.url,  # type: ignore[arg-type]
                        headers=req.headers,
                        content=req.body,
                    )
                    if attempts == max_retries or resp.status_code < 500:
                        resp.raise_for_status()
                        return resp
                    logger.error(
                        f"SDMX server responded {resp.status_code} on attempt {attempts}/{max_retries}"
                        f" ({time.monotonic() - start:.1f}s elapsed): {resp.text}\n"
                        f"Retrying in {delay} seconds...\nRequest: {request_desc}"
                    )
                except httpx.TimeoutException as e:
                    elapsed = time.monotonic() - start
                    if attempts == max_retries:
                        msg = (
                            f"SDMX request to {req.method} {req.url} timed out after "
                            f"{attempts} attempt(s), waited {elapsed:.1f}s total"
                        )
                        logger.error(
                            f"{msg}\nCause: {self._describe_exception_chain(e)}\n"
                            f"Timeouts: {self._httpx_client.timeout!r}\nbody={req.body!r}"
                        )
                        raise SdmxRequestTimeoutError(msg) from e
                    logger.error(
                        f"SDMX request timed out on attempt {attempts}/{max_retries}"
                        f" ({elapsed:.1f}s elapsed): {self._describe_exception_chain(e)}\n"
                        f"Retrying in {delay} seconds...\nRequest: {request_desc}"
                    )
                except (httpx.NetworkError, httpx.RemoteProtocolError) as e:
                    # Transient transport failures (connection reset or refused, TLS handshake
                    # aborted by the peer, server closed a keep-alive connection, etc.).
                    if attempts == max_retries:
                        raise
                    logger.error(
                        f"SDMX request failed with a transport error on attempt {attempts}/{max_retries}"
                        f" ({time.monotonic() - start:.1f}s elapsed): {self._describe_exception_chain(e)}\n"
                        f"Retrying in {delay} seconds...\nRequest: {request_desc}"
                    )
                await asyncio.sleep(delay)
        except SdmxRequestTimeoutError:
            raise  # already logged above with a specific, actionable message
        except Exception as e:
            logger.exception(
                f"SDMX request failed after {attempts} attempt(s)"
                f" ({time.monotonic() - start:.1f}s elapsed): {self._describe_exception_chain(e)}\n"
                f"Request: {request_desc}"
            )
            raise

    @staticmethod
    def _describe_exception_chain(exc: BaseException) -> str:
        """Render the exception and its causes on one line, e.g.
        ``httpx.ConnectError <- httpcore.ConnectError <- anyio.EndOfStream``.

        httpx transport errors often have an empty message, and the actual reason
        (e.g. an ``ssl.SSLError`` or a connection closed by the peer) is only visible
        in the underlying cause.
        """
        parts = []
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            exc_type = type(current)
            module = exc_type.__module__.split('.')[0]
            name = (
                exc_type.__qualname__
                if module == 'builtins'
                else f"{module}.{exc_type.__qualname__}"
            )
            text = str(current)
            parts.append(f"{name}: {text}" if text else name)
            current = current.__cause__ or (
                None if current.__suppress_context__ else current.__context__
            )
        return " <- ".join(parts)

    @staticmethod
    def _convert_response(httpx_resp: httpx.Response, req: PreparedRequest) -> requests.Response:
        """Convert httpx response to requests response."""
        res = requests.Response()

        res.status_code = httpx_resp.status_code
        res.url = str(httpx_resp.url)
        res.headers.update(httpx_resp.headers.items())
        res._content = httpx_resp.content  # type: ignore[reportPrivateUsage]
        res.request = req

        return res

    @staticmethod
    def _parse_response(
        response: requests.Response, tofile: os.PathLike | IO | None = None, dsd: Any = None
    ) -> Message:
        response_content: io.IOBase = ResponseIO(response, tee=tofile)  # Select reader class
        try:
            reader_class = get_reader(response)
        except ValueError:
            logger.info(
                f"Failed to parse response:\n{response.status_code} {response.url}\nheaders={response.headers!r}"
            )
            raise ValueError(
                "Can't determine a reader for response content type "
                + repr(response.headers.get("content-type", None))
                + f" and url {response.url}"
            ) from None

        # Instantiate reader from class
        reader = reader_class()

        msg = reader.convert(response_content, structure=dsd)
        msg.response = response
        return msg

    @staticmethod
    def _get_flow_ref(resource_id: str, agency_id: str, version: str) -> str:
        return f"{agency_id},{resource_id},{version}"
