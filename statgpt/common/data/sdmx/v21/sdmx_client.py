import asyncio
import io
import os
from typing import IO, Any, Mapping, Protocol, TypeGuard

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


class _ProxySdmxConfig(Protocol):
    headers: dict[str, dict[str, str]] | None
    versions: list[str] | None


def _is_proxy_config(config: SdmxDataSourceConfig) -> TypeGuard[_ProxySdmxConfig]:
    return hasattr(config, "headers") or hasattr(config, "versions")


def init_sdmx(config: SdmxDataSourceConfig):
    source_dict = config.sdmx_config.to_sdmx1_dict()
    if not source_dict.get("versions"):
        source_dict["versions"] = {"2.1"}
        logger.warning(
            f"SDMX source '{config.get_id()}' has no versions configured; " "defaulting to {'2.1'}."
        )
    if _is_proxy_config(config):
        if config.headers:
            source_dict["headers"] = config.headers
        if config.versions:
            source_dict["versions"] = config.versions
    sdmx.add_source(source_dict, override=True)


class AsyncSdmxClient:
    """Async client for interacting with the SDMX API."""

    _LOADING: set[str] = set()
    """Urls of SDMX requests that are currently being loaded."""
    _SENSITIVE_HEADER_KEYS = {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "x-auth-token",
        "api-key",
        "apikey",
    }

    @classmethod
    def from_config(
        cls, config: SdmxDataSourceConfig, auth_context: AuthContext, rate_limiter: SdmxRateLimiter
    ) -> "AsyncSdmxClient":
        """Initialize the client from a configuration object."""

        init_sdmx(config)
        sync_client = Client(config.get_id())
        httpx_client = cls._create_httpx_client()

        return cls(sync_client, httpx_client, None, rate_limiter)

    def __init__(
        self,
        sync_client: Client,
        httpx_client: httpx.AsyncClient,
        authorizer: IAuthorizer | None,
        rate_limiter: SdmxRateLimiter,
    ):
        self._sync_client = sync_client
        self._httpx_client = httpx_client
        self._authorizer = authorizer
        self._rate_limiter = rate_limiter

    @classmethod
    def _redact_headers(cls, headers: Mapping[str, str] | None) -> dict[str, str]:
        if not headers:
            return {}
        redacted: dict[str, str] = {}
        for key, value in headers.items():
            if key.lower() in cls._SENSITIVE_HEADER_KEYS:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = value
        return redacted

    @staticmethod
    def _preview_body(body: Any, limit: int = 1000) -> str:
        if body is None:
            return ""
        if isinstance(body, (bytes, bytearray)):
            text = body[:limit].decode("utf-8", errors="replace")
        else:
            text = str(body)
        if len(text) > limit:
            return f"{text[:limit]}...(truncated)"
        return text

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
        self, *, agency_id: str, resource_id: str, version: str, use_cache: bool = False
    ) -> StructureMessage:
        return await self._get_structure(  # type: ignore[return-value]
            resource_type=Resource.conceptscheme,
            agency_id=agency_id,
            resource_id=resource_id,
            version=version,
            use_cache=use_cache,
        )

    async def codelist(
        self, *, agency_id: str, resource_id: str, version: str, use_cache: bool = False
    ) -> StructureMessage:
        return await self._get_structure(  # type: ignore[return-value]
            resource_type=Resource.codelist,
            agency_id=agency_id,
            resource_id=resource_id,
            version=version,
            use_cache=use_cache,
        )

    async def hierarchicalcodelist(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        params: dict[str, str] | None = None,
        use_cache: bool = False,
    ) -> StructureMessage:
        return await self._get_structure(  # type: ignore[return-value]
            resource_type=Resource.hierarchicalcodelist,
            agency_id=agency_id,
            resource_id=resource_id,
            version=version,
            params=params,
            use_cache=use_cache,
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
    ) -> Message:
        params = params or {}
        headers = await self._construct_headers({}, resource_type)
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
            cached_response = await self._get_item_from_cache(req.url)  # type: ignore[arg-type]
            if cached_response is not None:
                return cached_response

        httpx_response = await self._perform_request(req)
        response = self._convert_response(httpx_response, req)
        msg = self._parse_response(response, tofile=tofile)

        if use_cache:
            self._sync_client.cache[req.url] = msg  # type: ignore[index]

        return msg

    async def _construct_headers(
        self, headers: dict[str, str], resource: Resource
    ) -> dict[str, str]:
        if self._authorizer is None:
            auth_headers = {}
        else:
            auth_headers = await self._authorizer.get_authorization_headers()
        default_headers = self._sync_client.source.headers.get(resource.name, {})
        return {**default_headers, **auth_headers, **headers}

    async def _get_item_from_cache(self, url: str, timeout: int = 120) -> Message | None:
        waited: float = 0.0
        while url in self._LOADING and waited < timeout:
            # Wait for the request to finish if it's currently being loaded
            await asyncio.sleep(0.5)
            waited += 0.5

        return self._sync_client.cache.get(url)

    async def _perform_request(self, req: PreparedRequest, max_retries=3, delay=3) -> Response:
        self._LOADING.add(req.url)  # type: ignore[arg-type]
        attempts = 0
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
                    else:
                        logger.error(
                            f"Server failed to respond after {attempts} attempts: {resp.status_code} {resp.text}\n"
                            f"Retrying in {delay} seconds...\nRequest: {req.method} {req.url} body={req.body!r}"
                        )
                        await asyncio.sleep(delay)
                except httpx.ConnectTimeout:
                    if attempts == max_retries:
                        logger.exception(
                            f"Connection timed out after {attempts} attempts: "
                            f"{req.method} {req.url} body={req.body!r}"
                        )
                        raise
                    else:
                        logger.error(
                            f"Connection timed out after {attempts} attempts. Retrying in {delay} seconds..."
                            f"\nRequest: {req.method} {req.url} body={req.body!r}\n"
                        )
                        await asyncio.sleep(delay)
        except Exception:
            logger.exception(
                f"Server failed to respond, after {attempts} attempts: "
                f"{req.method} {req.url} body={req.body!r}"
            )
            raise
        finally:
            self._LOADING.discard(req.url)  # type: ignore[arg-type]

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

    @staticmethod
    def _create_httpx_client(*, headers: dict[str, str] | None = None) -> httpx.AsyncClient:
        """Create an HTTPX client with default settings."""
        return httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=45.0), headers=headers)
