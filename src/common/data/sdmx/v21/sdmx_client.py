import asyncio
import io
import os
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

from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.data.sdmx.common.authorizer import IAuthorizer
from common.data.sdmx.common.config import SdmxDataSourceConfig


def init_sdmx(config: SdmxDataSourceConfig):
    sdmx.add_source(config.sdmx_config.to_sdmx1_dict(), override=True)


class AsyncSdmxClient:
    """Async client for interacting with the SDMX API."""

    _LOADING: set[str] = set()
    """Urls of SDMX requests that are currently being loaded."""

    @classmethod
    def from_config(
        cls, config: SdmxDataSourceConfig, auth_context: AuthContext
    ) -> "AsyncSdmxClient":
        """Initialize the client from a configuration object."""

        init_sdmx(config)
        sync_client = Client(config.get_id())
        httpx_client = cls._create_httpx_client()

        return cls(sync_client, httpx_client, None)

    def __init__(
        self,
        sync_client: Client,
        httpx_client: httpx.AsyncClient,
        authorizer: IAuthorizer | None,
    ):
        self._sync_client = sync_client
        self._httpx_client = httpx_client
        self._authorizer = authorizer

    async def dataflow(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        params: dict[str, Any],
        use_cache: bool = False,
    ) -> StructureMessage:
        return await self._get(  # type: ignore[return-value]
            resource_type=Resource.dataflow,
            agency_id=agency_id,
            resource_id=resource_id,
            version=version,
            params=params,
            use_cache=use_cache,
        )

    async def conceptscheme(
        self, *, agency_id: str, resource_id: str, version: str, use_cache: bool = False
    ) -> StructureMessage:
        return await self._get(  # type: ignore[return-value]
            resource_type=Resource.conceptscheme,
            agency_id=agency_id,
            resource_id=resource_id,
            version=version,
            use_cache=use_cache,
        )

    async def codelist(
        self, *, agency_id: str, resource_id: str, version: str, use_cache: bool = False
    ) -> StructureMessage:
        return await self._get(  # type: ignore[return-value]
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
        return await self._get(  # type: ignore[return-value]
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
        return await self._get(  # type: ignore[return-value]
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

        return await self._get(  # type: ignore[return-value]
            resource_type=Resource.data,
            resource_id=flow_ref,
            key=key,
            params=params,
            dsd=dsd,
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
        try:
            attempts = 0
            while True:
                attempts += 1
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
            raise ValueError(
                "can't determine a reader for response content type "
                + repr(response.headers.get("content-type", None))
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
        return httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0), headers=headers)
