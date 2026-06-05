import io
from typing import cast
from urllib.parse import urlencode

import httpx
import requests
from sdmx import Resource
from sdmx.message import DataMessage, StructureMessage
from sdmx.model.v21 import AttributeValue, Code, DataStructureDefinition
from sdmx.session import ResponseIO

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base.sdmx_schemas import SdmxPlusAvailabilityRequestBody
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiter
from statgpt.common.data.sdmx.v21.sdmx_client import AsyncSdmxClient
from statgpt.common.data.statgpt_sdmx_proxy.config import StatGptSdmxProxyDataSourceConfig
from statgpt.common.data.statgpt_sdmx_proxy.sdmx_schemas.structure_message import (
    ProxyAgencySchemeResponseBody,
    ProxyAvailabilityResponseBody,
)
from statgpt.common.data.statgpt_sdmx_proxy.v30.reader import StatGptSdmxProxyDataReader
from statgpt.common.utils import TtlCache


def _sdmx_attribute_value_display(av: AttributeValue) -> str | None:
    val = av.value
    if val is None:
        return None
    if isinstance(val, Code):
        return val.id
    return str(val)


def _dataset_level_attribute_map_from_data_message(msg: DataMessage) -> dict[str, str | None]:
    """Extract dataset-level attributes from a DataMessage as a flat {id: display_value} dict."""
    if len(msg.data) != 1:
        return {}
    ds = msg.data[0]
    return {attr_id: _sdmx_attribute_value_display(av) for attr_id, av in ds.attrib.items()}


def proxy_structure_extra_headers(dsd_urn: str | None) -> dict[str, str] | None:
    """Headers for proxy structure requests (disambiguates artefacts that share identity)."""
    return {"X-Source-Artefact-Urn": dsd_urn} if dsd_urn else None


# TODO: move this to the data source config
_SDMX_v30_STRUCTURE_ACCEPT_HEADER = "application/vnd.sdmx.structure+json;version=2.0.0"


class AsyncStatGptSdmxProxyClient(AsyncSdmxClient):
    """Async client for StatGPT SDMX proxy (SDMX 3.0 API, SDMX-JSON parsed as SDMX 2.1 models)."""

    _DATA_ACCEPT_DEFAULT = "application/vnd.sdmx.data+json;version=2.0.0"
    _attributes_cache: TtlCache[dict[str, str | None]] = TtlCache()

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        config: StatGptSdmxProxyDataSourceConfig,
        auth_context: AuthContext,
        rate_limiter: SdmxRateLimiter,
    ) -> "AsyncStatGptSdmxProxyClient":
        return super().from_config(config, auth_context, rate_limiter)  # type: ignore[return-value]

    async def agencyscheme(  # type: ignore[override]
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        use_cache: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> StructureMessage:
        """Fetch agencyschemes from the proxy.

        sdmx1 does not register a reader for SDMX-JSON 2.0.0 structure responses,
        so this method bypasses ``_parse_response`` and validates the body with
        :class:`ProxyAgencySchemeResponseBody` before converting to sdmx1 models.
        """
        url = self._build_url(
            path=f"/structure/agencyscheme/{agency_id}/{resource_id}/{version}",
            params=None,
        )
        headers = {"accept": _SDMX_v30_STRUCTURE_ACCEPT_HEADER, **(extra_headers or {})}

        if use_cache:
            return cast(
                StructureMessage,
                await self._cache.get(
                    key=url,
                    loader=lambda: self._fetch_proxy_agencyscheme(url=url, headers=headers),
                ),
            )
        return await self._fetch_proxy_agencyscheme(url=url, headers=headers)

    async def _fetch_proxy_agencyscheme(
        self, *, url: str, headers: dict[str, str]
    ) -> StructureMessage:
        req = requests.Request(method="GET", url=url, headers=headers).prepare()
        async with self._rate_limiter.structure_limiter():
            response = await self._perform_request(req)
        body = ProxyAgencySchemeResponseBody.model_validate(response.json())
        return body.to_sdmx1()

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
        async with self._rate_limiter.availability_limiter():
            return await self._proxy_available_constraint(
                agency_id=agency_id,
                resource_id=resource_id,
                version=version,
                use_cache=use_cache,
                key=key,
                params=params,
            )

    async def data(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        key: dict[str, list[str]] | None,
        params: dict[str, str] | None,
        dsd: DataStructureDefinition | None,
    ) -> DataMessage:
        async with self._rate_limiter.data_limiter():
            return await self._proxy_data(
                agency_id=agency_id,
                resource_id=resource_id,
                version=version,
                key=key,
                params=params,
                dsd=dsd,
            )

    async def dataset_level_attributes(
        self, *, agency_id: str, resource_id: str, version: str
    ) -> dict[str, str | None]:
        """Fetch dataset-level attributes and their resolved values for a dataflow."""
        url = self._build_url(
            path=f"/data/dataflow/{agency_id}/{resource_id}/{version}/*",
            params={"attributes": "dataset", "measures": "none", "limit": "1"},
        )
        if (item := self._attributes_cache.get(url)) is not None:
            return item

        async with self._rate_limiter.data_limiter():
            response, req = await self._perform_get(url, Resource.data)
            if response is None:
                return {}

            requests_response = self._convert_response(response, req)
            response_content: io.IOBase = ResponseIO(response)
            try:
                msg = StatGptSdmxProxyDataReader().convert(response_content, structure=None)
                msg.response = requests_response
            except Exception:
                logger.error(
                    "Failed to parse dataset-level attributes response: url=%r content-type=%r body=%r",
                    requests_response.url,
                    requests_response.headers.get("content-type"),
                    requests_response.text[:1000],
                )
                result = {}
            else:
                if isinstance(msg, DataMessage):
                    result = _dataset_level_attribute_map_from_data_message(msg)
                else:
                    logger.error(
                        "Unexpected response message type: %s for URL %r",
                        type(msg).__name__,
                        req.url,
                    )
                    result = {}

        self._attributes_cache.set(url, result)
        return result

    async def _fetch_proxy_available_constraint(
        self,
        *,
        url: str,
        key: dict[str, list[str]] | None,
        params: dict[str, str] | None,
    ) -> StructureMessage:
        """Fetch available constraints from the StatGPT SDMX proxy API."""
        resolved_key = {} if key is None else key
        req_body_obj = SdmxPlusAvailabilityRequestBody.get_from(key=resolved_key, params=params)
        body = req_body_obj.model_dump(mode='json', exclude_none=True, by_alias=True)
        headers = {'accept': _SDMX_v30_STRUCTURE_ACCEPT_HEADER}
        req = requests.Request(
            method="POST",
            url=url,
            headers=headers,
            json=body,
        ).prepare()

        try:
            response = await self._perform_request(req)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [400, 404]:
                logger.error(f"Bad request for URL {url!r}: {e.response.text}")
                logger.info(f"Request body: {req.body!r}")
                return StructureMessage()  # Return empty StructureMessage on bad request
            raise

        resp_body_obj = ProxyAvailabilityResponseBody.model_validate(response.json())
        structure_msg = resp_body_obj.to_sdmx1()
        return structure_msg

    async def _proxy_available_constraint(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        use_cache: bool,
        key: dict[str, list[str]] | None,
        params: dict[str, str] | None,
    ) -> StructureMessage:
        url = self._build_url(
            path=f"/availability/dataflow/{agency_id}/{resource_id}/{version}", params=None
        )

        if use_cache:
            if key or params:
                raise ValueError("`use_cache` is not supported with `key` or `params`")
            return cast(
                StructureMessage,
                await self._cache.get(
                    key=url,
                    loader=lambda: self._fetch_proxy_available_constraint(
                        url=url, key=None, params=None
                    ),
                ),
            )

        return await self._fetch_proxy_available_constraint(url=url, key=key, params=params)

    async def _proxy_data(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        key: dict[str, list[str]] | None,
        params: dict[str, str] | None,
        dsd: DataStructureDefinition | None,
    ) -> DataMessage:
        key_segment = self._build_key_segment(key=key, dsd=dsd)
        url = self._build_url(
            path=f"/data/dataflow/{agency_id}/{resource_id}/{version}/{key_segment}",
            params=params,
        )

        response, req = await self._perform_get(url, Resource.data)
        if response is None:
            return DataMessage()

        requests_response = self._convert_response(response, req)
        try:
            response_content: io.IOBase = ResponseIO(response)
            msg = StatGptSdmxProxyDataReader().convert(response_content, structure=dsd)
            msg.response = requests_response
        except Exception:
            logger.error(
                "Failed to parse StatGPT SDMX proxy response: url=%r content-type=%r body=%r",
                requests_response.url,
                requests_response.headers.get("content-type"),
                requests_response.text[:1000],
            )
            raise
        if not isinstance(msg, DataMessage):
            raise ValueError(
                f"Unexpected response message type: {type(msg).__name__} for URL {req.url!r}"
            )
        return msg

    def _build_key_segment(
        self,
        *,
        key: dict[str, list[str]] | None,
        dsd: DataStructureDefinition | None,
    ) -> str:
        key = {k: v for k, v in (key or {}).items() if v}  # Filter out empty values
        if not key:
            return "*"
        if not dsd:
            raise ValueError("Please provide a DataStructureDefinition (dsd) when using `key`.")
        dim_ids = [
            dim.id
            for dim in dsd.dimensions.components
            if not getattr(dim, "is_time_dimension", False) and dim.id != "TIME_PERIOD"
        ]
        parts = []
        for dim_id in dim_ids:
            values = key.get(dim_id)
            if not values:
                parts.append("*")
            else:
                parts.append("+".join(values))
        return ".".join(parts)

    def _build_url(self, *, path: str, params: dict[str, str] | None) -> str:
        url = f"{self._sync_client.source.url}{path}"
        if params:
            return f"{url}?{urlencode(params)}"
        return url

    async def _perform_get(
        self, url: str, resource: Resource
    ) -> tuple[httpx.Response | None, requests.PreparedRequest]:
        headers = await self._construct_headers({}, resource)
        if resource == Resource.data and "accept" not in {key.lower() for key in headers}:
            headers["accept"] = self._DATA_ACCEPT_DEFAULT
        req = requests.Request(method="GET", url=url, headers=headers).prepare()

        try:
            response = await self._perform_request(req)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [400, 404]:
                logger.error(f"Bad request for URL {url!r}: {e.response.text}")
                logger.info(f"Request body: {req.body!r}")
                return None, req
            raise
        return response, req
