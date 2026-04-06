import io
from urllib.parse import urlencode

import httpx
import requests
from sdmx import Resource
from sdmx.message import DataMessage, StructureMessage
from sdmx.model.v21 import DataStructureDefinition
from sdmx.session import ResponseIO

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base.sdmx_schemas import SdmxPlusAvailabilityRequestBody
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiter
from statgpt.common.data.sdmx.v21.sdmx_client import AsyncSdmxClient
from statgpt.common.data.statgpt_sdmx_proxy.config import StatGptSdmxProxyDataSourceConfig
from statgpt.common.data.statgpt_sdmx_proxy.sdmx_schemas.structure_message import (
    ProxyAvailabilityResponseBody,
)
from statgpt.common.data.statgpt_sdmx_proxy.v30.reader import StatGptSdmxProxyDataReader


def proxy_structure_extra_headers(dsd_urn: str | None) -> dict[str, str] | None:
    """Headers for proxy structure requests (disambiguates artefacts that share identity)."""
    return {"X-Source-Artefact-Urn": dsd_urn} if dsd_urn else None


class AsyncStatGptSdmxProxyClient(AsyncSdmxClient):
    """Async client for StatGPT SDMX proxy (SDMX 3.0 API, SDMX-JSON parsed as SDMX 2.1 models)."""

    _DATA_PARAM_ALLOWLIST = {"startPeriod", "endPeriod", "firstNObservations", "lastNObservations"}
    _DATA_ACCEPT_DEFAULT = "application/vnd.sdmx.data+json;version=2.0.0"

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        config: StatGptSdmxProxyDataSourceConfig,
        auth_context: AuthContext,
        rate_limiter: SdmxRateLimiter,
    ) -> "AsyncStatGptSdmxProxyClient":
        return super().from_config(config, auth_context, rate_limiter)  # type: ignore[return-value]

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
                dsd=dsd,
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

    async def _proxy_available_constraint(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        use_cache: bool,
        key: dict[str, list[str]] | None,
        params: dict[str, str] | None,
        dsd: DataStructureDefinition | None,
    ) -> StructureMessage:
        """Fetch available constraints from the StatGPT SDMX proxy API."""
        url = self._build_url(
            path=f"/availability/dataflow/{agency_id}/{resource_id}/{version}", params=None
        )

        if use_cache:
            if key or params:
                raise ValueError("`use_cache` is not supported with `key` or `params`")

            cached_response = await self._get_item_from_cache(url)
            if cached_response is not None:
                return cached_response  # type: ignore[return-value]

        key = {} if key is None else key
        req_body_obj = SdmxPlusAvailabilityRequestBody.get_from(key=key, params=params)
        body = req_body_obj.model_dump(mode='json', exclude_none=True, by_alias=True)
        headers = {'accept': 'application/vnd.sdmx.structure+json;version=2.0.0'}
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

        if use_cache:
            self._sync_client.cache[url] = structure_msg

        return structure_msg

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
        key_segment = self._build_key_segment(key=key, dsd=dsd, require_dsd=True)
        filtered = self._filter_params(params, self._DATA_PARAM_ALLOWLIST)
        converted = self._convert_time_params(filtered)
        url = self._build_url(
            path=f"/data/dataflow/{agency_id}/{resource_id}/{version}/{key_segment}",
            params=converted,
        )

        response, req = await self._perform_get(url, Resource.data)
        if response is None:
            return DataMessage()

        httpx_response = response
        requests_response = self._convert_response(httpx_response, req)
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
        require_dsd: bool = False,
    ) -> str:
        if not key:
            return "*"
        if not dsd:
            if require_dsd:
                raise ValueError(
                    "Please provide a DataStructureDefinition (dsd) for proxy data requests."
                )
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
                parts.append("")
            else:
                parts.append("+".join(values))
        return ".".join(parts).rstrip(".") or "*"

    def _build_url(self, *, path: str, params: dict[str, str | list[str]] | None) -> str:
        url = f"{self._sync_client.source.url}{path}"
        if params:
            return f"{url}?{urlencode(params, doseq=True)}"
        return url

    @staticmethod
    def _filter_params(params: dict[str, str] | None, allowlist: set[str]) -> dict[str, str] | None:
        if not params:
            return None
        return {k: v for k, v in params.items() if k in allowlist}

    @staticmethod
    def _convert_time_params(
        params: dict[str, str] | None,
    ) -> dict[str, str | list[str]] | None:
        """Convert startPeriod/endPeriod to SDMX 3.0 c[TIME_PERIOD] filter syntax."""
        if not params:
            return None
        result: dict[str, str | list[str]] = {}
        time_filters: list[str] = []
        for k, v in params.items():
            if k == "startPeriod":
                time_filters.append(f"ge:{v}")
            elif k == "endPeriod":
                time_filters.append(f"le:{v}")
            else:
                result[k] = v
        if time_filters:
            result["c[TIME_PERIOD]"] = time_filters
        return result or None

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
