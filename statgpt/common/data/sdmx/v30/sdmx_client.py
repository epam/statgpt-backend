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
from statgpt.common.data.quanthub.sdmx_schemas.v30.structure_message import (
    ProxyAvailabilityResponseBody,
)
from statgpt.common.data.quanthub.v21.qh_sdmx_client import AsyncQuanthubClient
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiter
from statgpt.common.data.sdmx.v30.config import ProxySdmx30DataSourceConfig
from statgpt.common.data.sdmx.v30.reader import ProxyDataReader

class AsyncProxySdmxClient(AsyncQuanthubClient):
    """Async client for Proxy SDMX 3.0 sources based on QuantHub client behavior."""
    _DATA_PARAM_ALLOWLIST = {"startPeriod", "endPeriod", "firstNObservations", "lastNObservations"}
    _DATA_ACCEPT_DEFAULT = "application/vnd.sdmx.data+json;version=2.0.0"

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        config: ProxySdmx30DataSourceConfig,
        auth_context: AuthContext,
        rate_limiter: SdmxRateLimiter,
    ) -> "AsyncProxySdmxClient":
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

        if self._availability_via_post_url:
            return await self._qh_available_constraint(
                agency_id=agency_id,
                resource_id=resource_id,
                version=version,
                use_cache=use_cache,
                key=key,
                params=params,
            )
        else:
            return await self._proxy_available_constraint(
                resource_id=resource_id,
                agency_id=agency_id,
                version=version,
                key=key,
                params=params,
                use_cache=use_cache,
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
        """Fetch available constraints from the QuantHub SDMX API."""
        key_segment = self._build_key_segment(key=key, dsd=dsd)
        url = self._build_url(
            path=f"/availability/dataflow/{agency_id}/{resource_id}/{version}/{key_segment}/*",
            params=params,
        )

        if use_cache:
            if key or params:
                raise ValueError("`use_cache` is not supported with `key` or `params`")

            cached_response = await self._get_item_from_cache(url)
            if cached_response is not None:
                return cached_response  # type: ignore[return-value]

        response, _ = await self._perform_get(
            url,
            Resource.availableconstraint,
            limiter=self._rate_limiter.availability_limiter,
        )
        if response is None:
            return StructureMessage()  # Return empty StructureMessage on bad request

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
        url = self._build_url(
            path=f"/data/dataflow/{agency_id}/{resource_id}/{version}/{key_segment}",
            params=self._filter_params(params, self._DATA_PARAM_ALLOWLIST),
        )

        response, req = await self._perform_get(
            url,
            Resource.data,
            limiter=self._rate_limiter.data_limiter,
        )
        if response is None:
            return DataMessage()

        httpx_response = response
        requests_response = self._convert_response(httpx_response, req)
        try:
            response_content: io.IOBase = ResponseIO(response)
            msg = ProxyDataReader().convert(response_content, structure=dsd)
            msg.response = requests_response
        except Exception:
            logger.error(
                "Failed to parse proxy SDMX response: url=%r content-type=%r body=%r",
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
        return ".".join(parts) or "*"

    def _build_url(self, *, path: str, params: dict[str, str] | None) -> str:
        url = f"{self._sync_client.source.url}{path}"
        if params:
            return f"{url}?{urlencode(params, doseq=True)}"
        return url

    @staticmethod
    def _filter_params(
        params: dict[str, str] | None, allowlist: set[str]
    ) -> dict[str, str] | None:
        if not params:
            return None
        return {k: v for k, v in params.items() if k in allowlist}

    async def _perform_get(
        self,
        url: str,
        resource: Resource,
        *,
        limiter,
    ) -> tuple[httpx.Response | None, requests.PreparedRequest]:
        headers = await self._construct_headers({}, resource)
        if resource == Resource.data and "accept" not in {key.lower() for key in headers}:
            headers["accept"] = self._DATA_ACCEPT_DEFAULT
        req = requests.Request(method="GET", url=url, headers=headers).prepare()

        try:
            async with limiter():
                response = await self._perform_request(req)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [400, 404]:
                logger.error(f"Bad request for URL {url!r}: {e.response.text}")
                logger.info(f"Request body: {req.body!r}")
                return None, req
            raise
        return response, req
