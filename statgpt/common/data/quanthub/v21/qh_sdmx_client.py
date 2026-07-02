import httpx
import requests
from sdmx import Client, Resource
from sdmx.message import StructureMessage
from sdmx.model.v21 import DataStructureDefinition

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base.sdmx_schemas import (
    Sdmx30AnnotationModel,
    SdmxPlusAvailabilityRequestBody,
)
from statgpt.common.data.quanthub.config import QuanthubSdmxDataSourceConfig
from statgpt.common.data.quanthub.sdmx_schemas.v30 import (
    QhAvailabilityResponseBody,
    QhDataflowMessage,
    QhDataMessage,
)
from statgpt.common.data.sdmx.v21.ratelimiter import SdmxRateLimiter
from statgpt.common.data.sdmx.v21.sdmx_client import AsyncSdmxClient, init_sdmx
from statgpt.common.utils import TtlCache
from statgpt.common.utils.http_pool import get_shared_sdmx_http_client

from .attributes_parser import AttributesParser
from .authorizer import QuanthubAuthorizer, QuanthubAuthorizerFactory
from .sdmx_extensions import init_qh_sdmx_extensions


class AsyncQuanthubClient(AsyncSdmxClient):
    """Async client for interacting with the QuantHub SDMX API.

    Contains methods unique to QuantHub, such as fetching dynamic annotations.
    """

    _annotation_cache: TtlCache[list[Sdmx30AnnotationModel]] = TtlCache()
    _attributes_cache: TtlCache[dict[str, str | None]] = TtlCache()

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        config: QuanthubSdmxDataSourceConfig,
        auth_context: AuthContext,
        rate_limiter: SdmxRateLimiter,
    ) -> "AsyncQuanthubClient":
        """Initialize the client from a configuration object."""

        init_qh_sdmx_extensions()
        init_sdmx(config)
        sync_client = Client(config.get_id())

        headers: dict[str, str] = {}
        if config.api_key and config.api_key_header:
            headers[config.api_key_header] = config.get_api_key().get_secret_value()
        # Static api-key headers stay client-level: they are part of the shared-pool key,
        # so rotating the key transparently yields a fresh client.
        httpx_client = get_shared_sdmx_http_client(config.get_id(), headers=headers)

        authorizer = None
        if config.auth_enabled:
            if config.auth_config is None:
                raise ValueError("Authorization is enabled but no auth_config provided.")

            authorizer = QuanthubAuthorizerFactory().create(auth_context, config.auth_config)

        return cls(
            sync_client,
            httpx_client,
            authorizer=authorizer,
            annotations_url=config.get_annotations_url(),
            attributes_url=config.get_attributes_url(),
            availability_via_post_url=config.get_availability_via_post_url(),
            rate_limiter=rate_limiter,
        )

    def __init__(
        self,
        sync_client: Client,
        httpx_client: httpx.AsyncClient,
        authorizer: QuanthubAuthorizer | None,
        annotations_url: str | None,
        attributes_url: str | None,
        availability_via_post_url: str | None,
        rate_limiter: SdmxRateLimiter,
    ):
        super().__init__(
            sync_client=sync_client,
            httpx_client=httpx_client,
            authorizer=authorizer,
            rate_limiter=rate_limiter,
        )
        self._annotations_url = annotations_url
        self._attributes_url = attributes_url
        self._availability_via_post_url = availability_via_post_url

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
            return await super().availableconstraint(
                agency_id=agency_id,
                resource_id=resource_id,
                version=version,
                use_cache=use_cache,
                key=key,
                params=params,
                dsd=dsd,
            )

    async def dataset_level_attributes(
        self, *, agency_id: str, resource_id: str, version: str
    ) -> dict[str, str | None]:
        """Fetch dataset-level attributes and their values for a given dataflow.

        Returns:
            A dictionary mapping attribute IDs to their values.
        """
        if not self._attributes_url:
            return {}

        url = f"{self._attributes_url}/data/dataflow/{agency_id}/{resource_id}/{version}/*"
        if (item := self._attributes_cache.get(url)) is not None:
            return item

        headers = {}
        if self._authorizer is not None:
            headers = await self._authorizer.get_authorization_headers()

        params: dict[str, str | int] = {"attributes": "dataset", "measures": "none", "limit": 1}

        async with self._rate_limiter.structure_limiter():
            resp = await self._httpx_client.get(url, headers=headers, params=params)
        resp.raise_for_status()

        response_data = QhDataMessage.model_validate(resp.json())
        attributes = AttributesParser.parse(response_data)

        self._attributes_cache.set(url, attributes)
        return attributes

    async def dynamic_dataflow_annotations(
        self, *, agency_id: str, resource_id: str, version: str
    ) -> list[Sdmx30AnnotationModel]:
        """Fetch dynamic annotations for a given dataflow."""

        if not self._annotations_url:
            return []

        url = f"{self._annotations_url}/structure/dataflow/{agency_id}/{resource_id}/{version}"
        if (item := self._annotation_cache.get(url)) is not None:
            return item

        headers = {}
        if self._authorizer is not None:
            headers = await self._authorizer.get_authorization_headers()

        params = {
            "references": "none",
            # "forceDataflowDynamicAnnotations": "true",
        }

        async with self._rate_limiter.structure_limiter():
            resp = await self._httpx_client.get(url, headers=headers, params=params)
        resp.raise_for_status()

        response_data = QhDataflowMessage.model_validate(resp.json())
        if len(response_data.data.dataflows) != 1:
            logger.info(f"GET {resp.request.url!r}. Response content:\n{resp.text}")
            raise ValueError(
                f"Expected exactly one dataflow in response, got {len(response_data.data.dataflows)}"
            )
        self._annotation_cache.set(url, response_data.data.dataflows[0].annotations)
        return response_data.data.dataflows[0].annotations

    async def _fetch_qh_available_constraint(
        self,
        *,
        url: str,
        key: dict[str, list[str]] | None,
        params: dict[str, str] | None,
    ) -> StructureMessage:
        """Fetch available constraints from the QuantHub SDMX API."""
        headers = await self._construct_headers(
            {'accept': 'application/json'}, Resource.availableconstraint
        )
        resolved_key = {} if key is None else key
        req_body_obj = SdmxPlusAvailabilityRequestBody.get_from(key=resolved_key, params=params)

        req = requests.Request(
            method="POST",
            url=url,
            headers=headers,
            json=req_body_obj.model_dump(mode='json', exclude_none=True, by_alias=True),
        ).prepare()

        try:
            async with self._rate_limiter.availability_limiter():
                response = await self._perform_request(req)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [400, 404]:
                logger.error(f"Bad request for URL {url!r}: {e.response.text}")
                logger.info(f"Request body: {req.body!r}")
                return StructureMessage()  # Return empty StructureMessage on bad request
            raise

        resp_body_obj = QhAvailabilityResponseBody.model_validate(response.json())
        structure_msg = resp_body_obj.to_sdmx1()
        return structure_msg

    async def _qh_available_constraint(
        self,
        *,
        agency_id: str,
        resource_id: str,
        version: str,
        use_cache: bool,
        key: dict[str, list[str]] | None,
        params: dict[str, str] | None,
    ) -> StructureMessage:
        url = f"{self._availability_via_post_url}/availability/dataflow/{agency_id}/{resource_id}/{version}"

        if use_cache:
            if key or params:
                raise ValueError("`use_cache` is not supported with `key` or `params`")
            return await self._cache.get(  # type: ignore[return-value]
                key=url,
                loader=lambda: self._fetch_qh_available_constraint(url=url, key=None, params=None),
            )

        return await self._fetch_qh_available_constraint(url=url, key=key, params=params)
