import httpx
import requests
from sdmx import Client, Resource
from sdmx.message import StructureMessage
from sdmx.model.v21 import DataStructureDefinition

from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.data.quanthub.config import QuanthubSdmxDataSourceConfig
from common.data.sdmx.v21.sdmx_client import AsyncSdmxClient, init_sdmx

from .authorizer import QuanthubAuthorizer, QuanthubAuthorizerFactory
from .qh_sdmx_30_schemas import (
    QhAnnotation,
    QhAvailabilityRequestBody,
    QhAvailabilityResponseBody,
    QhDataflowMessage,
)
from .sdmx_extensions import init_qh_sdmx_extensions


class AsyncQuanthubClient(AsyncSdmxClient):
    """Async client for interacting with the QuantHub SDMX API.

    Contains methods unique to QuantHub, such as fetching dynamic annotations.
    """

    _annotation_cache: dict[str, list[QhAnnotation]] = {}

    @classmethod
    def from_config(  # type: ignore[override]
        cls, config: QuanthubSdmxDataSourceConfig, auth_context: AuthContext
    ) -> "AsyncQuanthubClient":
        """Initialize the client from a configuration object."""

        init_qh_sdmx_extensions()
        init_sdmx(config)
        sync_client = Client(config.get_id())

        headers: dict[str, str] = {}
        if config.api_key and config.api_key_header:
            headers[config.api_key_header] = config.get_api_key().get_secret_value()
        httpx_client = cls._create_httpx_client(headers=headers)

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
            availability_via_post_url=config.get_availability_via_post_url(),
        )

    def __init__(
        self,
        sync_client: Client,
        httpx_client: httpx.AsyncClient,
        authorizer: QuanthubAuthorizer | None,
        annotations_url: str | None,
        availability_via_post_url: str | None,
    ):
        super().__init__(sync_client=sync_client, httpx_client=httpx_client, authorizer=authorizer)
        self._annotations_url = annotations_url
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

    async def dynamic_dataflow_annotations(
        self, *, agency_id: str, resource_id: str, version: str
    ) -> list[QhAnnotation]:
        """Fetch dynamic annotations for a given dataflow."""

        if not self._annotations_url:
            return []

        url = f"{self._annotations_url}/structure/dataflow/{agency_id}/{resource_id}/{version}"
        if url in self._annotation_cache:
            return self._annotation_cache[url]

        headers = {}
        if self._authorizer is not None:
            headers = await self._authorizer.get_authorization_headers()

        params = {
            "references": "none",
            # "forceDataflowDynamicAnnotations": "true",
        }

        resp = await self._httpx_client.get(url, headers=headers, params=params)
        resp.raise_for_status()

        response_data = QhDataflowMessage.model_validate(resp.json())
        if len(response_data.data.dataflows) != 1:
            logger.info(f"GET {resp.request.url!r}. Response content:\n{resp.text}")
            raise ValueError(
                f"Expected exactly one dataflow in response, got {len(response_data.data.dataflows)}"
            )
        self._annotation_cache[url] = response_data.data.dataflows[0].annotations
        return self._annotation_cache[url]

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
        """Fetch available constraints from the QuantHub SDMX API."""

        url = f"{self._availability_via_post_url}/availability/dataflow/{agency_id}/{resource_id}/{version}"

        if use_cache:
            if key or params:
                raise ValueError("`use_cache` is not supported with `key` or `params`")

            cached_response = await self._get_item_from_cache(url)
            if cached_response is not None:
                return cached_response  # type: ignore[return-value]

        headers = await self._construct_headers(
            {'accept': 'application/json'}, Resource.availableconstraint
        )
        key = {} if key is None else key
        req_body_obj = QhAvailabilityRequestBody.get_from(key=key, params=params)

        req = requests.Request(
            method="POST",
            url=url,
            headers=headers,
            json=req_body_obj.model_dump(mode='json', exclude_none=True, by_alias=True),
        ).prepare()
        response = await self._perform_request(req)

        resp_body_obj = QhAvailabilityResponseBody.model_validate(response.json())
        structure_msg = resp_body_obj.to_sdmx1()

        if use_cache:
            self._sync_client.cache[url] = structure_msg

        return structure_msg
