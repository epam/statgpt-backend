"""Tests for the request `AsyncQuanthubClient.dynamic_dataflow_annotations` sends."""

import httpx

from statgpt.common.data.quanthub.v21.qh_sdmx_client import AsyncQuanthubClient
from statgpt.common.data.sdmx.v21.ratelimiter import NoOpRateLimiter, SdmxRateLimiter

_ANNOTATIONS_URL = "https://annotations.example.invalid/sdmx/3.0"


def _client(handler) -> AsyncQuanthubClient:
    return AsyncQuanthubClient(
        sync_client=None,  # type: ignore[arg-type]
        httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        authorizer=None,
        annotations_url=_ANNOTATIONS_URL,
        attributes_url=None,
        availability_via_post_url=None,
        rate_limiter=SdmxRateLimiter(NoOpRateLimiter(), NoOpRateLimiter(), NoOpRateLimiter()),
    )


async def test_dynamic_annotations_are_forced() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = {"data": {"dataflows": [{"annotations": [{"id": "dataUpdatedAt", "value": "x"}]}]}}
        return httpx.Response(200, json=body)

    annotations = await _client(handler).dynamic_dataflow_annotations(
        agency_id="AGENCY", resource_id="FORCED_ANNOTATIONS", version="1.0.0"
    )

    assert [a.id for a in annotations] == ["dataUpdatedAt"]
    assert len(requests) == 1
    assert requests[0].url.path.endswith("/structure/dataflow/AGENCY/FORCED_ANNOTATIONS/1.0.0")
    assert dict(requests[0].url.params) == {
        "references": "none",
        "forceDataflowDynamicAnnotations": "true",
    }
