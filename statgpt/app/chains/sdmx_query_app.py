import logging
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import Field, field_validator, model_validator

from statgpt.app.chains.tools import ToolArgs, ToolUpstreamError
from statgpt.common.schemas.tool_details import SdmxQueryAppDetails
from statgpt.common.utils import ManagedHttpClient

_log = logging.getLogger(__name__)

_HTTP_TIMEOUT = httpx.Timeout(90.0, connect=45.0)

# Shared httpx client (lazy, closed on lifespan exit). Reused across calls so the connection
# pool (and TLS handshakes) is shared by this frequently-invoked passthrough instead of being
# rebuilt per request. Entered as an async context manager in the app lifespan.
sdmx_query_app_http_client = ManagedHttpClient(_HTTP_TIMEOUT)


class SdmxQueryAppArgs(ToolArgs):
    method: Literal["GET", "POST"] = Field(
        default="GET",
        description="HTTP method to use for the request. `GET` for structure/data, `POST` for availability.",
    )
    path: str = Field(
        description=(
            "Domain-less request path appended to the configured base URL, including any query"
            " string. Must start with a single '/', e.g."
            " '/structure/dataflow/IMF.RES/ED/1.0.0?details=full'."
        ),
    )
    body: dict[str, Any] | None = Field(
        default=None,
        description=(
            "JSON request body for `POST` requests (e.g. availability filters). Must be omitted"
            " for `GET` requests (supplying it with `GET` is rejected, not silently ignored)."
        ),
    )
    accept: str | None = Field(
        default=None,
        description="Value forwarded as the `Accept` header (e.g. an SDMX media type).",
    )

    @field_validator("path")
    @classmethod
    def _path_stays_on_configured_host(cls, path: str) -> str:
        if not path.startswith("/"):
            raise ValueError("`path` must start with '/'.")
        # Reject protocol-relative paths and absolute URLs to keep requests on the
        # configured host (the base URL is prepended verbatim, no urljoin).
        if path.startswith("//") or "://" in path:
            raise ValueError("`path` must be domain-less (no scheme or host).")
        return path

    @model_validator(mode="after")
    def _no_body_for_get(self) -> "SdmxQueryAppArgs":
        if self.method == "GET" and self.body is not None:
            raise ValueError("`body` is not supported for `GET` requests; use `method='POST'`.")
        return self


@dataclass(frozen=True)
class SdmxProxyResponse:
    """The upstream response, passed through as-is: the body regardless of status so the
    MCP-App component can render both payloads and upstream errors, plus the HTTP metadata the
    body alone can't reliably convey."""

    body: str
    status_code: int
    content_type: str | None


class SdmxQueryAppProxy:
    """Forwards a frontend-built SDMX request to the configured HTTP backend (e.g. an SDMX query
    application or proxy).

    The caller provides only a domain-less path; the trusted base URL prefix comes from the
    tool details, so the proxy can only reach the configured host (no SSRF surface). MCP-only:
    the request is built and invoked by the MCP-App component, never by the Supreme Agent.
    """

    def __init__(self, details: SdmxQueryAppDetails):
        self._details = details

    async def forward(
        self,
        path: str,
        method: Literal["GET", "POST"] = "GET",
        body: dict[str, Any] | None = None,
        accept: str | None = None,
    ) -> SdmxProxyResponse:
        url = f"{self._details.get_base_url()}{path}"
        headers = self._build_headers(method, accept)

        _log.info("SDMX query app passthrough: %s %s", method, url)
        try:
            response = await sdmx_query_app_http_client.client.request(
                method=method,
                url=url,
                headers=headers,
                json=body if method == "POST" else None,
            )
        except httpx.TimeoutException as e:
            raise ToolUpstreamError("The SDMX backend did not respond in time (timeout).") from e
        except httpx.HTTPError as e:
            # Connection errors, DNS failures, protocol errors, etc.
            raise ToolUpstreamError(f"Could not reach the SDMX backend: {e}") from e

        return SdmxProxyResponse(
            body=response.text,
            status_code=response.status_code,
            content_type=response.headers.get("content-type"),
        )

    @staticmethod
    def _build_headers(method: str, accept: str | None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if accept:
            headers["accept"] = accept
        if method == "POST":
            headers.setdefault("content-type", "application/json")
        return headers
