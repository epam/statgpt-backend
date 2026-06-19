from typing import Any, Literal

import httpx
from mcp.types import ToolAnnotations
from pydantic import Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import StatGptTool, ToolArgs, ToolInputError
from statgpt.app.schemas import ToolArtifact, ToolMessageState
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.schemas import SdmxQueryAppTool as SdmxQueryAppToolConfig
from statgpt.common.schemas import ToolTypes

_HTTP_TIMEOUT = httpx.Timeout(90.0, connect=45.0)


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
        description="JSON request body for `POST` requests (e.g. availability filters). Omit for `GET`.",
    )
    accept: str | None = Field(
        default=None,
        description="Value forwarded as the `Accept` header (e.g. an SDMX media type).",
    )


class SdmxQueryAppTool(StatGptTool[SdmxQueryAppToolConfig], tool_type=ToolTypes.SDMX_QUERY_APP):
    """MCP-only passthrough tool that forwards a frontend-built SDMX request to a configured
    HTTP backend (e.g. an SDMX query application or proxy) and returns the raw response for
    client-side rendering.

    The caller provides only a domain-less path; the trusted base URL prefix comes from the
    tool details, so the tool can only reach the configured host (no SSRF surface).
    """

    @classmethod
    def get_mcp_annotations(cls) -> ToolAnnotations:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @classmethod
    def get_args_schema(cls, tool_config: SdmxQueryAppToolConfig) -> type[SdmxQueryAppArgs]:
        return SdmxQueryAppArgs

    @staticmethod
    def _build_url(base_url: str, path: str) -> str:
        if not path.startswith("/"):
            raise ToolInputError("`path` must start with '/'.")
        # Reject protocol-relative paths and absolute URLs to keep requests on the
        # configured host (the base URL is prepended verbatim, no urljoin).
        if path.startswith("//") or "://" in path:
            raise ToolInputError("`path` must be domain-less (no scheme or host).")
        return f"{base_url}{path}"

    def _build_headers(self, method: str, accept: str | None, auth_context: AuthContext) -> dict:
        headers: dict[str, str] = {}
        if accept:
            headers["accept"] = accept
        if method == "POST":
            headers.setdefault("content-type", "application/json")
        return headers

    async def _arun(
        self,
        inputs: dict,
        path: str,
        method: Literal["GET", "POST"] = "GET",
        body: dict[str, Any] | None = None,
        accept: str | None = None,
        **kwargs,
    ) -> tuple[str, ToolArtifact]:
        auth_context = ChainParameters.get_auth_context(inputs)
        base_url = self._tool_config.details.get_base_url()
        url = self._build_url(base_url, path)

        headers = self._build_headers(method, accept, auth_context)

        logger.info(f"SDMX query app passthrough: {method} {url}")
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=body if method == "POST" else None,
            )

        # Passthrough: return the raw response body regardless of status so the MCP-App
        # component can render both successful payloads and upstream error responses.
        return response.text, ToolArtifact(state=ToolMessageState(type=self.tool_type))
