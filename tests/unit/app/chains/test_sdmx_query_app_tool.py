from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from statgpt.app.chains import sdmx_query_app_tool as sdmx_module
from statgpt.app.chains.sdmx_query_app_tool import SdmxQueryAppArgs
from statgpt.app.chains.sdmx_query_app_tool import SdmxQueryAppTool as SdmxQueryAppChainTool
from statgpt.app.chains.tools import ToolUpstreamError
from statgpt.common.schemas.tool_details import SdmxQueryAppDetails
from statgpt.common.schemas.tools import SdmxQueryAppTool as SdmxQueryAppToolConfig


def _args(**kwargs) -> SdmxQueryAppArgs:
    return SdmxQueryAppArgs(inputs={}, **kwargs)


class TestPathValidation:
    """`SdmxQueryAppArgs.path` validation is the SSRF guard: the trusted base URL is
    prepended verbatim (no urljoin), so the only defense is rejecting any caller path
    that could escape the configured host."""

    def test_simple_path_is_accepted(self):
        args = _args(path="/structure/dataflow/IMF.RES/ED/1.0.0")
        assert args.path == "/structure/dataflow/IMF.RES/ED/1.0.0"

    @pytest.mark.parametrize(
        "path",
        [
            "foo",
            "structure/dataflow",
            "https://malicious.com",
            "http://malicious.com/path",
            "malicious.com/path",
            "?query=only",
            "",
        ],
    )
    def test_path_without_leading_slash_is_rejected(self, path: str):
        with pytest.raises(ValidationError, match="must start with '/'"):
            _args(path=path)

    @pytest.mark.parametrize(
        "path",
        [
            "//malicious.com",
            "//malicious.com/path",
            "///malicious.com",
            "/path/with/://embedded",
            "/redirect?next=https://malicious.com",
        ],
    )
    def test_protocol_relative_and_scheme_paths_are_rejected(self, path: str):
        with pytest.raises(ValidationError, match="domain-less"):
            _args(path=path)

    @pytest.mark.parametrize(
        "path",
        [
            "/@malicious.com",
            "/a/../b",
            "/structure/dataflow/IMF.RES/ED/1.0.0?details=full",
            "/data?startPeriod=2020&endPeriod=2024",
            "/path%2F..%2Fother",
            "/a?next=//malicious.com",  # '//' only forbidden as a path prefix, not in a query string
        ],
    )
    def test_tricky_but_on_host_paths_are_allowed(self, path: str):
        args = _args(path=path)
        assert args.path == path


class TestBodyValidation:
    """`body` is only meaningful for `POST`; supplying it with `GET` is rejected
    instead of being silently ignored."""

    def test_body_with_get_is_rejected(self):
        with pytest.raises(ValidationError, match="not supported for `GET`"):
            _args(path="/data", method="GET", body={"key": "value"})

    def test_body_with_post_is_accepted(self):
        args = _args(path="/availability", method="POST", body={"key": "value"})
        assert args.body == {"key": "value"}


class TestBaseUrlTrailingSlash:
    """`get_base_url` trims trailing slashes so concatenation with a leading-'/' path
    never yields a double slash."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("https://sdmx.example.org/api", "https://sdmx.example.org/api"),
            ("https://sdmx.example.org/api/", "https://sdmx.example.org/api"),
            ("https://sdmx.example.org/api///", "https://sdmx.example.org/api"),
        ],
    )
    def test_trailing_slashes_are_trimmed(self, raw: str, expected: str):
        details = SdmxQueryAppDetails(base_url_raw=raw)
        assert details.get_base_url() == expected


class TestUpstreamErrorScrubbing:
    """A backend connection failure is raised as an internals-free ``ToolUpstreamError``;
    the underlying error (which can name the internal endpoint) stays in the server log only."""

    def _tool(self) -> SdmxQueryAppChainTool:
        config = SdmxQueryAppToolConfig(
            name="sdmx_query_app",
            description="passthrough",
            details=SdmxQueryAppDetails(base_url_raw="https://sdmx-internal.svc/api"),
        )
        return SdmxQueryAppChainTool(
            tool_config=config,
            channel_config=SimpleNamespace(),  # type: ignore[arg-type]
            name="sdmx_query_app",
            description="passthrough",
        )

    async def test_connection_error_does_not_leak_endpoint(self, monkeypatch):
        class _FakeClient:
            async def request(self, *, method, url, **kwargs):
                raise httpx.ConnectError("connection failed", request=httpx.Request(method, url))

        monkeypatch.setattr(
            sdmx_module, "sdmx_query_app_http_client", SimpleNamespace(client=_FakeClient())
        )

        with pytest.raises(ToolUpstreamError) as exc_info:
            await self._tool()._arun(inputs={}, path="/data")

        message = str(exc_info.value)
        assert message == "The SDMX backend could not be reached."
        assert "sdmx-internal.svc" not in message
