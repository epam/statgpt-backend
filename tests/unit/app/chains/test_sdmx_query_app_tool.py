import pytest

from statgpt.app.chains.sdmx_query_app_tool import SdmxQueryAppTool
from statgpt.app.chains.tools import ToolInputError
from statgpt.common.schemas.tool_details import SdmxQueryAppDetails

_BASE_URL = "https://sdmx.example.org/api"


class TestBuildUrl:
    """`_build_url` is the SSRF guard: it must keep every request on the trusted,
    pre-configured host. The base URL is prepended verbatim (no urljoin), so the
    only defense is rejecting any caller path that could escape the host."""

    def test_simple_path_is_appended_verbatim(self):
        url = SdmxQueryAppTool._build_url(_BASE_URL, "/structure/dataflow/IMF.RES/ED/1.0.0")
        assert url == f"{_BASE_URL}/structure/dataflow/IMF.RES/ED/1.0.0"

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
        with pytest.raises(ToolInputError, match="must start with '/'"):
            SdmxQueryAppTool._build_url(_BASE_URL, path)

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
        with pytest.raises(ToolInputError, match="domain-less"):
            SdmxQueryAppTool._build_url(_BASE_URL, path)

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
        url = SdmxQueryAppTool._build_url(_BASE_URL, path)
        assert url == f"{_BASE_URL}{path}"


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

    def test_no_double_slash_when_joined_with_path(self):
        details = SdmxQueryAppDetails(base_url_raw="https://sdmx.example.org/api/")
        url = SdmxQueryAppTool._build_url(details.get_base_url(), "/structure")
        assert url == "https://sdmx.example.org/api/structure"
