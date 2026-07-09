import logging

import httpx
from fastmcp.apps import AppConfig, ResourceCSP, app_config_to_meta_dict
from fastmcp.resources import Resource
from pydantic import PrivateAttr

from statgpt.common.schemas import ProxiedResourceConfig
from statgpt.common.utils import AsyncLoadingCache, ManagedHttpClient

_log = logging.getLogger(__name__)

_HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
widget_http_client = ManagedHttpClient(_HTTP_TIMEOUT)

# One TTL cache per distinct cache_ttl_seconds; cache key is the resolved html_url.
_caches: dict[int, AsyncLoadingCache[str]] = {}


class WidgetResourceError(Exception):
    """Raised when the backend cannot fetch the widget HTML from the frontend endpoint."""


def _cache_for_ttl(ttl: int) -> AsyncLoadingCache[str]:
    cache = _caches.get(ttl)
    if cache is None:
        cache = AsyncLoadingCache(ttl=ttl)
        _caches[ttl] = cache
    return cache


async def _fetch_html(url: str) -> str:
    """GET the widget HTML from the internal frontend endpoint and return the body verbatim."""
    _log.info("Fetching MCP-App widget HTML: GET %s", url)
    try:
        response = await widget_http_client.client.get(url, follow_redirects=True)
        response.raise_for_status()
    except httpx.TimeoutException as e:
        raise WidgetResourceError("The widget endpoint did not respond in time (timeout).") from e
    except httpx.HTTPError as e:
        # Connection errors, DNS failures, non-2xx status, protocol errors, etc.
        raise WidgetResourceError(f"Could not fetch the widget HTML: {e}") from e
    return response.text


class WidgetResource(Resource):
    """A FastMCP resource that serves widget HTML proxied verbatim from an external endpoint.

    The backend stores no HTML: ``read()`` fetches it from the configured internal endpoint and
    caches it for ``cache_ttl_seconds``. The CSP (``resourceDomains``) and MIME type come from the
    resource ``meta`` / ``mime_type`` and are propagated to the host by FastMCP.
    """

    _html_url: str = PrivateAttr()
    _cache_ttl_seconds: int = PrivateAttr()

    def __init__(self, *, html_url: str, cache_ttl_seconds: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._html_url = html_url
        self._cache_ttl_seconds = cache_ttl_seconds

    @classmethod
    def from_config(cls, config: ProxiedResourceConfig) -> "WidgetResource":
        return cls(
            uri=config.uri,  # type: ignore[arg-type]  # str is coerced to AnyUrl
            mime_type=config.mime_type,
            meta={
                "ui": app_config_to_meta_dict(
                    AppConfig(csp=ResourceCSP(resource_domains=[config.get_origin()]))
                )
            },
            html_url=config.get_html_url(),
            cache_ttl_seconds=config.cache_ttl_seconds,
        )

    async def read(self) -> str:
        return await _cache_for_ttl(self._cache_ttl_seconds).get(
            self._html_url, loader=lambda: _fetch_html(self._html_url)
        )
