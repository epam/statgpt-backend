"""Process-wide shared ``httpx.AsyncClient`` connection pools.

Building a fresh ``httpx.AsyncClient`` per call site discards the connection pool, paying a
TCP+TLS handshake on every request (and leaking never-closed clients). This module owns
lazily-created shared clients instead:

- one client for all LLM/embeddings traffic;
- one client per SDMX data source, keyed by the source id. The clients carry no auth
  state: static headers (e.g. API keys) are applied per request by the SDMX clients, so
  a reconfigured source (e.g. a rotated API key) keeps using its pooled connections.

``close_shared_http_clients`` must be awaited on application shutdown (both app lifespans).

For a component-owned shared client with its own lifecycle, use
``statgpt.common.utils.http_client.ManagedHttpClient`` instead; the pools here are
process-global and are all closed together via ``close_shared_http_clients``.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

logger = logging.getLogger(__name__)

# Deliberately lower than the openai SDK's DEFAULT_CONNECTION_LIMITS (1000/100): one process
# talks to a handful of DIAL/Azure endpoints, not thousands of hosts.
_LLM_LIMITS = httpx.Limits(max_connections=200, max_keepalive_connections=50)
# Mirrors the timeout previously set per-construction in AsyncSdmxClient._create_httpx_client.
_SDMX_TIMEOUT = httpx.Timeout(90.0, connect=45.0)

_llm_client: httpx.AsyncClient | None = None
_sdmx_clients: dict[str, httpx.AsyncClient] = {}


def get_shared_llm_http_client() -> httpx.AsyncClient:
    """Return the shared client for LLM/embeddings traffic, creating it lazily.

    Mirrors the openai SDK's ``_DefaultAsyncHttpxClient`` defaults where they matter:
    ``follow_redirects=True`` (the SDK-built client follows 3xx transparently, e.g. ingress
    http->https redirects), while no client-level timeout is set — the SDK applies its
    per-request timeout on top of a provided ``http_client``, and the model factories pass
    explicit timeouts.
    """
    global _llm_client
    if _llm_client is None or _llm_client.is_closed:
        _llm_client = httpx.AsyncClient(limits=_LLM_LIMITS, follow_redirects=True)
    return _llm_client


def get_shared_sdmx_http_client(source_id: str) -> httpx.AsyncClient:
    """Return the shared client for an SDMX data source, creating it lazily.

    The client carries no auth state: static headers (e.g. an API key) are applied per
    request by the SDMX clients, so a reconfigured source keeps using the same pooled
    connections.
    """
    client = _sdmx_clients.get(source_id)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=_SDMX_TIMEOUT)
        _sdmx_clients[source_id] = client
    return client


async def close_shared_http_clients() -> None:
    """Close all shared clients; failures are logged, not raised. Safe to call multiple times."""
    global _llm_client
    clients: list[httpx.AsyncClient] = []
    if _llm_client is not None:
        clients.append(_llm_client)
        _llm_client = None
    clients.extend(_sdmx_clients.values())
    _sdmx_clients.clear()
    results = await asyncio.gather(
        *(client.aclose() for client in clients if not client.is_closed),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException):
            logger.error(f"Failed to close a shared HTTP client: {result!r}")


@asynccontextmanager
async def shared_http_clients_context() -> AsyncIterator[None]:
    """Close all shared clients on exit (for use in an app lifespan's ``async with`` stack)."""
    try:
        yield
    finally:
        await close_shared_http_clients()
