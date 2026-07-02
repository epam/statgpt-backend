"""Process-wide shared ``httpx.AsyncClient`` connection pools.

Building a fresh ``httpx.AsyncClient`` per call site discards the connection pool, paying a
TCP+TLS handshake on every request (and leaking never-closed clients). This module owns
lazily-created shared clients instead:

- one client for all LLM/embeddings traffic;
- one client per SDMX data source configuration, keyed by the source id and its static
  headers, so a reconfigured source (e.g. a rotated API key) transparently gets a fresh
  client while in-flight requests keep using the old one.

``close_shared_http_clients`` must be awaited on application shutdown (both app lifespans).

For a component-owned shared client with its own lifecycle, use
``statgpt.common.utils.http_client.ManagedHttpClient`` instead; the pools here are
process-global and are all closed together via ``close_shared_http_clients``.
"""

import httpx

# Deliberately lower than the openai SDK's DEFAULT_CONNECTION_LIMITS (1000/100): one process
# talks to a handful of DIAL/Azure endpoints, not thousands of hosts.
_LLM_LIMITS = httpx.Limits(max_connections=200, max_keepalive_connections=50)
# Mirrors the timeout previously set per-construction in AsyncSdmxClient._create_httpx_client.
_SDMX_TIMEOUT = httpx.Timeout(90.0, connect=45.0)

_SdmxPoolKey = tuple[str, frozenset[tuple[str, str]]]

_llm_client: httpx.AsyncClient | None = None
_sdmx_clients: dict[_SdmxPoolKey, httpx.AsyncClient] = {}


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


def get_shared_sdmx_http_client(
    source_id: str, headers: dict[str, str] | None = None
) -> httpx.AsyncClient:
    """Return the shared client for an SDMX data source, creating it lazily.

    Keyed by ``(source_id, static headers)``: when the static headers change (the source was
    reconfigured), subsequent calls get a fresh client while in-flight requests keep using
    the old one, which stays pooled under its own key until ``close_shared_http_clients``
    (its idle keepalive sockets expire on their own). Pool growth is thus bounded by the
    number of distinct source configurations seen.
    """
    key: _SdmxPoolKey = (source_id, frozenset((headers or {}).items()))
    client = _sdmx_clients.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=_SDMX_TIMEOUT, headers=headers)
        _sdmx_clients[key] = client
    return client


async def close_shared_http_clients() -> None:
    """Close all shared clients. Safe to call multiple times."""
    global _llm_client
    clients: list[httpx.AsyncClient] = []
    if _llm_client is not None:
        clients.append(_llm_client)
        _llm_client = None
    clients.extend(_sdmx_clients.values())
    _sdmx_clients.clear()
    for client in clients:
        if not client.is_closed:
            await client.aclose()
