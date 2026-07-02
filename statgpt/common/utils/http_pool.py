"""Process-wide shared ``httpx.AsyncClient`` connection pools.

Building a fresh ``httpx.AsyncClient`` per call site discards the connection pool, paying a
TCP+TLS handshake on every request (and leaking never-closed clients). This module owns
lazily-created shared clients instead:

- one client for all LLM/embeddings traffic;
- one client per SDMX data source, keyed by the source id and its static headers, so a
  reconfigured source (e.g. a rotated API key) transparently gets a fresh client.

``close_shared_http_clients`` must be awaited on application shutdown (both app lifespans).
"""

import httpx

_LLM_LIMITS = httpx.Limits(max_connections=200, max_keepalive_connections=50)
# Mirrors the timeout previously set per-construction in AsyncSdmxClient._create_httpx_client.
_SDMX_TIMEOUT = httpx.Timeout(90.0, connect=45.0)

_llm_client: httpx.AsyncClient | None = None
_sdmx_clients: dict[str, tuple[frozenset[tuple[str, str]], httpx.AsyncClient]] = {}
_retired_sdmx_clients: list[httpx.AsyncClient] = []


def get_shared_llm_http_client() -> httpx.AsyncClient:
    """Return the shared client for LLM/embeddings traffic, creating it lazily.

    No client-level timeout: the openai SDK applies its per-request timeout on top of a
    provided ``http_client``, and the model factories pass explicit timeouts.
    """
    global _llm_client
    if _llm_client is None or _llm_client.is_closed:
        _llm_client = httpx.AsyncClient(limits=_LLM_LIMITS)
    return _llm_client


def get_shared_sdmx_http_client(
    source_id: str, headers: dict[str, str] | None = None
) -> httpx.AsyncClient:
    """Return the shared client for an SDMX data source, creating it lazily.

    Keyed by ``source_id``; when the static headers change (the source was reconfigured),
    a new client is created and the old one is retired rather than closed — in-flight
    requests may still use it, so it is only closed by ``close_shared_http_clients``.
    """
    headers_key = frozenset((headers or {}).items())
    cached = _sdmx_clients.get(source_id)
    if cached is not None:
        cached_headers_key, client = cached
        if cached_headers_key == headers_key and not client.is_closed:
            return client
        if not client.is_closed:
            _retired_sdmx_clients.append(client)
    client = httpx.AsyncClient(timeout=_SDMX_TIMEOUT, headers=headers)
    _sdmx_clients[source_id] = (headers_key, client)
    return client


async def close_shared_http_clients() -> None:
    """Close all shared clients (including retired ones). Safe to call multiple times."""
    global _llm_client
    clients: list[httpx.AsyncClient] = []
    if _llm_client is not None:
        clients.append(_llm_client)
        _llm_client = None
    clients.extend(client for _, client in _sdmx_clients.values())
    _sdmx_clients.clear()
    clients.extend(_retired_sdmx_clients)
    _retired_sdmx_clients.clear()
    for client in clients:
        if not client.is_closed:
            await client.aclose()
