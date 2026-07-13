import httpx


class ManagedHttpClient:
    """Owns a lazily-created, shared ``httpx.AsyncClient`` with a fixed timeout, plus its
    lifecycle.

    The client is created on first use so it binds to the running event loop, reused across
    calls so the connection pool (and TLS handshakes) is shared, and closed when this object
    is used as an async context manager (e.g. in the app lifespan).

    No ``base_url`` is configured: callers pass absolute URLs. A single client already pools
    connections per origin, so one instance can serve multiple hosts; create separate
    instances only when the client *config* (e.g. timeout) must differ.

    For process-global pools shared across call sites (LLM/embeddings and SDMX traffic,
    closed together on shutdown), use ``statgpt.common.utils.http_pool`` instead.
    """

    def __init__(self, timeout: httpx.Timeout) -> None:
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the lazily-created client and reset state. No-op if never opened."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "ManagedHttpClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()
