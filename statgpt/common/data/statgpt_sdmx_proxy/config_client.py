"""Client for the StatGPT SDMX proxy configuration server.

The config server (``epam/statgpt-sdmx-proxy``, module ``sdmx-proxy-config-server``) exposes the
proxy's registry/agency routing configuration on a single endpoint:

* ``GET`` returns the current configuration, ``404`` when nothing has been stored yet (cold start)
  and ``503`` when its storage backend is unavailable.
* ``POST`` replaces the configuration and echoes back what was stored, or returns ``400`` when its
  validator rejects the payload.

The payload is passed through verbatim as a JSON object: its schema is maintained in the proxy
repository, and the config server is the authority on what is valid.
"""

import logging
from typing import Any

import httpx

from statgpt.common.utils import ManagedHttpClient

_log = logging.getLogger(__name__)

_HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
proxy_config_http_client = ManagedHttpClient(_HTTP_TIMEOUT)

_ENV_PLACEHOLDER = "$env:{"

# Statuses that mean "the configuration you sent is not acceptable", as opposed to a transport or storage failure.
_REJECTED_STATUSES = frozenset(
    {httpx.codes.BAD_REQUEST, httpx.codes.UNPROCESSABLE_ENTITY}  # 400, 422
)


class ProxyConfigServerError(Exception):
    """Raised when the proxy config server cannot be reached or answers unexpectedly."""


class ProxyConfigValidationError(ProxyConfigServerError):
    """Raised when the proxy config server rejects the submitted configuration (HTTP 400)."""


def _validate_url(config_url: str) -> None:
    """Reject a URL that still contains an unresolved ``$env:{...}`` placeholder.

    ``replace_env`` leaves the placeholder verbatim when the environment variable is not set,
    so this is what an unconfigured deployment looks like.
    """
    if _ENV_PLACEHOLDER in config_url:
        raise ProxyConfigServerError(
            f"The proxy config server URL contains an unresolved environment variable "
            f"placeholder: {config_url!r}."
        )


def _error_detail(response: httpx.Response) -> str:
    """Extract the message from the config server's ``{"message": ..., "status": ...}`` body."""
    try:
        body = response.json()
        if isinstance(body, dict) and (message := body.get("message")):
            return str(message)
    except ValueError:
        pass
    return response.text.strip() or f"HTTP {response.status_code}"


async def fetch_proxy_config(config_url: str) -> dict[str, Any] | None:
    """Return the configuration stored by the config server, or None if it has none yet.

    Raises:
        ProxyConfigServerError: the server is unreachable or answered with an unexpected status.
    """
    _validate_url(config_url)

    _log.info("Fetching SDMX proxy configuration: GET %s", config_url)
    try:
        response = await proxy_config_http_client.client.get(config_url)
    except httpx.HTTPError as e:
        raise ProxyConfigServerError(f"Could not reach the proxy config server: {e}") from e

    if response.status_code == httpx.codes.NOT_FOUND:
        # Documented cold start: the server has no configuration stored yet.
        _log.info("The proxy config server has no configuration yet: GET %s -> 404", config_url)
        return None

    if response.is_error:
        raise ProxyConfigServerError(
            f"The proxy config server returned {response.status_code}: {_error_detail(response)}"
        )

    return response.json()


async def push_proxy_config(config_url: str, config: dict[str, Any]) -> dict[str, Any]:
    """Replace the configuration stored by the config server and return what it stored.

    Raises:
        ProxyConfigValidationError: the server rejected the configuration.
        ProxyConfigServerError: the server is unreachable or answered with an unexpected status.
    """
    _validate_url(config_url)

    _log.info("Updating SDMX proxy configuration: POST %s", config_url)
    try:
        response = await proxy_config_http_client.client.post(config_url, json=config)
    except httpx.HTTPError as e:
        raise ProxyConfigServerError(f"Could not reach the proxy config server: {e}") from e

    if response.status_code in _REJECTED_STATUSES:
        raise ProxyConfigValidationError(
            f"The proxy config server rejected the configuration: {_error_detail(response)}"
        )

    if response.is_error:
        raise ProxyConfigServerError(
            f"The proxy config server returned {response.status_code}: {_error_detail(response)}"
        )

    return response.json()
