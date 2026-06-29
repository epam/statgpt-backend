import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, BinaryIO

import httpx
from aidial_client import AsyncDial
from aidial_client._constants import DEFAULT_MAX_RETRIES
from aidial_client._http_client import AsyncHTTPClient
from pydantic import SecretStr

_log = logging.getLogger(__name__)

_DIAL_REQUEST_TIMEOUT = 600


@asynccontextmanager
async def dial_client_factory(base_url: str, api_key: str | SecretStr) -> AsyncIterator[AsyncDial]:
    """Yield an :class:`AsyncDial` whose underlying httpx client is owned here.

    The ``httpx.AsyncClient`` is created with ``async with`` and injected into
    the SDK via ``AsyncHTTPClient(internal_http_client=...)`` so the connection
    pool is closed deterministically by httpx's own context manager, without
    reaching into the SDK's private state after construction.
    """
    if isinstance(api_key, SecretStr):
        api_key = api_key.get_secret_value()

    async with httpx.AsyncClient(timeout=_DIAL_REQUEST_TIMEOUT) as http_client:
        yield AsyncDial(
            base_url=base_url,
            api_key=api_key,
            timeout=_DIAL_REQUEST_TIMEOUT,
            http_client=AsyncHTTPClient(
                base_url=base_url,
                api_key=api_key,
                bearer_token=None,
                max_retries=DEFAULT_MAX_RETRIES,
                timeout=_DIAL_REQUEST_TIMEOUT,
                internal_http_client=http_client,
            ),
        )


async def get_user_info(dial: AsyncDial) -> dict[str, Any]:
    """Retrieve information about the authenticated user.

    aidial-client (0.12.0) has no user-info resource, so a thin httpx call is
    made here, reusing the DIAL client's base URL and auth headers.
    """
    headers = await dial.auth_headers()
    async with httpx.AsyncClient(base_url=dial.base_url, headers=headers, timeout=60) as client:
        response = await client.get('/v1/user/info')
        response.raise_for_status()
        return response.json()


async def download_file_by_path(
    dial: AsyncDial, path: str, *, bucket: str | None = None
) -> tuple[bytes, str]:
    """Download a file stored under ``files/{bucket}/{path}`` and return
    ``(content, content_type)``. The body is buffered in memory; use
    :func:`open_file_stream` for large files."""
    if not bucket:
        bucket = await dial.my_bucket()
    url = f"files/{bucket}/{path}"
    download = await dial.files.download(url)
    content = await download.aget_content()
    metadata = await dial.files.get_metadata(url)
    return content, metadata.content_type or "application/octet-stream"


async def open_file_stream(
    dial: AsyncDial, url: str
) -> tuple[AsyncIterator[bytes], str, Callable[[], Awaitable[None]]]:
    """Open a streaming download for a DIAL relative file URL.

    Returns ``(chunks, content_type, aclose)``. The body is never buffered in
    memory, so this is safe for large files (e.g. exported channel archives).

    aidial-client (0.12.0) still buffers download bodies fully
    (``AsyncHTTPClient.request`` sends without ``stream=True``), so a dedicated
    httpx streaming connection is used here. The caller MUST invoke *aclose*
    once the stream is consumed.
    """
    headers = await dial.auth_headers()
    client = httpx.AsyncClient(base_url=dial.base_url, headers=headers, timeout=None)
    request = client.build_request('GET', f'/v1/{url}')
    response = await client.send(request, stream=True)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        await response.aclose()
        await client.aclose()
        raise

    content_type = response.headers.get('content-type', 'application/octet-stream')

    async def _aclose() -> None:
        await response.aclose()
        await client.aclose()

    return response.aiter_bytes(), content_type, _aclose


async def write_file_to(dial: AsyncDial, url: str, sink: BinaryIO) -> None:
    """Stream-download a DIAL relative file URL and write chunks to *sink*.

    Uses a dedicated httpx streaming connection so large files are never fully
    buffered in memory (see :func:`open_file_stream` for the rationale).
    """
    downloaded = 0
    chunk_count = 0
    log_interval = 100

    headers = await dial.auth_headers()
    async with httpx.AsyncClient(base_url=dial.base_url, headers=headers, timeout=None) as client:
        async with client.stream('GET', f'/v1/{url}') as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            _log.info(f"Starting download from {url}")
            if total_size > 0:
                _log.info(f"Total file size: {total_size / (1024 * 1024):.2f} MB")

            async for chunk in response.aiter_bytes(chunk_size=65536):
                sink.write(chunk)
                downloaded += len(chunk)
                chunk_count += 1

                if chunk_count % log_interval == 0:
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        _log.info(
                            f"Downloaded {downloaded / (1024 * 1024):.2f} MB / "
                            f"{total_size / (1024 * 1024):.2f} MB ({percent:.1f}%)"
                        )
                    else:
                        _log.info(f"Downloaded {downloaded / (1024 * 1024):.2f} MB")

            _log.info(f"Download completed: {downloaded / (1024 * 1024):.2f} MB total")
