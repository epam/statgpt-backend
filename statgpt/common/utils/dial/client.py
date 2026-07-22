import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import BinaryIO

from aidial_client import AsyncDial
from pydantic import SecretStr

_log = logging.getLogger(__name__)


@asynccontextmanager
async def dial_client_factory(base_url: str, api_key: str | SecretStr) -> AsyncIterator[AsyncDial]:
    """Yield an :class:`AsyncDial` and close its connection pool on exit.

    Relies on the SDK's own async lifecycle management: ``AsyncDial`` is used
    as an async context manager so its underlying httpx client is closed
    deterministically via ``aclose()`` when the block exits.
    """
    if isinstance(api_key, SecretStr):
        api_key = api_key.get_secret_value()

    async with AsyncDial(base_url=base_url, api_key=api_key) as dial:
        yield dial


async def resolve_bucket(dial: AsyncDial) -> str:
    """Return the storage prefix for the current key, preferring the
    per-application ``appdata`` path over the raw bucket id.

    When DIAL returns an ``appdata`` entry (i.e. the key belongs to a DIAL
    application), files must be namespaced under
    ``<user_bucket>/appdata/<app_name>`` rather than the raw bucket, otherwise
    they leak into the user's root bucket.
    """
    appdata_home = await dial.my_appdata_home()
    if appdata_home is not None:
        return str(appdata_home)
    return await dial.my_bucket()


async def download_file_by_path(
    dial: AsyncDial, path: str, *, bucket: str | None = None
) -> tuple[bytes, str]:
    """Download a file stored under ``files/{bucket}/{path}`` and return
    ``(content, content_type)``. The body is buffered in memory; use
    :func:`open_file_stream` for large files."""
    if not bucket:
        bucket = await resolve_bucket(dial)
    download = await dial.files.download(f"files/{bucket}/{path}")
    content = await download.aget_content()
    return content, download.content_type or "application/octet-stream"


async def open_file_stream(
    base_url: str, api_key: str | SecretStr, url: str
) -> tuple[AsyncIterator[bytes], str, Callable[[], Awaitable[None]]]:
    """Open a streaming download for a DIAL relative file URL.

    Returns ``(chunks, content_type, aclose)``. The body is streamed straight
    from DIAL (via the SDK's :meth:`AsyncFiles.stream_download`) and never fully
    buffered in memory, so this is safe for large files (e.g. exported channel
    archives).

    The DIAL client and the streaming response are kept open until the caller
    invokes *aclose*; *aclose* MUST be called once the stream is consumed.
    """
    stack = AsyncExitStack()
    try:
        dial = await stack.enter_async_context(dial_client_factory(base_url, api_key))
        download = await stack.enter_async_context(dial.files.stream_download(url))
    except BaseException:
        await stack.aclose()
        raise

    content_type = download.content_type or 'application/octet-stream'
    return aiter(download), content_type, stack.aclose


async def write_file_to(dial: AsyncDial, url: str, sink: BinaryIO) -> None:
    """Stream-download a DIAL relative file URL and write chunks to *sink*.

    Streams via the SDK so large files are never fully buffered in memory.
    """
    downloaded = 0
    chunk_count = 0
    log_interval = 100

    async with dial.files.stream_download(url) as download:
        total_size = int(download.headers.get('content-length', 0))

        _log.info(f"Starting download from {url}")
        if total_size > 0:
            _log.info(f"Total file size: {total_size / (1024 * 1024):.2f} MB")

        async for chunk in download:
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
