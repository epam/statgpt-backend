import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, BinaryIO

import aiofiles
import httpx
from aidial_client import AsyncDial, ResourceNotFoundError
from pydantic import SecretStr

_log = logging.getLogger(__name__)

_DIAL_REQUEST_TIMEOUT = 600


async def read_file_with_progress(file_path: str, chunk_size: int = 64 * 1024) -> BytesIO:
    """Read file asynchronously with progress logging."""
    file_size = os.path.getsize(file_path)
    uploaded = 0
    chunk_count = 0
    log_interval = 100
    buffer = BytesIO()

    _log.info(f"Starting upload of {file_path}")
    _log.info(f"Total file size: {file_size / (1024 * 1024):.2f} MB")

    async with aiofiles.open(file_path, 'rb') as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break

            buffer.write(chunk)
            uploaded += len(chunk)
            chunk_count += 1

            if chunk_count % log_interval == 0:
                percent = (uploaded / file_size) * 100
                _log.info(
                    f"Uploaded {uploaded / (1024 * 1024):.2f} MB / "
                    f"{file_size / (1024 * 1024):.2f} MB ({percent:.1f}%)"
                )

    _log.info(f"Upload completed: {uploaded / (1024 * 1024):.2f} MB total")
    buffer.seek(0)
    return buffer


class DialCore:
    """Unified DIAL API client backed by aidial-client's AsyncDial.

    Thin httpx wrappers remain only for operations not yet available in
    aidial-client (user-info endpoint, folder-metadata with pagination,
    and streaming file downloads).
    """

    def __init__(self, dial: AsyncDial, base_url: str, api_key: str) -> None:
        self._dial = dial
        self._base_url = base_url
        self._api_key = api_key

    async def get_user_info(self) -> dict[str, Any]:
        """Retrieve information about a user."""
        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers={'Api-Key': self._api_key},
            timeout=60,
        ) as client:
            response = await client.get('/v1/user/info')
            response.raise_for_status()
            return response.json()

    async def get_model_by(self, name: str) -> dict[str, Any]:
        model_info = await self._dial.model.get(name)
        return model_info.model_dump(by_alias=True, exclude_none=True)

    async def get_bucket(self, refresh: bool = False) -> str:
        """Return the cached bucket ID, optionally forcing a refresh."""
        if refresh:
            self._dial._my_bucket = None
        return await self._dial.my_bucket()

    async def get_file(self, url: str) -> bytes:
        """Download a file by its DIAL relative URL (e.g. ``files/{bucket}/path``)."""
        download = await self._dial.files.download(url)
        return await download.aget_content()

    async def get_file_with_type(self, url: str) -> tuple[bytes, str]:
        """Download a file and return ``(content, content_type)``."""
        download = await self._dial.files.download(url)
        content = await download.aget_content()
        content_type = download._response.headers.get("content-type", "application/octet-stream")
        return content, content_type

    async def get_file_by_path(self, path: str, *, bucket: str | None = None) -> tuple[bytes, str]:
        if not bucket:
            bucket = await self._dial.my_bucket()
        return await self.get_file_with_type(f"files/{bucket}/{path}")

    async def delete_file(self, url: str) -> None:
        await self._dial.files.delete(url)

    async def put_file(
        self,
        name: str,
        mime_type: str,
        content: BytesIO | bytes,
        *,
        bucket: str | None = None,
    ) -> dict[str, Any]:
        if not bucket:
            bucket = await self._dial.my_bucket()
        metadata = await self._dial.files.upload(
            f"files/{bucket}/{name}", file=(name, content, mime_type)
        )
        result = metadata.model_dump(by_alias=True)
        result['contentLength'] = metadata.content_length or 0
        result['contentType'] = metadata.content_type or ''
        return result

    async def put_local_file(
        self,
        name: str,
        path: str,
        *,
        bucket: str | None = None,
        show_progress: bool = False,
    ) -> dict[str, Any]:
        """Upload a local file to DIAL storage."""
        if not bucket:
            bucket = await self._dial.my_bucket()
        if show_progress:
            file_buffer = await read_file_with_progress(path)
            content: BytesIO | bytes = file_buffer
        else:
            async with aiofiles.open(path, 'rb') as f:
                content = BytesIO(await f.read())

        metadata = await self._dial.files.upload(
            f"files/{bucket}/{name}", file=(name, content, "application/octet-stream")
        )
        result = metadata.model_dump(by_alias=True)
        result['contentLength'] = metadata.content_length or 0
        result['contentType'] = metadata.content_type or ''
        return result

    async def get_file_metadata(
        self,
        path: str,
        *,
        token: str | None = None,
        limit: int = 100,
        bucket: str | None = None,
    ) -> dict[str, Any]:
        """Return raw metadata dict for a file or folder.

        Uses a thin httpx call so that the full server response (including
        ``updatedAt`` and any ``nextToken``) is preserved for callers that
        depend on those fields.
        """
        if not bucket:
            bucket = await self._dial.my_bucket()

        params: dict[str, Any] = {"limit": limit}
        if token:
            params["token"] = token

        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers={'Api-Key': self._api_key},
            timeout=60,
        ) as client:
            response = await client.get(f"/v1/metadata/files/{bucket}/{path}", params=params)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise ResourceNotFoundError(message=e.response.text) from e
                raise
            return response.json()

    async def write_file_to(self, url: str, sink: BinaryIO) -> None:
        """Stream-download a DIAL relative file URL and write chunks to *sink*.

        Uses a dedicated httpx streaming connection so large files are never
        fully buffered in memory.
        """
        total_size = 0
        downloaded = 0
        chunk_count = 0
        log_interval = 100

        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers={'Api-Key': self._api_key},
            timeout=None,
        ) as client:
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


@asynccontextmanager
async def dial_core_factory(base_url: str, api_key: str | SecretStr) -> AsyncIterator[DialCore]:
    if isinstance(api_key, SecretStr):
        api_key = api_key.get_secret_value()
    dial = AsyncDial(base_url=base_url, api_key=api_key, timeout=_DIAL_REQUEST_TIMEOUT)
    try:
        yield DialCore(dial, base_url=base_url, api_key=api_key)
    finally:
        await dial._http_client.internal_http_client.aclose()
