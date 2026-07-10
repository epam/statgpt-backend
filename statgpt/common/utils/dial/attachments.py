import base64
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from io import BufferedReader, BytesIO, FileIO

import pandas as pd
from aidial_client import AsyncDial, DialException
from aidial_client.types.metadata import FileItem

from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils.media_types import MediaTypes

from .client import dial_client_factory, resolve_bucket

_log = logging.getLogger(__name__)


class _ProgressBufferedReader(BufferedReader):
    """A ``BufferedReader`` that logs upload progress as it is read.

    Subclasses ``BufferedReader`` because the DIAL SDK validates upload content
    against ``bytes | str | BufferedReader | IO[bytes]`` using pydantic v1, and
    ``isinstance(obj, typing.IO)`` is ``False`` for arbitrary file-like objects
    (e.g. ``BytesIO``). A real ``BufferedReader`` subclass is the only file-like
    type that both passes validation and lets httpx stream from disk.
    """

    def __init__(self, path: str, log_interval: int = 100) -> None:
        super().__init__(FileIO(path, "rb"))
        self._total = os.path.getsize(path)
        self._uploaded = 0
        self._chunk_count = 0
        self._log_interval = log_interval
        _log.info(f"Starting upload of {path}")
        _log.info(f"Total file size: {self._total / (1024 * 1024):.2f} MB")

    def read(self, size: int | None = -1) -> bytes:
        chunk = super().read(size)
        if chunk:
            self._uploaded += len(chunk)
            self._chunk_count += 1
            if self._chunk_count % self._log_interval == 0:
                percent = (self._uploaded / self._total * 100) if self._total else 0.0
                _log.info(
                    f"Uploaded {self._uploaded / (1024 * 1024):.2f} MB / "
                    f"{self._total / (1024 * 1024):.2f} MB ({percent:.1f}%)"
                )
        return chunk


class AttachmentsStorage:
    """Higher-level helper for storing/retrieving attachments in DIAL storage.

    Wraps an :class:`AsyncDial` directly: uploads, deletes, and the paginated
    folder listing all go through the SDK (the listing uses the metadata
    resource's ``limit``/``token`` pagination).
    """

    def __init__(self, dial: AsyncDial):
        self._dial = dial

    async def get_files_in_folder(self, folder: str, bucket: str | None = None) -> list[FileItem]:
        """Return a list of files in the specified folder. If the folder does not exist, return an empty list."""

        if not bucket:
            bucket = await resolve_bucket(self._dial)

        folder = folder.strip('/')
        # The trailing "/" is required: DIAL treats "example/" as a folder and "example" as a file.
        url = f"files/{bucket}/{folder}/"
        files: list[FileItem] = []
        token: str | None = None

        while True:
            try:
                metadata = await self._dial.files.metadata.get(
                    resource="files", relative_url=url, limit=100, token=token
                )
            except DialException as e:
                if e.status_code == HTTPStatus.NOT_FOUND:
                    return files
                raise

            files.extend(metadata.items or [])

            token = metadata.next_token
            if not token:
                break

        return files

    async def delete_file(self, url: str) -> None:
        """Delete the file at the specified URL.

        Args:
            url: The value of the `url` field returned by the DIAL API. (FileItem.url)
        """
        await self._dial.files.delete(url)

    async def put_file(
        self, name: str, mime_type: str, content: BytesIO | bytes, bucket: str | None = None
    ) -> FileItem:
        if not bucket:
            bucket = await resolve_bucket(self._dial)
        # The SDK's pydantic-v1 validation does not accept BytesIO as IO[bytes],
        # so materialize it to bytes here.
        data = content.getvalue() if isinstance(content, BytesIO) else content
        return await self._dial.files.upload(f"files/{bucket}/{name}", file=(name, data, mime_type))

    async def put_local_file(
        self,
        name: str,
        path: str,
        *,
        mime_type: str = "application/octet-stream",
        bucket: str | None = None,
        show_progress: bool = False,
    ) -> FileItem:
        if not bucket:
            bucket = await resolve_bucket(self._dial)
        reader = (
            _ProgressBufferedReader(path) if show_progress else BufferedReader(FileIO(path, "rb"))
        )
        try:
            return await self._dial.files.upload(
                f"files/{bucket}/{name}", file=(name, reader, mime_type)
            )
        finally:
            reader.close()

    async def put_png(self, name: str, content: BytesIO) -> FileItem:
        file_name = f"{name}-{uuid.uuid4()}.png"
        return await self.put_file(file_name, MediaTypes.PNG, content)

    async def put_png_bytes(self, name: str, content: bytes) -> FileItem:
        buffer = BytesIO(content)
        buffer.seek(0)
        return await self.put_png(name, buffer)

    async def put_json(self, name: str, content: str) -> FileItem:
        buffer = BytesIO()
        buffer.write(content.encode("utf-8"))
        buffer.seek(0)
        return await self.put_file(
            name=f"{name}-{uuid.uuid4()}.json",
            mime_type=MediaTypes.JSON,
            content=buffer,
        )

    async def put_pdb(self, name: str, content: BytesIO) -> FileItem:
        return await self.put_file(
            name=f"{name}-{uuid.uuid4()}.pdb",
            mime_type=MediaTypes.PDB,
            content=content,
        )

    async def put_pdb_bytes(self, name: str, content: bytes) -> FileItem:
        buffer = BytesIO(content)
        buffer.seek(0)
        return await self.put_pdb(name, buffer)

    async def put_xlsx(self, name: str, content: BytesIO) -> FileItem:
        return await self.put_file(
            name=f"{name}-{uuid.uuid4()}.xlsx",
            content=content,
            mime_type=MediaTypes.XLSX,
        )

    async def put_csv(self, name: str, content: BytesIO) -> FileItem:
        return await self.put_file(
            name=f"{name}-{uuid.uuid4()}.csv",
            content=content,
            mime_type=MediaTypes.CSV,
        )

    async def put_csv_from_dataframe(self, name: str, dataframe: pd.DataFrame) -> FileItem:
        """Put a CSV file from a pandas DataFrame."""
        csv_buffer = BytesIO()
        dataframe.to_csv(csv_buffer, index=False, date_format="%Y-%m-%d", lineterminator="\n")
        csv_buffer.seek(0)
        return await self.put_csv(name, csv_buffer)


@asynccontextmanager
async def attachments_storage_factory(
    api_key: str, base_url: str = dial_settings.url
) -> AsyncIterator[AttachmentsStorage]:

    async with dial_client_factory(base_url=base_url, api_key=api_key) as dial:
        await resolve_bucket(dial)  # Warm the bucket/appdata cache
        yield AttachmentsStorage(dial)


def b64_encode_image(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")
