import base64
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from io import BytesIO
from typing import Self

import aiofiles
import pandas as pd
from aidial_client import AsyncDial, DialException
from aidial_client.types.metadata import FileItem, FileMetadata
from pydantic import ConfigDict, alias_generators

from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils.media_types import MediaTypes

from .client import dial_client_factory, resolve_bucket

_log = logging.getLogger(__name__)


async def _read_file_with_progress(file_path: str, chunk_size: int = 64 * 1024) -> BytesIO:
    """Read a local file into memory, logging upload progress."""
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


class AttachmentResponse(FileItem):
    """A DIAL file item extended with the ``updatedAt`` timestamp DIAL returns
    for folder-listing entries (not a typed field on :class:`FileItem`)."""

    name: str = ""
    content_length: int = 0
    content_type: str = ""
    updated_at: int | None = None

    model_config = ConfigDict(
        alias_generator=alias_generators.to_camel, populate_by_name=True, extra="ignore"
    )

    @classmethod
    def from_file_metadata(cls, metadata: FileMetadata) -> Self:
        return cls.model_validate(metadata.model_dump(by_alias=True, exclude_none=True))

    @classmethod
    def from_metadata_item(cls, item: FileItem) -> Self:
        return cls.model_validate(item.model_dump(by_alias=True, exclude_none=True))


class AttachmentsStorage:
    """Higher-level helper for storing/retrieving attachments in DIAL storage.

    Wraps an :class:`AsyncDial` directly: uploads, deletes, and the paginated
    folder listing all go through the SDK (the listing uses the metadata
    resource's ``limit``/``token`` pagination).
    """

    def __init__(self, dial: AsyncDial):
        self._dial = dial

    async def get_files_in_folder(
        self, folder: str, bucket: str | None = None
    ) -> list[AttachmentResponse]:
        """Return a list of files in the specified folder. If the folder does not exist, return an empty list."""

        if not bucket:
            bucket = await resolve_bucket(self._dial)

        url = f"files/{bucket}/{folder}"
        files: list[AttachmentResponse] = []
        token: str | None = None

        while True:
            try:
                metadata = await self._dial.files.get_metadata(url, limit=100, token=token)
            except DialException as e:
                if e.status_code == HTTPStatus.NOT_FOUND:
                    return files
                raise

            files.extend(
                AttachmentResponse.from_metadata_item(item) for item in (metadata.items or [])
            )

            token = metadata.next_token
            if not token:
                break

        return files

    async def delete_file(self, url: str) -> None:
        """Delete the file at the specified URL.

        Args:
            url: The value of the `url` filed returned by the DIAL API. (AttachmentResponse.url)
        """
        await self._dial.files.delete(url)

    async def put_file(
        self, name: str, mime_type: str, content: BytesIO | bytes, bucket: str | None = None
    ) -> AttachmentResponse:
        if not bucket:
            bucket = await resolve_bucket(self._dial)
        metadata = await self._dial.files.upload(
            f"files/{bucket}/{name}", file=(name, content, mime_type)
        )
        return AttachmentResponse.from_file_metadata(metadata)

    async def put_local_file(
        self, name: str, path: str, *, bucket: str | None = None, show_progress: bool = False
    ) -> AttachmentResponse:
        if not bucket:
            bucket = await resolve_bucket(self._dial)
        if show_progress:
            content: BytesIO | bytes = await _read_file_with_progress(path)
        else:
            async with aiofiles.open(path, 'rb') as f:
                content = BytesIO(await f.read())
        metadata = await self._dial.files.upload(
            f"files/{bucket}/{name}", file=(name, content, "application/octet-stream")
        )
        return AttachmentResponse.from_file_metadata(metadata)

    async def put_png(self, name: str, content: BytesIO) -> AttachmentResponse:
        file_name = f"{name}-{uuid.uuid4()}.png"
        return await self.put_file(file_name, MediaTypes.PNG, content)

    async def put_png_bytes(self, name: str, content: bytes) -> AttachmentResponse:
        buffer = BytesIO(content)
        buffer.seek(0)
        return await self.put_png(name, buffer)

    async def put_json(self, name: str, content: str) -> AttachmentResponse:
        buffer = BytesIO()
        buffer.write(content.encode("utf-8"))
        buffer.seek(0)
        return await self.put_file(
            name=f"{name}-{uuid.uuid4()}.json",
            mime_type=MediaTypes.JSON,
            content=buffer,
        )

    async def put_pdb(self, name: str, content: BytesIO) -> AttachmentResponse:
        return await self.put_file(
            name=f"{name}-{uuid.uuid4()}.pdb",
            mime_type=MediaTypes.PDB,
            content=content,
        )

    async def put_pdb_bytes(self, name: str, content: bytes) -> AttachmentResponse:
        buffer = BytesIO(content)
        buffer.seek(0)
        return await self.put_pdb(name, buffer)

    async def put_xlsx(self, name: str, content: BytesIO) -> AttachmentResponse:
        return await self.put_file(
            name=f"{name}-{uuid.uuid4()}.xlsx",
            content=content,
            mime_type=MediaTypes.XLSX,
        )

    async def put_csv(self, name: str, content: BytesIO) -> AttachmentResponse:
        return await self.put_file(
            name=f"{name}-{uuid.uuid4()}.csv",
            content=content,
            mime_type=MediaTypes.CSV,
        )

    async def put_csv_from_dataframe(
        self, name: str, dataframe: pd.DataFrame
    ) -> AttachmentResponse:
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
