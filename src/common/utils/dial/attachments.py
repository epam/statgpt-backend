import base64
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, alias_generators

from common.config import DialConfig
from common.utils.media_types import MediaTypes

from .core import DialCore, dial_core_factory


class AttachmentResponse(BaseModel):
    name: str = Field(description="The name of the attachment")
    parent_path: str | None = Field(default=None, description="The parent path of the attachment")
    bucket: str = Field(description="The bucket of the attachment")
    url: str = Field(description="The URL of the attachment")
    node_type: str = Field(description="The node type of the attachment")
    resource_type: str = Field(description="The resource type of the attachment")
    updated_at: int | None = Field(
        default=None, description="The updated timestamp in milliseconds"
    )
    content_length: int = Field(description="The content length of the attachment")
    content_type: str = Field(description="The content type of the attachment")

    model_config = ConfigDict(alias_generator=alias_generators.to_camel)


class AttachmentsStorage:
    def __init__(self, dial_core: DialCore):
        self._dial_core = dial_core

    async def get_files_in_folder(
        self, folder: str, bucket: str | None = None
    ) -> list[AttachmentResponse]:
        """Return a list of files in the specified folder. If the folder does not exist, return an empty list."""

        try:
            response_json = await self._dial_core.get_file_metadata(folder, bucket=bucket)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return []
            raise
        files = [AttachmentResponse.model_validate(item) for item in response_json['items']]

        while token := response_json.get('nextToken'):
            response_json = await self._dial_core.get_file_metadata(
                folder, bucket=bucket, token=token, limit=100
            )
            files.extend(AttachmentResponse.model_validate(item) for item in response_json['items'])

        return files

    async def delete_file(self, url: str) -> None:
        """Delete the file at the specified URL.

        Args:
            url: The value of the `url` filed returned by the DIAL API. (AttachmentResponse.url)
        """
        await self._dial_core.delete_file(url)

    async def put_file(
        self, name: str, mime_type: str, content: BytesIO, bucket: str | None = None
    ) -> AttachmentResponse:
        response_json = await self._dial_core.put_file(name, mime_type, content, bucket=bucket)
        return AttachmentResponse.model_validate(response_json)

    async def put_local_file(
        self, name: str, path: str, *, bucket: str | None = None
    ) -> AttachmentResponse:
        response_json = await self._dial_core.put_local_file(name, path, bucket=bucket)
        return AttachmentResponse.model_validate(response_json)

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
        dataframe.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return await self.put_csv(name, csv_buffer)


@asynccontextmanager
async def attachments_storage_factory(
    api_key: str, base_url: str = DialConfig.get_url()
) -> AsyncIterator[AttachmentsStorage]:

    async with dial_core_factory(base_url=base_url, api_key=api_key) as dial_core:
        await dial_core.get_bucket()  # Load the bucket ID in the cache
        yield AttachmentsStorage(dial_core)


def b64_encode_image(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")
