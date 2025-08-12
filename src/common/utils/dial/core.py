from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Literal

import httpx
from pydantic import SecretStr

HTTP_METHOD_TYPE = Literal['GET', 'POST', 'PUT', 'DELETE', 'PATCH']


def encode_url_characters(url: str) -> str:
    mapping = {
        '[': '%5B',
        ']': '%5D',
    }
    for char, replacement in mapping.items():
        url = url.replace(char, replacement)
    return url


class DialCore:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

        # Each instance of `DialCore` has its own api_key inside the client,
        # so we can cache the bucket
        self._bucket_id: str | None = None

    async def call_custom_endpoint(
        self, endpoint: str, method: HTTP_METHOD_TYPE = 'GET', **kwargs
    ) -> dict:
        response = await self._client.request(method, endpoint, **kwargs)
        response.raise_for_status()
        return response.json()

    async def get_models(self) -> dict[str, Any]:
        response = await self._client.get('/openai/models')
        response.raise_for_status()
        return response.json()

    async def get_model_by(self, name: str) -> dict[str, Any]:
        response = await self._client.get(f'/openai/models/{name}')
        response.raise_for_status()
        return response.json()

    async def get_bucket_json(self) -> dict[str, str]:
        response = await self._client.get('/v1/bucket')
        response.raise_for_status()
        return response.json()

    async def load_bucket(self) -> str:
        bucket_json = await self.get_bucket_json()
        if "appdata" in bucket_json:
            return bucket_json["appdata"]
        elif "bucket" in bucket_json:
            return bucket_json["bucket"]
        else:
            raise ValueError("No appdata or bucket found")

    async def get_bucket(self, refresh: bool = False) -> str:
        """Get the bucket ID from cache or load it from the API.
        Use `refresh=True` to force loading from the API.
        """

        if self._bucket_id is None or refresh:
            self._bucket_id = await self.load_bucket()

        return self._bucket_id

    async def get_file(self, url: str) -> bytes:
        """Get the file content from the specified URL.

        Args:
            url: The value of the `url` filed returned by the DIAL API.

        Returns:
            The file content as bytes.
        """

        response = await self._client.get(f"/v1/{url}")
        response.raise_for_status()
        return response.content

    async def get_file_by_path(self, path: str, *, bucket: str | None = None) -> tuple[bytes, str]:
        """
        Get the file content from the specified path.
        :param path: path to the file
        :param bucket: bucket to use
        :return: tuple of file content and content type
        """
        if not bucket:
            bucket = await self.get_bucket()

        response = await self._client.get(f"/v1/files/{bucket}/{path}")
        response.raise_for_status()
        return response.content, response.headers['Content-Type']

    async def delete_file(self, url: str) -> None:
        """Delete the file at the specified URL.

        Args:
            url: The value of the `url` filed returned by the DIAL API.
        """
        response = await self._client.delete(f"/v1/{url}")
        response.raise_for_status()

    async def put_file(
        self, name: str, mime_type: str, content: BytesIO, *, bucket: str | None = None
    ) -> dict[str, Any]:
        if not bucket:
            bucket = await self.get_bucket()

        response = await self._client.put(
            f"/v1/files/{bucket}/{name}",
            files={name: (name, content, mime_type)},
        )
        response.raise_for_status()
        return response.json()

    async def put_local_file(
        self, name: str, path: str, *, bucket: str | None = None
    ) -> dict[str, Any]:
        """Put file from local drive."""

        if not bucket:
            bucket = await self.get_bucket()

        response = await self._client.put(
            f"/v1/files/{bucket}/{name}",
            files={name: open(path, 'rb')},
        )
        response.raise_for_status()
        return response.json()

    async def get_file_metadata(
        self, path: str, *, token: str | None = None, limit: int = 100, bucket: str | None = None
    ) -> dict[str, Any]:
        """Call this endpoint to retrieve metadata for a file or folder at the specified path.
        If the path is a folder, it must end with a "/".

        If it is called for a folder, there can be optional `nextToken` field in the response to
        be used to request next items if present.

        Args:
            path: The path of the file or folder.
            token: The token from the previous request to request next items.
            limit: Limit on the number of items in the response.
            bucket: The bucket to use. If not provided, it will be fetched from the API.

        """

        if not bucket:
            bucket = await self.get_bucket()

        params: dict[str, Any] = {"limit": limit}
        if token:
            params["token"] = token

        url = f"/v1/metadata/files/{bucket}/{path}"
        response = await self._client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def put_conversation(
        self, path: str, json_data: dict, *, bucket: str | None = None
    ) -> dict[str, Any]:
        """Method to add a conversation to the specified bucket and path."""

        if not bucket:
            bucket = await self.get_bucket()

        path = encode_url_characters(path)
        response = await self._client.put(
            url=f"/v1/conversations/{bucket}/{path}",
            json=json_data,
        )
        response.raise_for_status()
        return response.json()

    async def create_publication_request(self, json_data: dict) -> dict[str, Any]:
        """Method to create a publish or unpublish request."""

        response = await self._client.post(
            url="/v1/ops/publication/create",
            json=json_data,
        )
        response.raise_for_status()
        return response.json()


@asynccontextmanager
async def dial_core_factory(base_url: str, api_key: str | SecretStr) -> AsyncIterator[DialCore]:
    if isinstance(api_key, SecretStr):
        api_key = api_key.get_secret_value()
    async with httpx.AsyncClient(
        base_url=base_url,
        headers={'Api-Key': api_key},
        timeout=600,
    ) as client:
        yield DialCore(client)
