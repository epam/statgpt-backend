"""Admin API client for CLI commands."""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
import requests
from pydantic import TypeAdapter

from statgpt.cli.settings import cli_settings
from statgpt.cli.shared.auth import get_cached_token
from statgpt.common.schemas import (
    Channel,
    ChannelDatasetBase,
    ChannelDatasetExpanded,
    ChannelIndexStatus,
    DataSet,
    DataSource,
    DataSourceType,
    GlossaryTerm,
    Job,
)

_log = logging.getLogger(__name__)

URL_PREFIX = "/admin/api/v1"
DEFAULT_TIMEOUT = 600  # 10 minutes
DEFAULT_LIMIT = 5000

# Type adapters for parsing lists
_channels_adapter = TypeAdapter(list[Channel])
_datasets_adapter = TypeAdapter(list[DataSet])
_channel_datasets_adapter = TypeAdapter(list[ChannelDatasetExpanded])
_data_sources_adapter = TypeAdapter(list[DataSource])
_data_source_types_adapter = TypeAdapter(list[DataSourceType])
_glossary_terms_adapter = TypeAdapter(list[GlossaryTerm])


class AdminAPIError(Exception):
    """Raised when Admin API request fails."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class AuthenticationRequired(Exception):
    """Raised when API returns 401 Unauthorized."""

    def __init__(self) -> None:
        super().__init__("Authentication required. Run 'auth login' first.")


class AdminClient:
    """Async HTTP client for StatGPT Admin API."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
    ):
        self._client = client
        self._base_url = base_url.rstrip("/")

    def _url(self, path: str) -> str:
        """Build full URL for API path."""
        return f"{self._base_url}{URL_PREFIX}{path}"

    async def health_check(self) -> bool:
        """Check if Admin API is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            _log.debug("Health check: %s", self._url("/health"))
            resp = await self._client.get(self._url("/health"))
            healthy = resp.json() == {"status": "ok"}
            _log.debug("Health check result: %s", healthy)
            return healthy
        except Exception as e:
            _log.debug("Health check failed: %s", e)
            return False

    async def get_channels(self, limit: int = DEFAULT_LIMIT) -> list[Channel]:
        """Fetch all channels."""
        resp = await self._client.get(self._url("/channels"), params={"limit": limit})
        resp.raise_for_status()
        return _channels_adapter.validate_python(resp.json()["data"])

    async def get_channel_by_deployment_id(self, deployment_id: str) -> Channel | None:
        """Find channel by deployment ID."""
        channels = await self.get_channels()
        for channel in channels:
            if channel.deployment_id == deployment_id:
                return channel
        return None

    async def get_datasets(
        self, channel_id: int | None = None, limit: int = DEFAULT_LIMIT
    ) -> list[DataSet]:
        """Fetch datasets, optionally filtered by channel."""
        params: dict[str, Any] = {"limit": limit}
        if channel_id is not None:
            params["channel_id"] = channel_id
        resp = await self._client.get(self._url("/datasets"), params=params)
        resp.raise_for_status()
        return _datasets_adapter.validate_python(resp.json()["data"])

    async def get_channel_datasets(
        self, channel_id: int, limit: int = DEFAULT_LIMIT
    ) -> list[ChannelDatasetExpanded]:
        """Fetch datasets for a specific channel."""
        resp = await self._client.get(
            self._url(f"/channels/{channel_id}/datasets"),
            params={"limit": limit},
        )
        resp.raise_for_status()
        return _channel_datasets_adapter.validate_python(resp.json()["data"])

    async def get_data_sources(self, limit: int = DEFAULT_LIMIT) -> list[DataSource]:
        """Fetch all data sources."""
        resp = await self._client.get(self._url("/data-sources"), params={"limit": limit})
        resp.raise_for_status()
        return _data_sources_adapter.validate_python(resp.json()["data"])

    async def get_data_source_types(self, limit: int = DEFAULT_LIMIT) -> list[DataSourceType]:
        """Fetch all data source types."""
        resp = await self._client.get(self._url("/data-sources/types"), params={"limit": limit})
        resp.raise_for_status()
        return _data_source_types_adapter.validate_python(resp.json()["data"])

    async def get_glossary_terms(
        self, channel_id: int, limit: int = DEFAULT_LIMIT
    ) -> list[GlossaryTerm]:
        """Fetch glossary terms for a channel."""
        resp = await self._client.get(
            self._url(f"/channels/{channel_id}/terms"),
            params={"limit": limit},
        )
        resp.raise_for_status()
        return _glossary_terms_adapter.validate_python(resp.json()["data"])

    async def get_channel_index_status(
        self, channel_id: int, scope: str = "latest_completed_versions"
    ) -> ChannelIndexStatus:
        """Fetch channel index status.

        Args:
            channel_id: Channel ID
            scope: Either "full" or "latest_completed_versions"

        Returns:
            ChannelIndexStatus object
        """
        resp = await self._client.get(
            self._url(f"/channels/{channel_id}/index-status"),
            params={"scope": scope},
        )
        resp.raise_for_status()
        return ChannelIndexStatus.model_validate(resp.json())

    async def import_channel(
        self,
        file_path: str,
        clean_up: bool = False,
        update_datasets: bool = False,
        update_data_sources: bool = False,
    ) -> Job:
        """Start channel import job.

        Returns:
            Job object
        """
        with open(file_path, "rb") as f:
            files = {"file": (file_path.split("/")[-1], f, "application/octet-stream")}
            params = {
                "clean_up": str(clean_up),
                "update_datasets": str(update_datasets),
                "update_data_sources": str(update_data_sources),
            }
            resp = await self._client.post(
                self._url("/channels/import"),
                params=params,
                files=files,
            )
        resp.raise_for_status()
        return Job.model_validate(resp.json())

    async def get_import_job_status(self, job_id: str) -> Job:
        """Get import job status."""
        resp = await self._client.get(self._url(f"/channels/jobs/{job_id}"))
        resp.raise_for_status()
        return Job.model_validate(resp.json())

    async def reload_channel_indicators(
        self,
        channel_id: int,
        dataset_ids: list[int] | None = None,
        max_n_embeddings: int | None = None,
    ) -> None:
        """Reload indicators for channel or specific datasets."""
        params: dict[str, Any] = {}
        if max_n_embeddings is not None:
            params["max_n_embeddings"] = max_n_embeddings

        if dataset_ids is None:
            # Reload all datasets in channel
            url = self._url(f"/channels/{channel_id}/datasets/reload-indicators")
            resp = await self._client.post(url, params=params)
            resp.raise_for_status()
        else:
            # Reload specific datasets
            for dataset_id in dataset_ids:
                url = self._url(f"/channels/{channel_id}/datasets/{dataset_id}/reload-indicators")
                resp = await self._client.post(url, params=params)
                resp.raise_for_status()

    async def deduplicate_channel(self, channel_id: int) -> None:
        """Deduplicate channel embeddings."""
        resp = await self._client.post(self._url(f"/channels/{channel_id}/datasets/deduplicate"))
        resp.raise_for_status()

    async def create_channel(self, channel_data: dict[str, Any]) -> Channel:
        """Create a new channel."""
        resp = await self._client.post(self._url("/channels"), json=channel_data)
        resp.raise_for_status()
        return Channel.model_validate(resp.json())

    async def update_channel(self, channel_id: int, channel_data: dict[str, Any]) -> Channel:
        """Update an existing channel."""
        resp = await self._client.post(self._url(f"/channels/{channel_id}"), json=channel_data)
        resp.raise_for_status()
        return Channel.model_validate(resp.json())

    async def create_data_source(self, data_source_data: dict[str, Any]) -> DataSource:
        """Create a new data source."""
        resp = await self._client.post(self._url("/data-sources"), json=data_source_data)
        resp.raise_for_status()
        return DataSource.model_validate(resp.json())

    async def update_data_source(
        self, data_source_id: int, data_source_data: dict[str, Any]
    ) -> DataSource:
        """Update an existing data source."""
        resp = await self._client.post(
            self._url(f"/data-sources/{data_source_id}"), json=data_source_data
        )
        resp.raise_for_status()
        return DataSource.model_validate(resp.json())

    async def create_dataset(self, dataset_data: dict[str, Any]) -> DataSet:
        """Create a new dataset."""
        resp = await self._client.post(self._url("/datasets"), json=dataset_data)
        resp.raise_for_status()
        return DataSet.model_validate(resp.json())

    async def update_dataset(self, dataset_id: int, dataset_data: dict[str, Any]) -> DataSet:
        """Update an existing dataset."""
        resp = await self._client.post(self._url(f"/datasets/{dataset_id}"), json=dataset_data)
        resp.raise_for_status()
        return DataSet.model_validate(resp.json())

    async def add_dataset_to_channel(self, channel_id: int, dataset_id: int) -> ChannelDatasetBase:
        """Add a dataset to a channel."""
        resp = await self._client.post(self._url(f"/channels/{channel_id}/datasets/{dataset_id}"))
        resp.raise_for_status()
        return ChannelDatasetBase.model_validate(resp.json())

    async def create_glossary_terms_bulk(
        self, channel_id: int, terms: list[dict[str, Any]]
    ) -> list[GlossaryTerm]:
        """Create multiple glossary terms."""
        resp = await self._client.post(
            self._url(f"/channels/{channel_id}/terms/bulk"),
            json=terms,
        )
        resp.raise_for_status()
        return _glossary_terms_adapter.validate_python(resp.json())

    async def update_glossary_terms_bulk(self, terms: list[dict[str, Any]]) -> list[GlossaryTerm]:
        """Update multiple glossary terms."""
        resp = await self._client.post(self._url("/terms/bulk"), json=terms)
        resp.raise_for_status()
        return _glossary_terms_adapter.validate_python(resp.json())

    async def delete_glossary_terms_bulk(self, term_ids: list[int]) -> None:
        """Delete multiple glossary terms.

        Note: Uses requests library as httpx doesn't support DELETE with body.
        """
        resp = requests.delete(
            f"{self._base_url}{URL_PREFIX}/terms/bulk",
            json=term_ids,
            headers=dict(self._client.headers),
            timeout=30,
        )
        resp.raise_for_status()


def _get_auth_headers() -> dict[str, str]:
    """Get auth headers using cached token if available."""
    cached_token = get_cached_token()
    if cached_token:
        return {"Authorization": f"Bearer {cached_token}"}
    return {}


async def _handle_response(response: httpx.Response) -> None:
    """Handle HTTP response, raising AuthenticationRequired on 401."""
    if response.status_code == 401:
        _log.warning("Received 401 Unauthorized from %s", response.url)
        raise AuthenticationRequired()


@asynccontextmanager
async def get_admin_client(
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncGenerator[AdminClient, None]:
    """Create an AdminClient context manager.

    Uses cached authentication token if available.
    Raises AuthenticationRequired on 401 responses.

    Args:
        base_url: Admin API base URL (defaults to settings)
        timeout: Request timeout in seconds

    Yields:
        AdminClient instance

    Raises:
        AuthenticationRequired: If API returns 401 Unauthorized
    """
    url = base_url or cli_settings.admin_url
    headers = _get_auth_headers()
    has_auth = "Authorization" in headers
    _log.debug("Creating AdminClient for %s (authenticated: %s)", url, has_auth)

    event_hooks = {"response": [_handle_response]}

    async with httpx.AsyncClient(
        headers=headers, timeout=timeout, event_hooks=event_hooks
    ) as client:
        yield AdminClient(client, url)
