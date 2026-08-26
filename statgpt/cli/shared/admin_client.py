"""Admin API client for CLI commands."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
from pydantic import TypeAdapter, ValidationError

from statgpt.cli.settings import cli_settings
from statgpt.cli.shared.auth import get_cached_token
from statgpt.cli.shared.batch_report import BatchReport
from statgpt.common.schemas import (
    Channel,
    ChannelDatasetBase,
    ChannelDatasetExpanded,
    ChannelIndexStatus,
    DataSet,
    DataSetUpdateResponse,
    DataSource,
    DataSourceType,
    DeduplicationJob,
    DiscoveryDataset,
    DiscoveryDatasetStats,
    DiscoveryIndexingJob,
    DiscoveryPayloadErrorDetail,
    DiscoveryUploadMode,
    DiscoveryUploadSummary,
    GlossaryTerm,
    GlossaryTermBase,
    GlossaryTermUpdateBulk,
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
_discovery_datasets_adapter = TypeAdapter(list[DiscoveryDataset])


class AdminAPIError(Exception):
    """Raised when Admin API request fails."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class AuthenticationRequired(Exception):
    """Raised when API returns 401 Unauthorized."""

    def __init__(self) -> None:
        super().__init__("Authentication required. Run 'auth login' first.")


class DiscoveryPayloadError(AdminAPIError):
    """Raised when a discovery write is refused as structurally unusable.

    Carries the per-cell report the API answers with, so a command can show which rows need
    fixing instead of printing a JSON body.
    """

    def __init__(self, detail: DiscoveryPayloadErrorDetail):
        self.detail = detail
        super().__init__(detail.message, status_code=400)


def _detail_text(detail: Any) -> str | None:
    """Human-readable text from a FastAPI ``detail`` payload, whatever shape it has.

    The reindex endpoints answer with ``InvalidConfigurationError.to_dict()`` - a dict
    carrying ``error`` - either on its own or inside a list, while validation errors use
    ``msg`` and plain aborts use a string.
    """
    if isinstance(detail, str):
        return detail or None
    if isinstance(detail, dict):
        for key in ("error", "msg", "message"):
            value = detail.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    if isinstance(detail, list):
        parts = [text for text in (_detail_text(item) for item in detail) if text]
        return "; ".join(parts) or None
    return None


def _describe_error(resp: httpx.Response) -> str:
    """One-line reason for a failed response, preferring the API's ``detail`` field."""
    prefix = f"HTTP {resp.status_code}"
    try:
        payload = resp.json()
    except ValueError:
        payload = None

    text = _detail_text(payload.get("detail")) if isinstance(payload, dict) else None
    if not text:
        text = resp.text.strip()
    return f"{prefix}: {text}" if text else prefix


def _payload_error(resp: httpx.Response) -> AdminAPIError:
    """Turn a refused discovery write into a `DiscoveryPayloadError`, when it is one.

    A 400 from these endpoints carries the per-cell report, but not every 400 has to: an error
    raised before the handler runs keeps FastAPI's own shape, so an unparseable body falls back
    to the generic one-line error.
    """
    try:
        payload = resp.json()
        detail = DiscoveryPayloadErrorDetail.model_validate(payload["detail"])
    except (ValueError, KeyError, TypeError, ValidationError):
        return AdminAPIError(_describe_error(resp), status_code=resp.status_code)
    return DiscoveryPayloadError(detail)


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

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        """Raise AdminAPIError on HTTP errors, including the response body."""
        if resp.is_success:
            return
        method = resp.request.method
        url = resp.request.url
        message = f"HTTP {resp.status_code} for {method} {url}"
        body = resp.text
        if body:
            message += f"\nResponse body: {body}"
        raise AdminAPIError(message, status_code=resp.status_code)

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
        self._raise_for_status(resp)
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
        self._raise_for_status(resp)
        return _datasets_adapter.validate_python(resp.json()["data"])

    async def get_channel_datasets(
        self, channel_id: int, limit: int = DEFAULT_LIMIT
    ) -> list[ChannelDatasetExpanded]:
        """Fetch datasets for a specific channel."""
        resp = await self._client.get(
            self._url(f"/channels/{channel_id}/datasets"),
            params={"limit": limit},
        )
        self._raise_for_status(resp)
        return _channel_datasets_adapter.validate_python(resp.json()["data"])

    async def get_data_sources(self, limit: int = DEFAULT_LIMIT) -> list[DataSource]:
        """Fetch all data sources."""
        resp = await self._client.get(self._url("/data-sources"), params={"limit": limit})
        self._raise_for_status(resp)
        return _data_sources_adapter.validate_python(resp.json()["data"])

    async def get_data_source_types(self, limit: int = DEFAULT_LIMIT) -> list[DataSourceType]:
        """Fetch all data source types."""
        resp = await self._client.get(self._url("/data-sources/types"), params={"limit": limit})
        self._raise_for_status(resp)
        return _data_source_types_adapter.validate_python(resp.json()["data"])

    async def get_glossary_terms(
        self, channel_id: int, limit: int = DEFAULT_LIMIT
    ) -> list[GlossaryTerm]:
        """Fetch glossary terms for a channel."""
        resp = await self._client.get(
            self._url(f"/channels/{channel_id}/terms"),
            params={"limit": limit},
        )
        self._raise_for_status(resp)
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
        self._raise_for_status(resp)
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
        self._raise_for_status(resp)
        return Job.model_validate(resp.json())

    async def get_import_job_status(self, job_id: str) -> Job:
        """Get import job status."""
        resp = await self._client.get(self._url(f"/channels/jobs/{job_id}"))
        self._raise_for_status(resp)
        return Job.model_validate(resp.json())

    async def reload_channel_indicators(
        self,
        channel_id: int,
        dataset_ids: list[int] | None = None,
        max_n_embeddings: int | None = None,
        labels: dict[int, str] | None = None,
    ) -> BatchReport:
        """Reload indicators for a channel or for specific datasets.

        Per-dataset rejections are recorded and do not stop the datasets after them: the
        endpoint answers 400 for a dataset whose configuration cannot be loaded, and one
        such dataset must not block a bulk reindex.

        Authentication failures still raise, since they are not a per-dataset problem and
        would otherwise be reported once per dataset.

        Args:
            channel_id: Channel to reindex.
            dataset_ids: Datasets to reindex, or None to reindex the whole channel.
            max_n_embeddings: Debugging cap on the number of documents embedded.
            labels: Dataset id -> display name (a URN), so the summary is readable.

        Returns:
            One result per submitted dataset.
        """
        params: dict[str, Any] = {}
        if max_n_embeddings is not None:
            params["max_n_embeddings"] = max_n_embeddings

        report = BatchReport(title="Reindex summary")

        if dataset_ids is None:
            # Reload all datasets in channel
            url = self._url(f"/channels/{channel_id}/datasets/reload-indicators")
            resp = await self._client.post(url, params=params)
            if resp.is_success:
                report.record_ok("channel", str(channel_id))
            else:
                report.record_failed("channel", str(channel_id), _describe_error(resp))
            return report

        # Reload specific datasets
        for dataset_id in dataset_ids:
            label = (labels or {}).get(dataset_id) or str(dataset_id)
            url = self._url(f"/channels/{channel_id}/datasets/{dataset_id}/reload-indicators")
            resp = await self._client.post(url, params=params)

            if resp.is_success:
                report.record_ok("dataset", label)
                continue
            if resp.status_code in (401, 403):
                self._raise_for_status(resp)

            detail = _describe_error(resp)
            _log.warning("Reindex rejected for dataset %s: %s", label, detail)
            report.record_failed("dataset", label, detail)

        return report

    async def deduplicate_channel(self, channel_id: int) -> DeduplicationJob:
        """Trigger channel deduplication; returns the created job for polling."""
        resp = await self._client.post(self._url(f"/channels/{channel_id}/datasets/deduplicate"))
        self._raise_for_status(resp)
        return DeduplicationJob.model_validate(resp.json())

    async def get_deduplication_job(self, job_id: int) -> DeduplicationJob:
        """Get a deduplication job by ID for polling status."""
        resp = await self._client.get(self._url(f"/channels/deduplication-jobs/{job_id}"))
        self._raise_for_status(resp)
        return DeduplicationJob.model_validate(resp.json())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ discovery datasets ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def get_discovery_stats(self, channel_id: int) -> DiscoveryDatasetStats:
        """Fetch the channel's discovery record counts per validation and indexing status."""
        resp = await self._client.get(self._url(f"/channels/{channel_id}/discovery-datasets/stats"))
        self._raise_for_status(resp)
        return DiscoveryDatasetStats.model_validate(resp.json())

    async def upload_discovery_datasets(
        self,
        channel_id: int,
        file_path: str,
        mode: DiscoveryUploadMode = DiscoveryUploadMode.UPSERT,
    ) -> DiscoveryUploadSummary:
        """Load a discovery workbook (.xlsx) or CSV into a channel.

        The API classifies the file by its signature bytes and, failing that, its name, so the
        real basename travels with the upload and no format detection happens here.

        A refused write comes back as `DiscoveryPayloadError` carrying one entry per offending
        record; every other failure keeps the generic error.
        """
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
            resp = await self._client.post(
                self._url(f"/channels/{channel_id}/discovery-datasets/upload"),
                params={"mode": mode.value},
                files=files,
            )

        if resp.status_code == 400:
            raise _payload_error(resp)
        self._raise_for_status(resp)
        return DiscoveryUploadSummary.model_validate(resp.json())

    async def clear_discovery_datasets(self, channel_id: int) -> list[DiscoveryDataset]:
        """Delete every discovery dataset of a channel, and its published document.

        The API answers with the deleted records themselves as a bare list, not the
        `{"data": [...]}` envelope the paginated `get_*` endpoints use.
        """
        resp = await self._client.delete(
            self._url(f"/channels/{channel_id}/discovery-datasets/bulk")
        )
        self._raise_for_status(resp)
        return _discovery_datasets_adapter.validate_python(resp.json())

    async def trigger_discovery_indexing(
        self, channel_id: int, force: bool = False
    ) -> DiscoveryIndexingJob:
        """Trigger a discovery indexing job; returns the created job for polling.

        A 409 - a job is already running, or the channel has no discovery RAG application -
        is answered with the API's own one-line reason rather than a dump of the response.
        """
        resp = await self._client.post(
            self._url(f"/channels/{channel_id}/discovery-datasets/indexing-jobs"),
            params={"force": str(force).lower()},
        )
        if resp.status_code == 409:
            raise AdminAPIError(_describe_error(resp), status_code=resp.status_code)
        self._raise_for_status(resp)
        return DiscoveryIndexingJob.model_validate(resp.json())

    async def get_discovery_indexing_job(self, job_id: int) -> DiscoveryIndexingJob:
        """Get a discovery indexing job by ID for polling status."""
        resp = await self._client.get(self._url(f"/discovery-datasets/indexing-jobs/{job_id}"))
        self._raise_for_status(resp)
        return DiscoveryIndexingJob.model_validate(resp.json())

    async def create_channel(self, channel_data: dict[str, Any]) -> Channel:
        """Create a new channel."""
        resp = await self._client.post(self._url("/channels"), json=channel_data)
        self._raise_for_status(resp)
        return Channel.model_validate(resp.json())

    async def update_channel(self, channel_id: int, channel_data: dict[str, Any]) -> Channel:
        """Update an existing channel."""
        resp = await self._client.post(self._url(f"/channels/{channel_id}"), json=channel_data)
        self._raise_for_status(resp)
        return Channel.model_validate(resp.json())

    async def create_data_source(self, data_source_data: dict[str, Any]) -> DataSource:
        """Create a new data source."""
        resp = await self._client.post(self._url("/data-sources"), json=data_source_data)
        self._raise_for_status(resp)
        return DataSource.model_validate(resp.json())

    async def update_data_source(
        self, data_source_id: int, data_source_data: dict[str, Any]
    ) -> DataSource:
        """Update an existing data source."""
        resp = await self._client.post(
            self._url(f"/data-sources/{data_source_id}"), json=data_source_data
        )
        self._raise_for_status(resp)
        return DataSource.model_validate(resp.json())

    async def create_dataset(self, dataset_data: dict[str, Any]) -> DataSet:
        """Create a new dataset."""
        resp = await self._client.post(self._url("/datasets"), json=dataset_data)
        self._raise_for_status(resp)
        return DataSet.model_validate(resp.json())

    async def update_dataset(
        self, dataset_id: int, dataset_data: dict[str, Any]
    ) -> DataSetUpdateResponse:
        """Update an existing dataset.

        Returns the updated dataset along with the results of propagating config
        changes to all channel datasets that reference this dataset.
        """
        resp = await self._client.post(self._url(f"/datasets/{dataset_id}"), json=dataset_data)
        self._raise_for_status(resp)
        return DataSetUpdateResponse.model_validate(resp.json())

    async def add_dataset_to_channel(self, channel_id: int, dataset_id: int) -> ChannelDatasetBase:
        """Add a dataset to a channel."""
        resp = await self._client.post(self._url(f"/channels/{channel_id}/datasets/{dataset_id}"))
        self._raise_for_status(resp)
        return ChannelDatasetBase.model_validate(resp.json())

    async def create_glossary_terms_bulk(
        self, channel_id: int, terms: list[GlossaryTermBase]
    ) -> list[GlossaryTerm]:
        """Create multiple glossary terms."""
        resp = await self._client.post(
            self._url(f"/channels/{channel_id}/terms/bulk"),
            json=[term.model_dump(mode="json") for term in terms],
        )
        self._raise_for_status(resp)
        return _glossary_terms_adapter.validate_python(resp.json())

    async def update_glossary_terms_bulk(
        self, terms: list[GlossaryTermUpdateBulk]
    ) -> list[GlossaryTerm]:
        """Update multiple glossary terms."""
        resp = await self._client.post(
            self._url("/terms/bulk"),
            json=[term.model_dump(mode="json", exclude_unset=True) for term in terms],
        )
        self._raise_for_status(resp)
        return _glossary_terms_adapter.validate_python(resp.json())

    async def delete_glossary_terms_bulk(self, term_ids: list[int]) -> None:
        """Delete multiple glossary terms."""
        resp = await self._client.request(
            "DELETE",
            self._url("/terms/bulk"),
            json=term_ids,
        )
        self._raise_for_status(resp)


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
