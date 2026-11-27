"""Opt-in refresh uses normal writes without expanding source configuration defaults."""

import datetime
import json
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import yaml

from statgpt.cli.commands import content
from statgpt.cli.shared.admin_client import AdminClient
from statgpt.cli.shared.batch_report import BatchItemStatus, BatchPartialFailureError, BatchReport
from statgpt.common.schemas import (
    Channel,
    ChannelConfig,
    ChannelDatasetUpdateResult,
    ChannelDatasetUpdateStatus,
    DataSet,
    DataSetUpdateResponse,
    DataSource,
    DataSourceType,
)
from statgpt.common.schemas.dataset import Status

_NOW = datetime.datetime(2026, 1, 1)


def _channel(deployment_id: str = "channel-a", *, id: int = 11) -> Channel:
    return Channel(
        id=id,
        created_at=_NOW,
        updated_at=_NOW,
        deployment_id=deployment_id,
        title=deployment_id,
        description="",
        llm_model="gpt-4o",
        details=ChannelConfig.model_validate(
            {
                "supremeAgent": {
                    "name": "bot",
                    "domain": "statistics",
                    "terminologyDomain": "statistics",
                },
                "locale": "en",
            }
        ),
    )


def _channel_config(deployment_id: str = "channel-a") -> dict[str, Any]:
    return {
        "deployment_id": deployment_id,
        "title": deployment_id,
        "llm_model": "gpt-4o",
        "details": {
            "supremeAgent": {
                "name": "bot",
                "domain": "statistics",
                "terminologyDomain": "statistics",
            }
        },
    }


def _source(title: str = "source-a") -> DataSource:
    return DataSource(
        id=3,
        created_at=_NOW,
        updated_at=_NOW,
        title=title,
        description="",
        type_id=1,
        details={
            "sdmxConfig": {"id": "sample", "url": "https://example.invalid", "name": "sample"}
        },
        type=DataSourceType(id=1, created_at=_NOW, updated_at=_NOW, name="SDMX21", description=""),
    )


def _dataset(title: str = "DF_A") -> DataSet:
    return DataSet(
        id=7,
        id_=uuid.uuid5(uuid.NAMESPACE_URL, title),
        created_at=_NOW,
        updated_at=_NOW,
        title=title,
        data_source_id=3,
        data_source=None,
        details={
            "urn": {"agencyId": "AG", "resourceId": title, "version": "1.0"},
            "dimensions": {
                "INDICATOR": {"dimensionType": "INDICATOR"},
                "TIME_PERIOD": {"dimensionType": "TIME_PERIOD"},
                "FREQ": {"dimensionType": "NON_INDICATOR", "subtype": "FREQUENCY"},
            },
        },
        status=Status(status="online"),
    )


def _dataset_config(dataset: DataSet) -> dict[str, Any]:
    return {
        "id_": str(dataset.id_),
        "title": dataset.title,
        "dataSource": "source-a",
        "details": deepcopy(dataset.details),
    }


def _write_datasets(config_dir: Path, configs: list[dict[str, Any]]) -> None:
    datasets_dir = config_dir / "datasets"
    datasets_dir.mkdir(parents=True)
    (datasets_dir / "datasets.yaml").write_text(
        yaml.safe_dump({"dataSets": configs}), encoding="utf-8"
    )


def _client() -> AsyncMock:
    client = AsyncMock(spec=AdminClient)
    client.get_channels.return_value = [_channel()]
    client.get_data_sources.return_value = [_source()]
    client.get_data_source_types.return_value = [_source().type]
    client.update_channel.return_value = _channel()
    client.create_channel.return_value = _channel("new-channel", id=12)
    client.update_data_source.return_value = _source()
    client.create_data_source.return_value = _source("new-source")
    client.update_dataset.return_value = DataSetUpdateResponse(dataset=_dataset())
    client.create_dataset.return_value = _dataset("DF_NEW")
    return client


def _comparison(monkeypatch, name: str, update_unchanged: bool) -> Mock:
    comparison = Mock(return_value=False)
    if update_unchanged:
        comparison.side_effect = AssertionError("refresh must bypass only change detection")
    monkeypatch.setattr(content, name, comparison)
    return comparison


def test_refresh_flag_is_opt_in_and_does_not_imply_yes() -> None:
    assert content.init_command.parse_args("")["update_unchanged"] is False
    parsed = content.init_command.parse_args("--only channels --update-unchanged")
    assert parsed["update_unchanged"] is True
    assert parsed["yes"] is False
    assert parsed["verify"] is False
    help_text = content.init_command.get_help()
    assert "--update-unchanged" in help_text
    assert "does not skip prompts" in help_text
    assert "reindexing" in help_text


@pytest.mark.asyncio
@pytest.mark.parametrize("update_unchanged", [False, True])
async def test_channel_refresh_updates_existing_but_preserves_creation(
    monkeypatch, update_unchanged: bool
) -> None:
    client = _client()
    comparison = _comparison(monkeypatch, "_channel_changed", update_unchanged)
    report = BatchReport(title="Summary")
    existing_cfg, new_cfg = _channel_config(), _channel_config("new-channel")

    channels = await content._process_channels(
        client,
        report,
        {"channels": [existing_cfg, new_cfg]},
        {},
        None,
        False,
        "",
        update_unchanged=update_unchanged,
    )

    assert set(channels) == {"channel-a", "new-channel"}
    client.create_channel.assert_awaited_once_with(new_cfg)
    if update_unchanged:
        client.update_channel.assert_awaited_once_with(11, existing_cfg)
        comparison.assert_not_called()
        assert report.counts().get(BatchItemStatus.UNCHANGED, 0) == 0
    else:
        client.update_channel.assert_not_awaited()
        comparison.assert_called_once()
        assert [
            item.item_id for item in report.items if item.status == BatchItemStatus.UNCHANGED
        ] == ["channel-a"]
    assert not report.failed


@pytest.mark.asyncio
@pytest.mark.parametrize("update_unchanged", [False, True])
async def test_source_refresh_updates_existing_but_preserves_creation(
    monkeypatch, update_unchanged: bool
) -> None:
    client = _client()
    comparison = _comparison(monkeypatch, "_data_source_changed", update_unchanged)
    report = BatchReport(title="Summary")
    configs = [
        {"title": title, "type": "SDMX21", "details": {}} for title in ["source-a", "new-source"]
    ]

    sources = await content._process_data_sources(
        client,
        report,
        {"dataSources": configs},
        update_unchanged=update_unchanged,
    )

    assert set(sources) == {"source-a", "new-source"}
    client.create_data_source.assert_awaited_once_with(configs[1])
    if update_unchanged:
        client.update_data_source.assert_awaited_once_with(3, configs[0])
        comparison.assert_not_called()
        assert report.counts().get(BatchItemStatus.UNCHANGED, 0) == 0
    else:
        client.update_data_source.assert_not_awaited()
        comparison.assert_called_once()
        assert [
            item.item_id for item in report.items if item.status == BatchItemStatus.UNCHANGED
        ] == ["source-a"]
    assert not report.failed


@pytest.mark.asyncio
@pytest.mark.parametrize("update_unchanged", [False, True])
async def test_dataset_refresh_updates_existing_but_preserves_creation(
    tmp_path, monkeypatch, update_unchanged: bool
) -> None:
    client = _client()
    comparison = _comparison(monkeypatch, "_dataset_changed", update_unchanged)
    report = BatchReport(title="Summary")
    existing, new = _dataset(), _dataset("DF_NEW")
    _write_datasets(tmp_path, [_dataset_config(existing), _dataset_config(new)])

    touched = await content._process_datasets(
        client,
        report,
        "sample",
        str(tmp_path),
        {"source-a": _source()},
        {},
        {str(existing.id_): existing},
        None,
        link_channels=False,
        update_unchanged=update_unchanged,
    )

    assert [dataset.title for _, dataset in touched] == ["DF_A", "DF_NEW"]
    client.create_dataset.assert_awaited_once()
    if update_unchanged:
        client.update_dataset.assert_awaited_once()
        assert client.update_dataset.call_args.args[0] == existing.id
        comparison.assert_not_called()
        assert report.counts().get(BatchItemStatus.UNCHANGED, 0) == 0
    else:
        client.update_dataset.assert_not_awaited()
        comparison.assert_called_once()
        assert [
            item.item_id for item in report.items if item.status == BatchItemStatus.UNCHANGED
        ] == ["AG:DF_A(1.0)"]
    assert not report.failed


@pytest.mark.asyncio
async def test_refresh_flag_flows_from_handler_to_selected_processors(
    tmp_path, monkeypatch
) -> None:
    config_dir = tmp_path / "config"
    client_dir = config_dir / "clients" / "sample"
    client_dir.mkdir(parents=True)
    monkeypatch.setattr(content.cli_settings, "config_dir", str(config_dir))
    monkeypatch.setattr(content.utils, "read_yaml", lambda path: {})
    monkeypatch.setattr(content.utils, "optional_read_yaml", lambda path: None)
    client = _client()
    context = AsyncMock()
    context.__aenter__.return_value = client
    client.health_check.return_value = True
    client.get_datasets.return_value = []
    monkeypatch.setattr(content, "get_admin_client", lambda: context)
    channels = AsyncMock(return_value={})
    sources = AsyncMock(return_value={})
    datasets = AsyncMock(return_value=[])
    monkeypatch.setattr(content, "_process_channels", channels)
    monkeypatch.setattr(content, "_process_data_sources", sources)
    monkeypatch.setattr(content, "_process_datasets", datasets)

    await content.init_handler(
        client_id="sample",
        only="channels,datasets",
        yes=True,
        update_unchanged=True,
    )

    for processor in [channels, sources, datasets]:
        processor.assert_awaited_once()
        assert processor.call_args.kwargs["update_unchanged"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("update_unchanged", [False, True])
async def test_existing_source_requires_details_before_comparison_and_isolates_failure(
    monkeypatch, update_unchanged: bool
) -> None:
    client = _client()
    client.get_data_sources.return_value = [_source(), _source("source-b")]
    comparison = _comparison(monkeypatch, "_data_source_changed", update_unchanged)
    configs = [
        {"title": "source-a", "type": "SDMX21"},
        {"title": "source-b", "type": "SDMX21", "details": {}},
    ]
    report = BatchReport(title="Summary")

    sources = await content._process_data_sources(
        client,
        report,
        {"dataSources": configs},
        update_unchanged=update_unchanged,
    )

    assert set(sources) == {"source-b"}
    assert [item.item_id for item in report.failed] == ["source-a"]
    assert "missing required 'details'" in report.failed[0].message
    if update_unchanged:
        comparison.assert_not_called()
        client.update_data_source.assert_awaited_once_with(3, configs[1])
    else:
        comparison.assert_called_once_with(configs[1], client.get_data_sources.return_value[1])
        client.update_data_source.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("update_unchanged", [False, True])
@pytest.mark.parametrize("details", [{}, None])
async def test_existing_source_preserves_present_empty_or_null_details(
    monkeypatch, update_unchanged: bool, details: Any
) -> None:
    client = _client()
    comparison = Mock(return_value=True)
    monkeypatch.setattr(content, "_data_source_changed", comparison)
    config = {"title": "source-a", "type": "SDMX21", "details": details}
    report = BatchReport(title="Summary")

    await content._process_data_sources(
        client,
        report,
        {"dataSources": [config]},
        update_unchanged=update_unchanged,
    )

    client.update_data_source.assert_awaited_once_with(3, config)
    assert client.update_data_source.call_args.args[1]["details"] is details
    assert not report.failed


@pytest.mark.asyncio
@pytest.mark.parametrize("update_unchanged", [False, True])
async def test_missing_details_guard_does_not_change_source_creation(
    update_unchanged: bool,
) -> None:
    client = _client()
    config = {"title": "new-source", "type": "SDMX21"}
    report = BatchReport(title="Summary")

    await content._process_data_sources(
        client,
        report,
        {"dataSources": [config]},
        update_unchanged=update_unchanged,
    )

    client.create_data_source.assert_awaited_once_with(config)
    client.update_data_source.assert_not_awaited()
    assert not report.failed


@pytest.mark.asyncio
@pytest.mark.parametrize("pin_locale", [False, True])
async def test_channel_refresh_sends_sparse_assembled_override_config_without_merging_legacy_defaults(
    pin_locale: bool,
) -> None:
    requests: list[httpx.Request] = []
    stored = _channel().model_dump(mode="json", by_alias=True)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [stored]})
        return httpx.Response(200, json=stored)

    config = _channel_config()
    if pin_locale:
        config["details"]["locale"] = "en"
    tools = {
        "tools": [
            {
                "channels": ["channel-a"],
                "type": "plain_content",
                "name": "Info",
                "description": "Information",
                "enabled": False,
                "details": {"filePath": "info.md", "replaceEnvs": False},
            }
        ]
    }
    onboarding = {
        "configurations": [
            {
                "channels": ["channel-a"],
                "introMessage": "Welcome",
                "completionMessage": "Done",
                "topicSelectionPrompt": "Choose",
                "topics": {},
            }
        ]
    }
    report = BatchReport(title="Summary")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = AdminClient(http, "http://admin.invalid")
        await content._process_channels(
            client,
            report,
            {"channels": [config]},
            tools,
            onboarding,
            False,
            "",
            update_unchanged=True,
        )

    writes = [request for request in requests if request.method == "POST"]
    assert len(writes) == 1
    assert writes[0].url.path == "/admin/api/v1/channels/11"
    assert not writes[0].url.params
    payload = json.loads(writes[0].content)
    assert payload == config
    assert "locale" in stored["details"]
    assert ("locale" in payload["details"]) is pin_locale
    if pin_locale:
        assert payload["details"]["locale"] == "en"
    assert payload["details"]["plain_content"]["enabled"] is False
    assert payload["details"]["plain_content"]["details"] == {
        "filePath": "info.md",
        "replaceEnvs": False,
    }
    assert payload["details"]["onboarding"] == {
        key: value for key, value in onboarding["configurations"][0].items() if key != "channels"
    }
    assert "completionButtonTitle" not in payload["details"]["onboarding"]
    assert not report.failed


@pytest.mark.asyncio
async def test_failed_refresh_retains_api_validation_batch_summary_and_failure_exit(
    tmp_path, monkeypatch
) -> None:
    config_dir = tmp_path / "config"
    client_dir = config_dir / "clients" / "sample"
    client_dir.mkdir(parents=True)
    bad, valid = _channel_config(), _channel_config("channel-b")
    bad["llm_model"] = None
    (client_dir / "channels.yaml").write_text(
        yaml.safe_dump({"channels": [bad, valid]}), encoding="utf-8"
    )
    (client_dir / "tools.yaml").write_text("tools: []\n", encoding="utf-8")
    monkeypatch.setattr(content.cli_settings, "config_dir", str(config_dir))
    channels = [_channel(), _channel("channel-b", id=12), _channel("unlisted-channel", id=13)]
    requests: list[httpx.Request] = []
    reports: list[BatchReport] = []
    monkeypatch.setattr(BatchReport, "render", lambda report: reports.append(report))

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [channel.model_dump(mode="json", by_alias=True) for channel in channels]
                },
            )
        if request.url.path.endswith("/11"):
            return httpx.Response(422, json={"detail": "llm_model cannot be null"})
        return httpx.Response(200, json=channels[1].model_dump(mode="json", by_alias=True))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        context = AsyncMock()
        context.__aenter__.return_value = AdminClient(http, "http://admin.invalid")
        monkeypatch.setattr(content, "get_admin_client", lambda: context)
        with pytest.raises(BatchPartialFailureError, match="completed with failures"):
            await content.init_handler(
                client_id="sample", only="channels", yes=True, update_unchanged=True
            )

    writes = [request for request in requests if request.method == "POST"]
    assert [request.url.path for request in writes] == [
        "/admin/api/v1/channels/11",
        "/admin/api/v1/channels/12",
    ]
    assert len(reports) == 1
    assert [item.item_id for item in reports[0].failed] == ["channel-a"]
    assert "HTTP 422" in reports[0].failed[0].message
    assert reports[0].counts() == {BatchItemStatus.FAILED: 1, BatchItemStatus.OK: 1}


@pytest.mark.asyncio
async def test_proxy_refresh_does_not_send_enriched_external_config_when_omitted() -> None:
    source = _source()
    source.type.name = "PROXY_SDMX30"
    source.details["proxyConfig"] = {"agencies": [{"name": "Sample"}], "configs": []}
    config = {
        "title": source.title,
        "type": "PROXY_SDMX30",
        "details": {"sdmxConfig": deepcopy(source.details["sdmxConfig"])},
    }
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/types"):
            return httpx.Response(200, json={"data": [source.type.model_dump(mode="json")]})
        if request.method == "GET":
            return httpx.Response(200, json={"data": [source.model_dump(mode="json")]})
        return httpx.Response(200, json=source.model_dump(mode="json"))

    report = BatchReport(title="Summary")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = AdminClient(http, "http://admin.invalid")
        sources = await content._process_data_sources(
            client, report, {"dataSources": [config]}, update_unchanged=True
        )

    writes = [request for request in requests if request.method == "POST"]
    assert len(requests) == 3
    assert len(writes) == 1
    assert writes[0].url.path == "/admin/api/v1/data-sources/3"
    assert not writes[0].url.params
    payload = json.loads(writes[0].content)
    assert payload == config
    assert "proxyConfig" not in payload["details"]
    assert sources[source.title].details["proxyConfig"] == source.details["proxyConfig"]
    assert not report.failed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, expected_message",
    [
        (ChannelDatasetUpdateStatus.AUTO_UPDATED, "Auto-updated in:"),
        (ChannelDatasetUpdateStatus.NEEDS_REINDEX, "Needs reindex in:"),
        (ChannelDatasetUpdateStatus.NO_VERSION, "No indexed version in:"),
        (ChannelDatasetUpdateStatus.INDEXING_IN_PROGRESS, "Indexing in progress in:"),
    ],
)
async def test_dataset_refresh_reports_normal_propagation_without_starting_reindexing(
    tmp_path, capsys, status: ChannelDatasetUpdateStatus, expected_message: str
) -> None:
    client = _client()
    dataset = _dataset()
    client.update_dataset.return_value = DataSetUpdateResponse(
        dataset=dataset,
        channel_results=[
            ChannelDatasetUpdateResult(channel_dataset_id=5, channel=_channel(), status=status)
        ],
    )
    _write_datasets(tmp_path, [_dataset_config(dataset)])
    report = BatchReport(title="Summary")

    touched = await content._process_datasets(
        client,
        report,
        "sample",
        str(tmp_path),
        {"source-a": _source()},
        {},
        {str(dataset.id_): dataset},
        None,
        link_channels=False,
        update_unchanged=True,
    )

    assert touched == [("AG:DF_A(1.0)", dataset)]
    assert client.update_dataset.call_args.args[0] == dataset.id
    assert [call[0] for call in client.mock_calls] == ["update_dataset"]
    output = capsys.readouterr().out
    assert expected_message in output
    assert "channel-a" in output
    if status == ChannelDatasetUpdateStatus.NEEDS_REINDEX:
        assert "Datasets requiring reindexing:" in output
        assert "AG:DF_A(1.0)" in output
    assert report.counts() == {BatchItemStatus.OK: 1}
