"""Tests for the Data Query eval attachments displayer.

These files are debug-only output, so the displayer has two obligations beyond attaching them:
touch nothing when debug attachments are off, and never let a storage failure cost the user
their answer.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from statgpt.app.chains.data_query import eval_attachments_displayer as module
from statgpt.app.chains.data_query.eval_attachments_displayer import (
    DataQueryEvalAttachmentsDisplayer,
)
from statgpt.app.schemas.discovery_datasets import DiscoveryDatasetsEvalAttachment
from statgpt.app.schemas.query_builder import DataQueryEvalAttachment, QueryBuilderAgentState
from statgpt.app.schemas.tool_artifact import DataQueryArtifact
from statgpt.common.utils import MediaTypes


def _artifact(*, with_discovery: bool = False) -> DataQueryArtifact:
    return DataQueryArtifact(
        state=QueryBuilderAgentState(),
        data_responses={},
        eval_attachment=DataQueryEvalAttachment(),
        discovery_datasets_eval_attachment=(
            DiscoveryDatasetsEvalAttachment(query="gdp") if with_discovery else None
        ),
    )


class _Choice:
    """Records the attachments the displayer adds, in the order it adds them."""

    def __init__(self) -> None:
        self.attachments: list[dict[str, Any]] = []

    def add_attachment(self, **kwargs: Any) -> None:
        self.attachments.append(kwargs)


class _Storage:
    """Stands in for the DIAL attachments storage, reporting what was uploaded."""

    def __init__(self, *, fail_for: str | None = None) -> None:
        self._fail_for = fail_for
        self.uploaded: list[str] = []

    async def put_json(self, name: str, content: str) -> SimpleNamespace:
        if self._fail_for is not None and self._fail_for in name:
            raise RuntimeError("upload exploded")
        self.uploaded.append(name)
        # The real storage appends a uuid and the extension, hence the shape of the url.
        return SimpleNamespace(url=f"files/bucket/{name}-uuid.json")


class _Factory:
    """Stands in for `attachments_storage_factory`, reporting whether it was ever entered."""

    def __init__(self, storage: _Storage | None = None, *, error: Exception | None = None) -> None:
        self.storage = storage if storage is not None else _Storage()
        self._error = error
        self.entered = False

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        @asynccontextmanager
        async def factory(api_key: str):
            self.entered = True
            if self._error is not None:
                raise self._error
            yield self.storage

        monkeypatch.setattr(module, "attachments_storage_factory", factory)


def _displayer(choice: _Choice, *, enabled: bool = True) -> DataQueryEvalAttachmentsDisplayer:
    auth_context = SimpleNamespace(api_key="key")
    return DataQueryEvalAttachmentsDisplayer(
        choice=choice,  # type: ignore[arg-type]
        auth_context=auth_context,  # type: ignore[arg-type]
        enabled=enabled,
    )


def _filenames(choice: _Choice) -> set[str]:
    # Attachments are gathered per tool call, so only the set is meaningful.
    return {attachment["title"] for attachment in choice.attachments}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the gate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_debug_attachments_off_never_opens_a_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate sits ahead of the storage, so a normal turn pays nothing for this."""
    factory = _Factory()
    factory.install(monkeypatch)
    choice = _Choice()

    await _displayer(choice, enabled=False).display({"call-1": _artifact()})

    assert factory.entered is False
    assert choice.attachments == []


async def test_nothing_to_report_never_opens_a_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _Factory()
    factory.install(monkeypatch)
    choice = _Choice()

    await _displayer(choice).display({})

    assert factory.entered is False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ what is attached ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_each_tool_call_gets_its_own_eval_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn with several data queries must be judgeable per call, not in aggregate."""
    factory = _Factory()
    factory.install(monkeypatch)
    choice = _Choice()

    await _displayer(choice).display(
        {"call-1": _artifact(with_discovery=True), "call-2": _artifact()}
    )

    assert _filenames(choice) == {
        "Data Query Eval data: call-1",
        "Discovery Datasets Eval data: call-1",
        "Data Query Eval data: call-2",
    }
    assert sorted(factory.storage.uploaded) == [
        "data_query_eval_attachment_call-1.json",
        "data_query_eval_attachment_call-2.json",
        "discovery_datasets_eval_attachment_call-1.json",
    ]
    for attachment in choice.attachments:
        assert attachment["type"] == MediaTypes.JSON
        assert attachment["url"].startswith("files/bucket/")


async def test_a_call_without_a_discovery_lookup_reports_the_query_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`None` means the channel has no discovery lookup - there is nothing to report."""
    factory = _Factory()
    factory.install(monkeypatch)
    choice = _Choice()

    await _displayer(choice).display({"call-1": _artifact()})

    assert factory.storage.uploaded == ["data_query_eval_attachment_call-1.json"]
    assert _filenames(choice) == {"Data Query Eval data: call-1"}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ failures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_failed_upload_does_not_cost_the_other_call_its_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _Factory(_Storage(fail_for="call-1"))
    factory.install(monkeypatch)
    choice = _Choice()

    await _displayer(choice).display({"call-1": _artifact(), "call-2": _artifact()})

    assert _filenames(choice) == {"Data Query Eval data: call-2"}


async def test_an_unreachable_storage_does_not_break_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Debug-only output must never fail the answer the user is waiting on."""
    factory = _Factory(error=RuntimeError("no bucket"))
    factory.install(monkeypatch)
    choice = _Choice()

    await _displayer(choice).display({"call-1": _artifact()})

    assert factory.entered is True
    assert choice.attachments == []
