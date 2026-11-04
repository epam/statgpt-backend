"""Rounding of data query values: the opt-in `DATA_QUERY_DECIMAL_PLACES` setting, the values the
agent reads and the UI grid shows, and the CSV download that keeps full precision."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from pydantic import ValidationError

from statgpt.app.chains.data_query.data_query_artifacts_displayer import DataQueryArtifactDisplayer
from statgpt.app.settings.dial_app import DialAppSettings, dial_app_settings
from statgpt.common.data.base import DataResponseStatus
from statgpt.common.schemas.data_query_tool import DataQueryAttachments, ToolAttachment
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus

_ENV = "DATA_QUERY_DECIMAL_PLACES"
_FULL = pd.DataFrame({"REF_AREA": ["FR"], "2020": [0.9431918]})
_ROUNDED = pd.DataFrame({"REF_AREA": ["FR"], "2020": [0.943]})


def test_rounding_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)

    assert DialAppSettings().data_query_decimal_places is None


def test_rounding_is_turned_on_with_the_env_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "3")

    assert DialAppSettings().data_query_decimal_places == 3


@pytest.mark.parametrize("value", ["-1", "16", "2.5"])
def test_decimal_places_out_of_range_are_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(_ENV, value)

    with pytest.raises(ValidationError, match="DATA_QUERY_DECIMAL_PLACES"):
        DialAppSettings()


# ~~~ What the agent reads, what the grid shows and what the user downloads ~~~


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        status=DataResponseStatus(
            request_status=DataRequestStatus.SUCCESS, parsing_status=DataParsingStatus.SUCCESS
        ),
        dataset_name="CPI [IMF:CPI]",
        file_name="cpi",
        visual_dataframe=_FULL,
        csv_dataframe=_FULL,
        rounded_visual_dataframe=lambda decimal_places: (
            _FULL if decimal_places is None else _ROUNDED
        ),
        plotly_grid=Mock(return_value=None),
        dimension_ids={"REF_AREA"},
        attribute_ids=set(),
        enrich_attachment_name=lambda name: name,
    )


def _displayer() -> DataQueryArtifactDisplayer:
    config = DataQueryAttachments(
        plotly_grid=ToolAttachment(enabled_str="True", name="Plotly Grid: {dataset_source_id}")
    )
    return DataQueryArtifactDisplayer(
        choice=Mock(),
        config=config,
        chat_config=SimpleNamespace(enable_debug_attachments=False),  # type: ignore[arg-type]
        max_cells=300,
        auth_context=Mock(),
    )


async def test_agent_reads_the_rounded_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dial_app_settings, "data_query_decimal_places", 3)
    artifact = SimpleNamespace(data_responses={"ds1": _response()})

    content = await _displayer().get_system_message_content([artifact])  # type: ignore[list-item]

    assert "0.943" in content
    assert "0.9431918" not in content
    assert "full precision" not in content


async def test_agent_reads_full_precision_when_rounding_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dial_app_settings, "data_query_decimal_places", None)
    artifact = SimpleNamespace(data_responses={"ds1": _response()})

    content = await _displayer().get_system_message_content([artifact])  # type: ignore[list-item]

    assert "0.9431918" in content


@pytest.mark.parametrize("decimal_places", [3, None])
async def test_ui_grid_is_built_with_the_configured_precision(
    monkeypatch: pytest.MonkeyPatch, decimal_places: int | None
) -> None:
    monkeypatch.setattr(dial_app_settings, "data_query_decimal_places", decimal_places)
    response = _response()

    await _displayer()._attach_plotly_grid(Mock(), response)  # type: ignore[arg-type]

    response.plotly_grid.assert_called_once_with(decimal_places)


async def test_csv_download_keeps_full_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dial_app_settings, "data_query_decimal_places", 3)
    storage = Mock()
    storage.put_csv_from_dataframe = AsyncMock(
        return_value=SimpleNamespace(content_type=None, url="files/cpi.csv")
    )

    attachment = await _displayer()._attach_csv(storage, _response())  # type: ignore[arg-type]

    assert attachment is not None
    name, df = storage.put_csv_from_dataframe.await_args.args
    assert name == "cpi"
    assert df is _FULL
