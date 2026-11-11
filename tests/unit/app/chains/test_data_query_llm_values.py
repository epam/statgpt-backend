"""The data query values the Supreme Agent reads, transformed as the dataset config sets, next to
the user's attachments (the UI grid, the CSV download) that keep the values the source stores."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import plotly.graph_objects as go

from statgpt.app.chains.data_query.data_query_artifacts_displayer import DataQueryArtifactDisplayer
from statgpt.common.data.base import DataResponseStatus
from statgpt.common.schemas.data_query_tool import DataQueryAttachments, ToolAttachment
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus

# A fraction of one at full precision, as the source stores it, and the rounded percentage the
# LLM reads.
_SOURCE = pd.DataFrame({"REF_AREA": ["FR"], "2020": [0.009431918]})
_TRANSFORMED = pd.DataFrame({"REF_AREA": ["FR"], "UNIT": ["%"], "2020": [0.943]})


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        status=DataResponseStatus(
            request_status=DataRequestStatus.SUCCESS, parsing_status=DataParsingStatus.SUCCESS
        ),
        dataset_name="CPI [IMF:CPI]",
        file_name="cpi",
        visual_dataframe=_SOURCE,
        csv_dataframe=_SOURCE,
        llm_visual_dataframe=_TRANSFORMED,
        llm_csv_dataframe=_TRANSFORMED,
        plotly_grid=go.Figure(go.Table(cells=dict(values=[[0.009431918]]))),
        dimension_ids={"REF_AREA"},
        attribute_ids={"UNIT"},
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


def _storage() -> Mock:
    storage = Mock()
    storage.put_json = AsyncMock(return_value=SimpleNamespace(url="files/cpi.json"))
    storage.put_csv_from_dataframe = AsyncMock(
        return_value=SimpleNamespace(content_type=None, url="files/cpi.csv")
    )
    return storage


async def test_agent_reads_only_the_transformed_values() -> None:
    artifact = SimpleNamespace(data_responses={"ds1": _response()})

    content = await _displayer().get_system_message_content([artifact])  # type: ignore[list-item]

    assert "0.943" in content
    assert "ATTRIBUTE:UNIT" in content
    assert "0.009431918" not in content


async def test_ui_grid_shows_the_source_values() -> None:
    response = _response()
    storage = _storage()

    attachment = await _displayer()._attach_plotly_grid(storage, response)  # type: ignore[arg-type]

    assert attachment is not None
    _, chart_json = storage.put_json.await_args.args
    assert chart_json == response.plotly_grid.to_json()


async def test_csv_download_keeps_the_source_values() -> None:
    storage = _storage()

    attachment = await _displayer()._attach_csv(storage, _response())  # type: ignore[arg-type]

    assert attachment is not None
    name, df = storage.put_csv_from_dataframe.await_args.args
    assert name == "cpi"
    assert df is _SOURCE
