"""Rounding of data query values: the config, the precision a call can ask for instead, the execute
step that applies it to every response, the rounded values the agent reads, and the CSV download
that keeps full precision."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from langchain_core.runnables import RunnableLambda

from statgpt.app.chains.data_query.data_query_artifacts_displayer import DataQueryArtifactDisplayer
from statgpt.app.chains.data_query.parameters import DataQueryParameters
from statgpt.app.chains.data_query.query_builder.factory import QueryBuilderFactory
from statgpt.app.chains.data_query.query_builder.query.execute_query import ExecuteQueryChain
from statgpt.app.chains.data_query.runner import DataQueryRunner
from statgpt.common.data.base import DataResponseStatus
from statgpt.common.schemas.data_query_tool import (
    DataQueryAttachments,
    DataQueryDetails,
    DataQueryValueRounding,
)
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus

_FULL = pd.DataFrame({"REF_AREA": ["FR"], "2020": [0.9431918]})
_ROUNDED = pd.DataFrame({"REF_AREA": ["FR"], "2020": [0.943]})


def test_values_are_rounded_to_three_decimal_places_by_default() -> None:
    assert DataQueryDetails().value_rounding.decimal_places == 3


def test_rounding_can_be_turned_off_in_the_tool_config() -> None:
    details = DataQueryDetails.model_validate({"valueRounding": {"decimalPlaces": None}})

    assert details.value_rounding.decimal_places is None


# ~~~ The precision of one call ~~~


def _chain(decimal_places: int | None) -> ExecuteQueryChain:
    chain = ExecuteQueryChain.__new__(ExecuteQueryChain)
    chain._value_rounding = DataQueryValueRounding(decimal_places=decimal_places)  # type: ignore[attr-defined]
    return chain


@pytest.mark.parametrize(
    ("configured", "requested", "expected"),
    [
        (3, None, 3),
        (3, DataQueryValueRounding(decimal_places=1), 1),
        (3, DataQueryValueRounding(decimal_places=0), 0),
        # A call can ask for full precision where the channel rounds...
        (3, DataQueryValueRounding(decimal_places=None), None),
        # ...and for rounding where the channel turns it off.
        (None, DataQueryValueRounding(decimal_places=2), 2),
        (None, None, None),
    ],
)
def test_a_call_can_override_the_configured_rounding(
    configured: int | None, requested: DataQueryValueRounding | None, expected: int | None
) -> None:
    inputs = {DataQueryParameters.VALUE_ROUNDING: requested}

    assert _chain(configured)._decimal_places(inputs) == expected


def test_the_configured_precision_applies_when_nothing_is_requested() -> None:
    assert _chain(3)._decimal_places({}) == 3


async def test_runner_hands_each_call_its_own_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[DataQueryValueRounding | None] = []

    async def _create_chain(self: QueryBuilderFactory, inputs: dict) -> RunnableLambda:
        def _record(d: dict) -> dict:
            seen.append(d[DataQueryParameters.VALUE_ROUNDING])
            return {DataQueryParameters.RESPONSE_FIELD: "answer"}

        return RunnableLambda(_record)

    monkeypatch.setattr(QueryBuilderFactory, "create_chain", _create_chain)
    channel_config = SimpleNamespace(is_discovery_lookup_available=False)
    runner = DataQueryRunner(DataQueryDetails(), channel_config)  # type: ignore[arg-type]
    inputs: dict = {}

    await runner.run(inputs, "cpi", value_rounding=DataQueryValueRounding(decimal_places=1))
    await runner.run(inputs, "cpi")

    # The same inputs serve both calls, and the second asks for nothing.
    assert seen == [DataQueryValueRounding(decimal_places=1), None]


# ~~~ The execute step ~~~


def _dataset(response: object) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(query=AsyncMock(return_value=response)))


async def test_execute_step_sets_the_precision_on_the_response() -> None:
    response = SimpleNamespace(value_decimal_places=None)
    chain = ExecuteQueryChain._get_data_query_chain(_dataset(response), Mock(), Mock(), 3)  # type: ignore[arg-type]

    result = await chain.ainvoke({})

    assert result is response
    assert response.value_decimal_places == 3


async def test_execute_step_passes_a_missing_response_through() -> None:
    chain = ExecuteQueryChain._get_data_query_chain(_dataset(None), Mock(), Mock(), 3)  # type: ignore[arg-type]

    assert await chain.ainvoke({}) is None


# ~~~ What the agent reads and what the user downloads ~~~


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        status=DataResponseStatus(
            request_status=DataRequestStatus.SUCCESS, parsing_status=DataParsingStatus.SUCCESS
        ),
        dataset_name="CPI [IMF:CPI]",
        file_name="cpi",
        visual_dataframe=_FULL,
        csv_dataframe=_FULL,
        rounded_visual_dataframe=_ROUNDED,
        rounded_csv_dataframe=_ROUNDED,
        dimension_ids={"REF_AREA"},
        attribute_ids=set(),
        enrich_attachment_name=lambda name: name,
    )


def _displayer() -> DataQueryArtifactDisplayer:
    return DataQueryArtifactDisplayer(
        choice=Mock(),
        config=DataQueryAttachments(),
        chat_config=SimpleNamespace(enable_debug_attachments=False),  # type: ignore[arg-type]
        max_cells=300,
        auth_context=Mock(),
    )


async def test_agent_reads_the_rounded_values() -> None:
    artifact = SimpleNamespace(data_responses={"ds1": _response()})

    content = await _displayer().get_system_message_content([artifact])  # type: ignore[list-item]

    assert "0.943" in content
    assert "0.9431918" not in content
    assert "full precision" not in content


async def test_csv_download_keeps_full_precision() -> None:
    storage = Mock()
    storage.put_csv_from_dataframe = AsyncMock(
        return_value=SimpleNamespace(content_type=None, url="files/cpi.csv")
    )

    attachment = await _displayer()._attach_csv(storage, _response())  # type: ignore[arg-type]

    assert attachment is not None
    name, df = storage.put_csv_from_dataframe.await_args.args
    assert name == "cpi"
    assert df is _FULL
