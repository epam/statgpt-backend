from datetime import datetime, timezone
from io import StringIO
from types import SimpleNamespace

import pandas as pd

from statgpt.app.mcp.attachments import data_query_artifact_to_resources
from statgpt.app.schemas.tool_artifact import DataQueryArtifact

_FIXED_TS = datetime(2026, 4, 20, 15, 30, 0, tzinfo=timezone.utc)
_FIXED_TS_STR = _FIXED_TS.strftime("%Y%m%dT%H%M%SZ")


def _make_artifact(data_responses: dict) -> DataQueryArtifact:
    # Bypass pydantic validation — the converter only reads data_responses.
    return DataQueryArtifact.model_construct(data_responses=data_responses)


def _response(
    resource_path: str,
    df: pd.DataFrame,
    created_at: datetime = _FIXED_TS,
) -> SimpleNamespace:
    return SimpleNamespace(
        resource_path=resource_path,
        visual_dataframe=df,
        csv_dataframe=df,
        created_at=created_at,
    )


def test_single_dataset_produces_one_csv_resource():
    df = pd.DataFrame({"country": ["FR", "DE"], "value": [1, 2]})
    artifact = _make_artifact({"ds1": _response("IMF:CPI(2.0.0)", df)})

    resources = data_query_artifact_to_resources(artifact)

    assert len(resources) == 1
    resource = resources[0].resource
    assert resource.mimeType == "text/csv"
    assert str(resource.uri) == f"statgpt://data_query/IMF%3ACPI%282.0.0%29/{_FIXED_TS_STR}.csv"
    parsed = pd.read_csv(StringIO(resource.text))
    assert list(parsed.columns) == ["country", "value"]
    assert parsed.shape == (2, 2)


def test_multiple_datasets_preserve_insertion_order():
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"b": [2]})
    artifact = _make_artifact(
        {
            "ds1": _response("IMF:CPI(1.0.0)", df1),
            "ds2": _response("BIS:IR(2.1.0)", df2),
        }
    )

    resources = data_query_artifact_to_resources(artifact)

    assert [str(r.resource.uri) for r in resources] == [
        f"statgpt://data_query/IMF%3ACPI%281.0.0%29/{_FIXED_TS_STR}.csv",
        f"statgpt://data_query/BIS%3AIR%282.1.0%29/{_FIXED_TS_STR}.csv",
    ]


def test_each_response_uses_its_own_created_at():
    ts1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    artifact = _make_artifact(
        {
            "ds1": _response("IMF:CPI(1.0.0)", pd.DataFrame({"a": [1]}), created_at=ts1),
            "ds2": _response("BIS:IR(2.1.0)", pd.DataFrame({"b": [2]}), created_at=ts2),
        }
    )

    resources = data_query_artifact_to_resources(artifact)

    assert [str(r.resource.uri) for r in resources] == [
        "statgpt://data_query/IMF%3ACPI%281.0.0%29/20260101T000000Z.csv",
        "statgpt://data_query/BIS%3AIR%282.1.0%29/20260102T000000Z.csv",
    ]


def test_empty_dataframes_are_skipped():
    artifact = _make_artifact(
        {
            "empty": _response("IMF:EMPTY(1.0.0)", pd.DataFrame()),
            "ok": _response("IMF:OK(1.0.0)", pd.DataFrame({"x": [1]})),
        }
    )

    resources = data_query_artifact_to_resources(artifact)

    assert len(resources) == 1
    assert (
        str(resources[0].resource.uri)
        == f"statgpt://data_query/IMF%3AOK%281.0.0%29/{_FIXED_TS_STR}.csv"
    )


def test_no_responses_returns_empty_list():
    artifact = _make_artifact({})

    assert data_query_artifact_to_resources(artifact) == []
