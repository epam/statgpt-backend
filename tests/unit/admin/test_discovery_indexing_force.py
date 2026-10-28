"""Tests for reading the `force` query of the discovery indexing trigger.

`force` names the stages to redo. A boolean is still read, for callers written against the old
`force: bool`, until it is removed.
"""

import logging

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from statgpt.admin.routers.discovery_datasets import _force_stages
from statgpt.common.schemas import DiscoveryIndexingStage

_SUMMARIZATION = DiscoveryIndexingStage.SUMMARIZATION.value
_PUBLISHING = DiscoveryIndexingStage.PUBLISHING.value


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.post("/trigger")
    async def trigger(
        force: frozenset[DiscoveryIndexingStage] = Depends(_force_stages),
    ) -> list[str]:
        return sorted(force)

    return TestClient(app)


def _post(client: TestClient, *force: str) -> tuple[int, object]:
    response = client.post("/trigger", params=[("force", value) for value in force])
    return response.status_code, response.json()


@pytest.mark.parametrize(
    ("force", "expected"),
    [
        ((), []),
        ((_SUMMARIZATION,), [_SUMMARIZATION]),
        ((_PUBLISHING,), [_PUBLISHING]),
        ((_SUMMARIZATION, _PUBLISHING), [_PUBLISHING, _SUMMARIZATION]),
        ((_PUBLISHING, _PUBLISHING), [_PUBLISHING]),
    ],
)
def test_stages_are_read_from_the_repeated_parameter(
    client: TestClient, force: tuple[str, ...], expected: list[str]
) -> None:
    assert _post(client, *force) == (200, expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", [_PUBLISHING, _SUMMARIZATION]),
        ("True", [_PUBLISHING, _SUMMARIZATION]),
        ("1", [_PUBLISHING, _SUMMARIZATION]),
        ("yes", [_PUBLISHING, _SUMMARIZATION]),
        ("false", []),
        ("0", []),
        ("off", []),
    ],
)
def test_a_deprecated_boolean_is_still_read_and_warned_about(
    client: TestClient, caplog: pytest.LogCaptureFixture, value: str, expected: list[str]
) -> None:
    with caplog.at_level(logging.WARNING):
        assert _post(client, value) == (200, expected)

    assert "deprecated" in caplog.text


@pytest.mark.parametrize(
    "force",
    [
        pytest.param(("true", _SUMMARIZATION), id="boolean with a stage"),
        pytest.param(("true", "false"), id="two booleans"),
        pytest.param(("validation",), id="unknown stage"),
        pytest.param((_SUMMARIZATION, "everything"), id="unknown among stages"),
    ],
)
def test_an_unreadable_value_is_refused_with_422(
    client: TestClient, force: tuple[str, ...]
) -> None:
    status_code, body = _post(client, *force)

    assert status_code == 422
    assert isinstance(body, dict)
    (error,) = body["detail"]
    assert error["loc"] == ["query", "force"]
