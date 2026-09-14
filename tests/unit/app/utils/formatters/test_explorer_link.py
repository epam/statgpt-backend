import pandas as pd
import pytest

from statgpt.app.utils.formatters import DatasetQueryFormatter, DatasetQueryFormatterConfig
from statgpt.common.data.base.dataset import DataResponseStatus
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus, ExplorerLinkPolicy

_URL = "https://explorer.example/view?q=1"

_SUCCESS = DataResponseStatus(
    request_status=DataRequestStatus.SUCCESS, parsing_status=DataParsingStatus.SUCCESS
)
_REQUEST_FAILED = DataResponseStatus(
    request_status=DataRequestStatus.FAILED, parsing_status=DataParsingStatus.NA
)
_PARSING_FAILED = DataResponseStatus(
    request_status=DataRequestStatus.SUCCESS, parsing_status=DataParsingStatus.FAILED
)
_EMPTY = DataResponseStatus(
    request_status=DataRequestStatus.SUCCESS, parsing_status=DataParsingStatus.SUCCESS
)

# Every outcome `_format_execution_result` branches on, and whether it delivered rows.
_OUTCOMES = {
    "data_received": (_SUCCESS, True),
    "request_failed": (_REQUEST_FAILED, False),
    "parsing_failed": (_PARSING_FAILED, False),
    "no_rows": (_EMPTY, False),
}
_NO_DATA_OUTCOMES = [name for name, (_, has_data) in _OUTCOMES.items() if not has_data]


class _FakeDataResponse:
    """The slice of `DataResponse` that `_format_execution_result` reads."""

    def __init__(self, status: DataResponseStatus, has_data: bool, url: str | None = _URL):
        self.status = status
        self.visual_dataframe = pd.DataFrame({"value": [1]}) if has_data else pd.DataFrame()
        self.url_query = url

    def get_display_series_count(self) -> int:
        return 1


def _render(policy: ExplorerLinkPolicy, outcome: str, url: str | None = _URL) -> str:
    status, has_data = _OUTCOMES[outcome]
    formatter = DatasetQueryFormatter(
        config=DatasetQueryFormatterConfig(explorer_link=policy), auth_context=None
    )
    lines = formatter._format_execution_result(_FakeDataResponse(status, has_data, url))
    return "\n".join(lines)


def _has_link(rendered: str) -> bool:
    return _URL in rendered


class TestAlways:
    """The default: today's behavior, a link on every outcome."""

    @pytest.mark.parametrize("outcome", list(_OUTCOMES))
    def test_link_rendered(self, outcome: str):
        assert _has_link(_render(ExplorerLinkPolicy.always, outcome))

    def test_is_the_config_default(self):
        assert DatasetQueryFormatterConfig().explorer_link is ExplorerLinkPolicy.always


class TestOnlyWhenNoData:
    """Link out only where StatGPT could not deliver."""

    def test_no_link_on_success(self):
        """The point of the setting: a delivered answer must stand on its own."""
        assert not _has_link(_render(ExplorerLinkPolicy.only_when_no_data, "data_received"))

    @pytest.mark.parametrize("outcome", _NO_DATA_OUTCOMES)
    def test_link_kept_where_nothing_was_delivered(self, outcome: str):
        assert _has_link(_render(ExplorerLinkPolicy.only_when_no_data, outcome))


class TestNever:
    @pytest.mark.parametrize("outcome", list(_OUTCOMES))
    def test_no_link_rendered(self, outcome: str):
        assert not _has_link(_render(ExplorerLinkPolicy.never, outcome))


@pytest.mark.parametrize("policy", list(ExplorerLinkPolicy))
@pytest.mark.parametrize("outcome", list(_OUTCOMES))
def test_nothing_rendered_when_the_response_carries_no_url(
    policy: ExplorerLinkPolicy, outcome: str
):
    assert "](" not in _render(policy, outcome, url=None)


@pytest.mark.parametrize("policy", list(ExplorerLinkPolicy))
@pytest.mark.parametrize(
    "outcome,marker",
    [
        ("data_received", "Data received"),
        ("request_failed", "The request to the data source failed."),
        ("parsing_failed", "parsing the response failed"),
        ("no_rows", "does not contain any data"),
    ],
)
def test_the_policy_gates_only_the_link(policy: ExplorerLinkPolicy, outcome: str, marker: str):
    """Which outcome branch is reported must not depend on the link policy."""
    assert marker in _render(policy, outcome)
