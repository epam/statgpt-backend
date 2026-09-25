import pytest

from statgpt.common.schemas.data_query_tool import DataQueryDetails, DataQueryExplorerLink
from statgpt.common.schemas.enums import ExplorerLinkPolicy

_SURFACES = ["stage", "agent"]


class TestDefaults:
    """Every surface defaults to `always`, reproducing the behavior before the config existed."""

    @pytest.mark.parametrize("surface", _SURFACES)
    def test_explorer_link_defaults_to_always(self, surface: str):
        assert getattr(DataQueryExplorerLink(), surface) is ExplorerLinkPolicy.always

    @pytest.mark.parametrize("surface", _SURFACES)
    def test_details_without_the_block_defaults_to_always(self, surface: str):
        """A stored channel config that predates the field must keep linking as it does today."""
        details = DataQueryDetails()
        assert getattr(details.explorer_link, surface) is ExplorerLinkPolicy.always


class TestYamlParsing:
    def test_camel_case_block_parses(self):
        details = DataQueryDetails.model_validate(
            {"explorerLink": {"stage": "never", "agent": "only_when_no_data"}}
        )
        assert details.explorer_link.stage is ExplorerLinkPolicy.never
        assert details.explorer_link.agent is ExplorerLinkPolicy.only_when_no_data

    def test_mcp_structured_content_explorer_link_parses(self):
        details = DataQueryDetails.model_validate(
            {"mcpStructuredContent": {"dataExplorerUrl": "only_when_no_data"}}
        )
        assert (
            details.mcp_structured_content.data_explorer_url is ExplorerLinkPolicy.only_when_no_data
        )
        assert (
            DataQueryDetails().mcp_structured_content.data_explorer_url is ExplorerLinkPolicy.always
        )

    def test_partial_block_leaves_the_rest_on_the_default(self):
        details = DataQueryDetails.model_validate({"explorerLink": {"agent": "never"}})
        assert details.explorer_link.agent is ExplorerLinkPolicy.never
        assert details.explorer_link.stage is ExplorerLinkPolicy.always
