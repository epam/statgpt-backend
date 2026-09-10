import pytest

from statgpt.common.schemas.data_query_tool import DataQueryDetails, DataQueryExplorerLink
from statgpt.common.schemas.enums import ExplorerLinkPolicy, InvocationSource

_SURFACES = ["stage", "agent", "mcp"]


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


class TestForSource:
    """The tool response policy follows the flow the call belongs to."""

    def test_agent_source_reads_the_agent_policy(self):
        config = DataQueryExplorerLink(
            agent=ExplorerLinkPolicy.only_when_no_data, mcp=ExplorerLinkPolicy.never
        )
        assert config.for_source(InvocationSource.AGENT) is ExplorerLinkPolicy.only_when_no_data

    def test_mcp_source_reads_the_mcp_policy(self):
        config = DataQueryExplorerLink(
            agent=ExplorerLinkPolicy.only_when_no_data, mcp=ExplorerLinkPolicy.never
        )
        assert config.for_source(InvocationSource.MCP) is ExplorerLinkPolicy.never

    def test_stage_policy_is_not_a_tool_response_policy(self):
        """The stage setting must never leak into what the model reads."""
        config = DataQueryExplorerLink(
            stage=ExplorerLinkPolicy.always, agent=ExplorerLinkPolicy.never
        )
        assert config.for_source(InvocationSource.AGENT) is ExplorerLinkPolicy.never


class TestYamlParsing:
    def test_camel_case_block_parses(self):
        details = DataQueryDetails.model_validate(
            {"explorerLink": {"stage": "always", "agent": "only_when_no_data", "mcp": "never"}}
        )
        assert details.explorer_link.stage is ExplorerLinkPolicy.always
        assert details.explorer_link.agent is ExplorerLinkPolicy.only_when_no_data
        assert details.explorer_link.mcp is ExplorerLinkPolicy.never

    def test_partial_block_leaves_the_rest_on_the_default(self):
        details = DataQueryDetails.model_validate({"explorerLink": {"agent": "never"}})
        assert details.explorer_link.agent is ExplorerLinkPolicy.never
        assert details.explorer_link.stage is ExplorerLinkPolicy.always
        assert details.explorer_link.mcp is ExplorerLinkPolicy.always
