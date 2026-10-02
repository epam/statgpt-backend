import pytest
from pydantic import ValidationError

from statgpt.common.schemas.data_query_tool import DataQueryDetails
from statgpt.common.schemas.mcp_meta import McpMeta
from statgpt.common.schemas.tool_details import AvailableDatasetsDetails


class TestMcpMeta:
    def test_defaults_keep_the_client_payload_off(self):
        cfg = McpMeta()

        assert cfg.get_namespace() == "statgpt.dialx.ai"
        assert cfg.client_key == "statgpt.dialx.ai/client"
        assert cfg.client.enabled is False

    def test_client_payload_is_opt_in(self):
        cfg = McpMeta.model_validate({"client": {"enabledStr": "True"}})

        assert cfg.client.enabled is True

    def test_namespace_resolves_an_environment_variable(self, monkeypatch):
        monkeypatch.setenv("MCP_META_NAMESPACE", "data.example.org")

        cfg = McpMeta.model_validate({"namespace": "$env:{MCP_META_NAMESPACE}"})

        assert cfg.client_key == "data.example.org/client"

    def test_namespace_trailing_slash_is_dropped(self):
        assert McpMeta.model_validate({"namespace": "statgpt.dialx.ai/"}).client_key == (
            "statgpt.dialx.ai/client"
        )

    @pytest.mark.parametrize("namespace", ["", "statgpt ai", "statgpt.dialx.ai/extra", ".statgpt"])
    def test_unusable_namespace_is_rejected(self, namespace: str):
        # The namespace becomes a `_meta` key prefix, so it is validated at config-load time
        # rather than on every tool call.
        with pytest.raises(ValidationError, match="namespace"):
            McpMeta.model_validate({"namespace": namespace})

    @pytest.mark.parametrize("namespace", ["modelcontextprotocol.io", "mcp", "mcp.something"])
    def test_reserved_namespace_is_rejected(self, namespace: str):
        with pytest.raises(ValidationError, match="reserved"):
            McpMeta.model_validate({"namespace": namespace})


class TestToolMcpMeta:
    def test_data_query_adds_the_mcp_app_key(self):
        cfg = DataQueryDetails().mcp_meta

        assert cfg.mcp_app_key == "statgpt.dialx.ai/mcp-app"
        assert cfg.client_key == "statgpt.dialx.ai/client"
        # The MCP-App payload has no toggle: it follows the tool's `mcp_app_resource_uri`.
        assert cfg.client.enabled is False

    def test_available_datasets_client_payload_is_opt_in(self):
        details = AvailableDatasetsDetails.model_validate(
            {"mcpMeta": {"namespace": "data.example.org", "client": {"enabledStr": "True"}}}
        )

        assert details.mcp_meta.client.enabled is True
        assert details.mcp_meta.client_key == "data.example.org/client"
        assert AvailableDatasetsDetails().mcp_meta.client.enabled is False
