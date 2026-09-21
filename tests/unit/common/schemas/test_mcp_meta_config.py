import pytest
from pydantic import ValidationError

from statgpt.common.schemas.data_query_tool import DataQueryDetails, DataQueryMcpMeta


class TestDataQueryMcpMeta:
    def test_defaults_publish_both_audiences(self):
        cfg = DataQueryDetails().mcp_meta

        assert cfg.get_namespace() == "statgpt.dialx.ai"
        assert cfg.mcp_app_key == "statgpt.dialx.ai/mcp-app"
        assert cfg.client_key == "statgpt.dialx.ai/client"
        assert cfg.mcp_app.enabled is True
        assert cfg.client.enabled is True

    def test_audiences_are_toggled_independently(self):
        cfg = DataQueryMcpMeta.model_validate(
            {"mcpApp": {"enabledStr": "False"}, "client": {"enabledStr": "True"}}
        )

        assert cfg.mcp_app.enabled is False
        assert cfg.client.enabled is True

    def test_namespace_resolves_an_environment_variable(self, monkeypatch):
        monkeypatch.setenv("MCP_META_NAMESPACE", "data.example.org")

        cfg = DataQueryMcpMeta.model_validate({"namespace": "$env:{MCP_META_NAMESPACE}"})

        assert cfg.mcp_app_key == "data.example.org/mcp-app"

    def test_namespace_trailing_slash_is_dropped(self):
        assert DataQueryMcpMeta.model_validate({"namespace": "statgpt.dialx.ai/"}).client_key == (
            "statgpt.dialx.ai/client"
        )

    @pytest.mark.parametrize("namespace", ["", "statgpt ai", "statgpt.dialx.ai/extra", ".statgpt"])
    def test_unusable_namespace_is_rejected(self, namespace: str):
        # The namespace becomes a `_meta` key prefix, so it is validated at config-load time
        # rather than on every tool call.
        with pytest.raises(ValidationError, match="namespace"):
            DataQueryMcpMeta.model_validate({"namespace": namespace})

    @pytest.mark.parametrize("namespace", ["modelcontextprotocol.io", "mcp", "mcp.something"])
    def test_reserved_namespace_is_rejected(self, namespace: str):
        with pytest.raises(ValidationError, match="reserved"):
            DataQueryMcpMeta.model_validate({"namespace": namespace})
