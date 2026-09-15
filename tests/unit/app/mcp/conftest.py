import pytest

from statgpt.app.settings.mcp import mcp_settings


@pytest.fixture(autouse=True)
def _disable_mcp_guards(monkeypatch):
    """Turn off the cross-cutting MCP guards for the whole MCP test tree.

    Rate limiting and the payload budget run on every model-facing tool ``run()``. Tests that
    exercise tool behaviour must not be throttled by the process-wide limiter (whose state would
    otherwise leak between tests) or have their results trimmed. The dedicated guard tests
    re-enable each guard explicitly.
    """
    monkeypatch.setattr(mcp_settings, "mcp_rate_limit_enabled", False)
    monkeypatch.setattr(mcp_settings, "mcp_payload_budget_enabled", False)
