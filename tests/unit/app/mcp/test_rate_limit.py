"""The in-process per-caller rate limiter for model-facing MCP tools (#600): the token-bucket
mechanics, caller identification, and the `enforce_rate_limit` wrapper."""

import hashlib
import hmac
from types import SimpleNamespace

import pytest
from fastmcp.exceptions import ToolError

from statgpt.app.mcp import rate_limit
from statgpt.app.mcp.rate_limit import (
    BucketConfig,
    McpRateLimiter,
    McpRateLimitExceeded,
    McpToolCostClass,
    caller_identity,
    enforce_rate_limit,
)
from statgpt.app.settings.mcp import mcp_settings


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _limiter(clock: _Clock, *, capacity: int = 2, rate: float = 1.0) -> McpRateLimiter:
    configs = {c: BucketConfig(capacity=capacity, refill_rate=rate) for c in McpToolCostClass}
    return McpRateLimiter(configs, time_fn=clock)


# ~~~~~~~~~~~~~ token bucket ~~~~~~~~~~~~~


def test_allows_up_to_burst_then_blocks():
    limiter = _limiter(_Clock(), capacity=2, rate=1.0)
    limiter.acquire("caller", McpToolCostClass.CHEAP)
    limiter.acquire("caller", McpToolCostClass.CHEAP)
    with pytest.raises(McpRateLimitExceeded):
        limiter.acquire("caller", McpToolCostClass.CHEAP)


def test_refills_over_time():
    clock = _Clock()
    limiter = _limiter(clock, capacity=1, rate=1.0)  # one token per second
    limiter.acquire("c", McpToolCostClass.CHEAP)
    with pytest.raises(McpRateLimitExceeded):
        limiter.acquire("c", McpToolCostClass.CHEAP)
    clock.advance(1.0)
    limiter.acquire("c", McpToolCostClass.CHEAP)  # replenished


def test_refill_is_capped_at_capacity():
    clock = _Clock()
    limiter = _limiter(clock, capacity=2, rate=1.0)
    limiter.acquire("c", McpToolCostClass.CHEAP)  # bucket drops from 2 to 1
    clock.advance(100)  # a long idle refills to the cap of 2, not to 101
    limiter.acquire("c", McpToolCostClass.CHEAP)
    limiter.acquire("c", McpToolCostClass.CHEAP)
    with pytest.raises(McpRateLimitExceeded):
        limiter.acquire("c", McpToolCostClass.CHEAP)


def test_retry_after_reflects_refill_rate():
    clock = _Clock()
    limiter = _limiter(clock, capacity=1, rate=2.0)  # two tokens per second => 0.5s each
    limiter.acquire("c", McpToolCostClass.CHEAP)
    with pytest.raises(McpRateLimitExceeded) as exc:
        limiter.acquire("c", McpToolCostClass.CHEAP)
    assert exc.value.retry_after == pytest.approx(0.5)


def test_callers_have_separate_buckets():
    limiter = _limiter(_Clock(), capacity=1, rate=1.0)
    limiter.acquire("a", McpToolCostClass.CHEAP)
    limiter.acquire("b", McpToolCostClass.CHEAP)  # b is untouched by a exhausting its bucket


def test_cost_classes_have_separate_buckets():
    limiter = _limiter(_Clock(), capacity=1, rate=1.0)
    limiter.acquire("a", McpToolCostClass.CHEAP)
    limiter.acquire("a", McpToolCostClass.EXPENSIVE)  # different class, own bucket


# ~~~~~~~~~~~~~ caller identity ~~~~~~~~~~~~~


def _expected_id(token: str) -> str:
    return hmac.new(rate_limit._CALLER_ID_KEY, token.encode("utf-8"), hashlib.sha256).hexdigest()[
        :32
    ]


def test_caller_identity_hashes_the_bearer_token():
    ident = caller_identity(SimpleNamespace(dial_access_token="secret-token"))
    assert ident == _expected_id("secret-token")
    assert "secret-token" not in ident  # the raw secret is never used as the key


def test_caller_identity_is_deterministic_and_token_specific():
    a1 = caller_identity(SimpleNamespace(dial_access_token="token-a"))
    a2 = caller_identity(SimpleNamespace(dial_access_token="token-a"))
    b = caller_identity(SimpleNamespace(dial_access_token="token-b"))
    assert a1 == a2  # same caller shares a bucket
    assert a1 != b  # different callers get different buckets


def test_caller_identity_falls_back_to_api_key():
    ctx = SimpleNamespace(dial_access_token=None, api_key="the-key")
    assert caller_identity(ctx) == _expected_id("the-key")


def test_caller_identity_is_anonymous_when_unidentifiable():
    # No bearer and accessing api_key raises (mirrors a context with no key): share one bucket.
    class _NoKey:
        dial_access_token = None

        @property
        def api_key(self):
            raise RuntimeError("no key")

    assert caller_identity(_NoKey()) == "anonymous"


# ~~~~~~~~~~~~~ enforce_rate_limit ~~~~~~~~~~~~~


def test_enforce_raises_actionable_tool_error_on_exhaustion(monkeypatch):
    monkeypatch.setattr(mcp_settings, "mcp_rate_limit_enabled", True)
    monkeypatch.setattr(rate_limit, "_mcp_rate_limiter", _limiter(_Clock(), capacity=1, rate=1.0))
    ctx = SimpleNamespace(dial_access_token="tok")

    enforce_rate_limit("data_query", McpToolCostClass.EXPENSIVE, ctx)  # first call allowed
    with pytest.raises(ToolError, match="Rate limit reached"):
        enforce_rate_limit("data_query", McpToolCostClass.EXPENSIVE, ctx)


def test_enforce_is_a_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(mcp_settings, "mcp_rate_limit_enabled", False)
    monkeypatch.setattr(rate_limit, "_mcp_rate_limiter", _limiter(_Clock(), capacity=1, rate=1.0))
    ctx = SimpleNamespace(dial_access_token="tok")

    for _ in range(50):  # far past the capacity of 1: the limiter must not be consulted
        enforce_rate_limit("data_query", McpToolCostClass.EXPENSIVE, ctx)
