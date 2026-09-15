"""Per-caller rate limiting for model-facing MCP tools (#600).

Marketplace guidance requires rate limits on tools that are expensive to serve or externally
reachable. StatGPT data tools can trigger large SDMX queries and LLM-backed processing, so an
unconstrained caller (or a model in a retry loop) can drive cost and degrade service for everyone.

Enforcement lives in the StatGPT MCP layer (not DIAL Core) so it can key off the validated bearer
token and skip app-only tools. Counters are held in process: there is no shared store in the
deployment, so limits are enforced per replica and are approximate under horizontal scaling. That
is enough to stop a single caller from hammering one instance; tighten to a shared store if exact
global limits are ever needed.
"""

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from fastmcp.exceptions import ToolError

from statgpt.app.settings.mcp import McpSettings, mcp_settings
from statgpt.common.auth.auth_context import AuthContext

_log = logging.getLogger(__name__)

# Prune idle buckets once the table grows past this, so a long-lived process that has served many
# distinct callers does not retain a bucket per caller forever.
_PRUNE_THRESHOLD = 10_000


class McpToolCostClass(str, Enum):
    """How expensive a tool is to serve, which selects the rate-limit allowance applied to it."""

    CHEAP = "cheap"  # local lookups, e.g. glossary
    MODERATE = "moderate"  # upstream metadata reads, e.g. dataset listing/structure
    EXPENSIVE = "expensive"  # large SDMX queries and/or LLM-backed processing, e.g. data query


class McpRateLimitExceeded(Exception):
    """Raised by the limiter when a caller has exhausted its allowance for a cost class."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded; retry in {retry_after:.1f}s")


@dataclass
class BucketConfig:
    """A token bucket's shape: `capacity` back-to-back calls, refilled at `refill_rate` per second."""

    capacity: int
    refill_rate: float


@dataclass
class _TokenBucket:
    tokens: float
    updated_at: float


class McpRateLimiter:
    """In-process token-bucket limiter keyed by (caller, cost class).

    A token bucket gives both a burst allowance (the capacity) and a sustained rate (the refill),
    which matches the "short burst plus a longer sustained window" the guidance asks for. All
    arithmetic runs without an await, so it is atomic on the event loop and needs no lock.
    """

    def __init__(
        self,
        configs: dict[McpToolCostClass, BucketConfig],
        time_fn: Callable[[], float] = time.monotonic,
    ):
        self._configs = configs
        self._time_fn = time_fn
        self._buckets: dict[tuple[str, McpToolCostClass], _TokenBucket] = {}

    def acquire(self, caller_id: str, cost_class: McpToolCostClass) -> None:
        """Consume one token for `caller_id` in `cost_class`, or raise `McpRateLimitExceeded`."""
        config = self._configs[cost_class]
        now = self._time_fn()
        key = (caller_id, cost_class)

        bucket = self._buckets.get(key)
        if bucket is None:
            # A fresh caller starts with a full bucket: the first calls are the burst allowance.
            bucket = _TokenBucket(tokens=float(config.capacity), updated_at=now)
            self._buckets[key] = bucket
            if len(self._buckets) > _PRUNE_THRESHOLD:
                self._prune(now)
        else:
            elapsed = now - bucket.updated_at
            bucket.tokens = min(config.capacity, bucket.tokens + elapsed * config.refill_rate)
            bucket.updated_at = now

        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return

        retry_after = (1 - bucket.tokens) / config.refill_rate
        raise McpRateLimitExceeded(retry_after)

    def _prune(self, now: float) -> None:
        """Drop buckets that have fully refilled: they carry no debt, so forgetting them only
        resets a caller to a full bucket, which it would have had anyway."""
        stale = [
            key
            for key, bucket in self._buckets.items()
            if bucket.tokens >= self._configs[key[1]].capacity
        ]
        for key in stale:
            del self._buckets[key]


def _configs_from_settings(settings: McpSettings) -> dict[McpToolCostClass, BucketConfig]:
    window = settings.mcp_rate_limit_window_seconds
    return {
        McpToolCostClass.CHEAP: BucketConfig(
            capacity=settings.mcp_rate_limit_cheap_burst,
            refill_rate=settings.mcp_rate_limit_cheap_per_window / window,
        ),
        McpToolCostClass.MODERATE: BucketConfig(
            capacity=settings.mcp_rate_limit_moderate_burst,
            refill_rate=settings.mcp_rate_limit_moderate_per_window / window,
        ),
        McpToolCostClass.EXPENSIVE: BucketConfig(
            capacity=settings.mcp_rate_limit_expensive_burst,
            refill_rate=settings.mcp_rate_limit_expensive_per_window / window,
        ),
    }


def caller_identity(auth_context: AuthContext) -> str:
    """A stable, non-reversible id for the caller, used only as a rate-limit key.

    Derived from the bearer token (or the DIAL API key when there is no bearer). Callers we cannot
    identify share a single "anonymous" bucket, which throttles them together rather than exempting
    them.
    """
    token = auth_context.dial_access_token
    if not token:
        try:
            token = auth_context.api_key
        except Exception:
            token = None
    if not token:
        return "anonymous"
    # Derive the id with the builtin `hash` (a per-process-seeded SipHash), not a crypto hash: it
    # runs on every call ahead of the limit check, so it must stay cheap (an expensive KDF here
    # would itself be a CPU-exhaustion vector), the raw secret is never held as a key or logged, and
    # a high-entropy token cannot be recovered from the digest. A crypto hash is avoided on purpose:
    # CodeQL rejects any fast crypto hash of a credential as weak, accepting only an expensive KDF.
    return f"{hash(token) & 0xFFFFFFFFFFFFFFFF:016x}"


_mcp_rate_limiter = McpRateLimiter(_configs_from_settings(mcp_settings))


def enforce_rate_limit(
    tool_name: str, cost_class: McpToolCostClass, auth_context: AuthContext
) -> None:
    """Charge one call against the caller's allowance for `cost_class`.

    Does nothing when rate limiting is disabled. On exhaustion, logs the hit (keyed by the hashed
    caller, never the raw token) and raises a `ToolError` the model can act on, rather than a bare
    429, telling it how long to wait.
    """
    if not mcp_settings.mcp_rate_limit_enabled:
        return

    caller = caller_identity(auth_context)
    try:
        _mcp_rate_limiter.acquire(caller, cost_class)
    except McpRateLimitExceeded as e:
        retry_after = max(1, math.ceil(e.retry_after))
        _log.warning(
            "MCP rate limit hit: tool=%s cost_class=%s caller=%s retry_after=%ss",
            tool_name,
            cost_class.value,
            caller,
            retry_after,
        )
        raise ToolError(
            f"Rate limit reached for '{tool_name}'. Retry in {retry_after} seconds, "
            "or reduce how often you call this tool."
        ) from e
