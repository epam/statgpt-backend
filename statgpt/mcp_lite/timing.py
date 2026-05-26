"""Per-tool timing middleware for mcp_lite.

Logs server-side wall-time for each `tools/call` to a dedicated logger.
Output is one line per call so it's grep-able for after-the-fact analysis:

    2026-05-26 12:34:56  search_indicators  duration_ms=234.5  ok=true
    2026-05-26 12:34:57  execute_sdmx_query duration_ms=1845.2 ok=true

The logger writes to `/tmp/mcp_lite_timing.log` by default; override via the
`STATGPT_MCP_LITE_TIMING_LOG` env var. Server stdout/stderr stay clean.
"""

import logging
import os
import time

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

_DEFAULT_LOG_PATH = "/tmp/mcp_lite_timing.log"
_LOG_PATH = os.environ.get("STATGPT_MCP_LITE_TIMING_LOG", _DEFAULT_LOG_PATH)
_LOGGER_NAME = "mcp_lite.timing"


def _build_logger() -> logging.Logger:
    log = logging.getLogger(_LOGGER_NAME)
    if log.handlers:
        return log
    handler = logging.FileHandler(_LOG_PATH)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


_log = _build_logger()


class TimingMiddleware(Middleware):
    """Logs server-side wall-time per `tools/call`.

    Captures the full server-side cost — from JSON-RPC dispatch through tool
    body execution and response serialization. Doesn't measure the network/agent
    round-trip; for that, compare to the subagent's wall-clock duration.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext):
        tool_name = getattr(context.message, "name", "<unknown>")
        t0 = time.monotonic()
        ok = True
        try:
            return await call_next(context)
        except Exception:
            ok = False
            raise
        finally:
            dur_ms = (time.monotonic() - t0) * 1000
            _log.info(f"{tool_name:24s}  duration_ms={dur_ms:7.1f}  ok={str(ok).lower()}")
