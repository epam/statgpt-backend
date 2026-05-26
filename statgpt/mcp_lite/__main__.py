"""Standalone runner for MCP-Lite.

Boots ONLY the MCP-Lite server on port 5050, mounted at /mcp-lite/.
Useful for `tools/list` smoke tests with MCPJam / mcp-inspector;
`tools/call` will fail without the full DIAL stack (no channel facade).

Usage:
    PYDANTIC_V2=True python -m statgpt.mcp_lite

For full functionality run the DIAL app via `make statgpt_app` instead.
"""

import logging
import os

import dotenv
import uvicorn

dotenv.load_dotenv(os.path.join(os.getcwd(), ".env"))

from statgpt.mcp_lite.app import mcp

logging.basicConfig(level=logging.INFO)

app = mcp.http_app(path="/mcp-lite/", transport="streamable-http", stateless_http=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050, log_config=None)
