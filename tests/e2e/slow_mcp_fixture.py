"""A stdio MCP server that is slow to come up: the cold-worker shape.

It sleeps BEFORE serving, so the client's ``initialize`` sits unread on stdin
and its reply lands about ``SLOW_MCP_DELAY_S`` after spawn. One second is
deliberately above the manager's 250 ms startup gate and inside the measured
cold-start range of a real Python stdio server (median 0.68-0.85 s, max
3.48 s; see ``DECLARED_MCP_READY_TIMEOUT_S``), so the gate always misses it.
"""

from __future__ import annotations

import os
import time
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

server = MCPServer(name="slowfix", log_level="WARNING")


@server.tool()
def search(query: str) -> str:
    """Search the fixture corpus."""
    return f"fixture result for {query}"


#: bd_tools' own cap, and the commonest fetch size in the 2026-11-20 benchmark
#: (322 of 509 fetches returned exactly this many chars).
PAGE_CHARS = 12_000


@server.tool()
def fetch(url: str) -> str:
    """Fetch a fixture page."""
    head = f"fixture page {url} BEGIN "
    tail = " END-OF-PAGE"
    return head + "p" * (PAGE_CHARS - len(head) - len(tail)) + tail


#: Opt-in OVER-CAP tool: a positive ``SLOW_MCP_BIG_SCHEMA_CHARS`` registers
#: ``bigsearch``, whose input schema carries a parameter description that long.
#: Past ``mcp.resources.MAX_INLINE_TOOL_SCHEMA_CHARS`` (16,000) an ``mcp://``
#: read cannot inline the schema and PUBLISHES the tool instead, the one read
#: route that goes through the factory's ``activate`` (PR #4349 review R1's
#: duplicate-name repro). Env-gated so every other test sees the same two tools.
BIG_SCHEMA_CHARS = int(os.environ.get("SLOW_MCP_BIG_SCHEMA_CHARS", "0") or 0)

if BIG_SCHEMA_CHARS > 0:

    @server.tool()
    def bigsearch(query: Annotated[str, Field(description="q" * BIG_SCHEMA_CHARS)]) -> str:
        """Search the fixture corpus through an over-cap input schema."""
        return f"big fixture result for {query}"


if __name__ == "__main__":
    time.sleep(float(os.environ.get("SLOW_MCP_DELAY_S", "1.0")))
    server.run()
