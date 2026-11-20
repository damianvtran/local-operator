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

from mcp.server.mcpserver import MCPServer

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


if __name__ == "__main__":
    time.sleep(float(os.environ.get("SLOW_MCP_DELAY_S", "1.0")))
    server.run()
