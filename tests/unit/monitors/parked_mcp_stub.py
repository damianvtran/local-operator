"""A real MCP peer for the parked-probe tests and evidence rig.

Spawned per probe run (stdio) or once (``--http PORT``), it serves:

- ``read_thread`` — ``readOnlyHint: true`` unless ``<STUB_DIR>/no_hint`` exists
  when the server STARTS (each probe run spawns a fresh stdio server, so the
  flag models a server that dropped the hint between runs). Returns the text of
  ``<STUB_DIR>/thread.txt`` and appends one line to ``<STUB_DIR>/calls`` per
  invocation, so a test can prove whether the tool ran.
- ``post`` — no hint at all (a write tool the gate must never call).

HTTP mode answers 401 with ``WWW-Authenticate: Bearer`` unless the request
carries ``Authorization: Bearer <STUB_TOKEN>``, and always 401 while
``<STUB_DIR>/deny`` exists — the re-authentication case.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

STUB_DIR = Path(os.environ["STUB_DIR"])

server = MCPServer("parked-probe-stub")

_HINT = None if (STUB_DIR / "no_hint").exists() else ToolAnnotations(read_only_hint=True)


@server.tool(annotations=_HINT)
def read_thread(thread: str) -> str:
    """Return the watched thread's current text."""
    with (STUB_DIR / "calls").open("a", encoding="utf-8") as handle:
        handle.write(f"{thread}\n")
    return (STUB_DIR / "thread.txt").read_text(encoding="utf-8")


@server.tool()
def post(text: str) -> str:
    """A write the probe must never call."""
    with (STUB_DIR / "posts").open("a", encoding="utf-8") as handle:
        handle.write(f"{text}\n")
    return "posted"


def _serve_http(port: int) -> None:
    import uvicorn
    from starlette.responses import PlainTextResponse

    token = os.environ.get("STUB_TOKEN", "")
    inner = server.streamable_http_app()

    async def app(scope, receive, send):  # noqa: ANN001, ANN202
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            denied = (STUB_DIR / "deny").exists()
            if denied or headers.get("authorization") != f"Bearer {token}":
                with (STUB_DIR / "unauthorized").open("a", encoding="utf-8") as handle:
                    handle.write("401\n")
                response = PlainTextResponse(
                    "unauthorized", status_code=401, headers={"WWW-Authenticate": "Bearer"}
                )
                await response(scope, receive, send)
                return
        await inner(scope, receive, send)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--http":
        _serve_http(int(sys.argv[2]))
    else:
        server.run(transport="stdio")
