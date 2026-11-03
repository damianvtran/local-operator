"""MCP replay stub, public loadout: serve open-source MCP tool schemas, execute nothing.

WHY: the benchmark's "full loadout" arm gives every harness the same large MCP
surface, because tool-schema volume is the main cost difference between
harnesses at scale. v1 replayed a private cache that cannot be published;
this public variant serves `public_mcp_schemas.json`, the verbatim
`tools/list` responses of pinned open-source servers, captured by
`capture_mcp_schemas.py`. Anyone can rebuild that file from the pins and get
the same bytes (`capture_mcp_schemas.py --check`).

Behaviour is identical to v1's mcp_replay.py:
  * `initialize` / `tools/list` answered from the snapshot, tool order kept;
  * `tools/call` answered with an MCP error result (isError) and logged as
    tool name + run tag only, never arguments, so no real service is reached
    and the report can say which harnesses reached for MCP tools;
  * Streamable HTTP with plain JSON responses (no SSE), stateless.

Endpoints: POST /<run_id>/<server>/mcp     (run_id tags call attempts)
           GET  /health                    ({server: tool_count})
Config (env): BENCH_MCP_SCHEMAS (snapshot path, default next to this file),
BENCH_MCP_SERVERS (comma list, default every server in the snapshot),
BENCH_MCP_PORT (default 8790), BENCH_MCP_LOG (call log, jsonl).
Dependencies: starlette + uvicorn, nothing else.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

SNAPSHOT = Path(os.environ.get("BENCH_MCP_SCHEMAS", Path(__file__).resolve().parent / "public_mcp_schemas.json"))
LOG = Path(os.environ.get("BENCH_MCP_LOG", "mcp_calls.jsonl"))


def load_schemas() -> dict[str, list[dict]]:
    """Return {server: tools} for BENCH_MCP_SERVERS, refusing to start on a gap.

    Fails loudly rather than serving a partial surface: a missing server would
    silently shrink the loadout and make the run incomparable.
    """
    snap = json.loads(SNAPSHOT.read_text())["servers"]
    wanted = [s for s in os.environ.get("BENCH_MCP_SERVERS", ",".join(snap)).split(",") if s]
    missing = sorted(set(wanted) - set(snap))
    if missing:
        raise SystemExit(f"no captured schemas for {missing} in {SNAPSHOT.name}")
    return {s: snap[s]["tools"] for s in wanted}


SCHEMAS = load_schemas()


def _ok(rid, result):
    return {"jsonrpc": "2.0", "id": rid, "result": result}


async def mcp(request: Request) -> Response:
    server = request.path_params["server"]
    run_id = request.path_params["run_id"]
    if server not in SCHEMAS:
        return JSONResponse({"error": "unknown server"}, status_code=404)
    if request.method == "GET":
        return Response(status_code=405)  # no server-initiated stream
    if request.method == "DELETE":
        return Response(status_code=200)
    payload = json.loads(await request.body() or b"null")
    batch = payload if isinstance(payload, list) else [payload]
    replies = []
    for msg in batch:
        method, rid = msg.get("method"), msg.get("id")
        if rid is None:  # notification (e.g. notifications/initialized)
            continue
        if method == "initialize":
            ver = (msg.get("params") or {}).get("protocolVersion", "2025-06-18")
            replies.append(_ok(rid, {
                "protocolVersion": ver,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": server, "version": "replay"},
            }))
        elif method == "tools/list":
            replies.append(_ok(rid, {"tools": SCHEMAS[server]}))
        elif method == "tools/call":
            name = (msg.get("params") or {}).get("name")
            with LOG.open("a") as fh:
                fh.write(json.dumps({"ts": time.time(), "run_id": run_id, "server": server, "tool": name}) + "\n")
            replies.append(_ok(rid, {
                "isError": True,
                "content": [{"type": "text", "text": f"{server} is unavailable in this environment (offline benchmark); complete the task without it."}],
            }))
        elif method == "ping":
            replies.append(_ok(rid, {}))
        elif method in ("resources/list", "prompts/list", "resources/templates/list"):
            key = {"resources/list": "resources", "prompts/list": "prompts", "resources/templates/list": "resourceTemplates"}[method]
            replies.append(_ok(rid, {key: []}))
        else:
            replies.append({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}})
    if not replies:
        return Response(status_code=202)
    body = replies if isinstance(payload, list) else replies[0]
    return JSONResponse(body)


async def health(_: Request) -> Response:
    return JSONResponse({s: len(t) for s, t in SCHEMAS.items()})


app = Starlette(routes=[
    Route("/health", health),
    Route("/{run_id}/{server}/mcp", mcp, methods=["GET", "POST", "DELETE"]),
])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("BENCH_MCP_PORT", "8790")), log_level="warning")
