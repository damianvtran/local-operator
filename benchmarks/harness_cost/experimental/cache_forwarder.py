"""Lane-local final-wire namespace adapter forwarding to an existing meter.

Why: metering must archive the AFTER-transformation Messages body, not a
pre-LiteLLM/bridge approximation. Point a lane-local translator's api_base at
/r/<schedule-trial-id> on this adapter; it receives the final Messages tools,
instruments the body once, and forwards to the existing shared meter's same
run route. The route's trial id is the stable namespace for every main/summary
and side call. Separate actual requests remain rows under that id in the meter.
Only hashes and metadata are logged locally, never headers or private prompts.

Constraints: development-only cachedev__ ids; immutable shared services; no
upstream credential needed; localhost-only; caller owns/reaps exact process pid.
The helper refuses unsupported first descriptors/system blocks before billing.
This adapter PRESERVES native cache markers and TTLs. No silent normalization.
It does not implement production concurrent budget admission or session mapping.

Usage: CACHE_FORWARD_LOG=<path> python cache_forwarder.py <port>
  translator api_base: adapter origin plus /r/cachedev__<trial>
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from cache_namespace import instrument

LOG = Path(os.environ.get("CACHE_FORWARD_LOG", "cache-forwarder.jsonl"))
METER = os.environ.get("CACHE_METER_BASE", "http://127.0.0.1:8787").rstrip("/")
client = httpx.AsyncClient(timeout=180)


async def forward(request: Request):
    run_id = request.path_params["run_id"]
    if not re.fullmatch(r"cachedev__[A-Za-z0-9_-]+", run_id):
        return JSONResponse({"error": "development cachedev__ trial id is needed"}, status_code=400)
    try:
        body, meta = instrument(await request.json(), run_id)
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    started = time.time()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in
               {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}}
    headers["accept-encoding"] = "identity"
    response = await client.post(f"{METER}/r/{run_id}/v1/messages", json=body, headers=headers)
    with LOG.open("a") as f:
        f.write(json.dumps({"run_id": run_id, "started": started, "ended": time.time(),
                            "status": response.status_code, **meta}) + "\n")
    # Buffering is deliberate for the bounded development control: preserve
    # SSE bytes/events, but no claim about production streaming latency.
    out_headers = {k: v for k, v in response.headers.items() if k.lower() not in
                   {"content-length", "content-encoding", "transfer-encoding", "connection"}}
    return Response(response.content, status_code=response.status_code, headers=out_headers)


app = Starlette(routes=[Route("/r/{run_id}/v1/messages", forward, methods=["POST"])])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
