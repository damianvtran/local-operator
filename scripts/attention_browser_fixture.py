"""Private browser receipt fixture: real desktop route/store, no session runtime.

Run with env -i HOME=<scratch> LOCAL_OPERATOR_CONFIG_DIR=<scratch/config>
LOCAL_OPERATOR_DESKTOP_TOKEN=<synthetic> .../python scripts/attention_browser_fixture.py.
The paired UI scripts/attention-fixture uses the supported development proxy;
this fixture never acknowledges directly. Only the production /seen route writes
receipts. This is browser/component evidence, NOT Electron native-focus proof.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402

from local_operator.server.routes import desktop_sessions  # noqa: E402
from local_operator.server.utils.desktop_sessions import DesktopSessions  # noqa: E402
from local_operator.session.attention import AttentionStore  # noqa: E402

root = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]).resolve()
if root == Path.home() / ".local-operator" or not str(root).startswith(("/tmp/", "/private/tmp/")):
    raise SystemExit("Use an explicit scratch config under /tmp and an isolated HOME")
root.mkdir(parents=True, exist_ok=True)
pool = DesktopSessions(root)
sid = asyncio.run(pool.create(str(root)))
store = AttentionStore(root / "attention.db")
anchor = "fixture-final-answer"
store.publish(f"session/{sid}", str(uuid.uuid4()), anchor, "complete")
app = FastAPI()
app.state.desktop_sessions = pool
app.include_router(desktop_sessions.router)


@app.middleware("http")
async def audit_receipt(request: Request, call_next):
    """Record request token, actual response and durable delta, not a 200 count."""
    if not request.url.path.endswith("/seen"):
        return await call_next(request)
    body = await request.json()
    before = store.state(f"session/{sid}")
    response = await call_next(request)
    chunks = [chunk async for chunk in response.body_iterator]
    from starlette.responses import Response

    payload = b"".join(chunks)
    record = {
        "at": time.time(),
        "source": "supported-browser-development-proxy",
        "request": body,
        "status": response.status_code,
        "response": json.loads(payload),
        "before": before,
        "after": store.state(f"session/{sid}"),
    }
    with (root / "receipt-audit.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")
    return Response(payload, status_code=response.status_code, headers=dict(response.headers))


@app.get("/fixture/state")
async def fixture_state():
    return {"session_id": sid, "attention": {"supported": True, **store.state(f"session/{sid}")}}


@app.post("/fixture/publish")
async def publish():
    """Generate a genuine new outcome for another controlled visibility case."""
    store.publish(f"session/{sid}", str(uuid.uuid4()), anchor, "complete")
    return await fixture_state()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("FIXTURE_PORT", "18764")))
