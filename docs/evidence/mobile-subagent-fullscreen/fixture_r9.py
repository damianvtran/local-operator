"""Round-9 lifecycle validation harness.

Wraps the committed review fixture's daemon + real production bundle and adds
the deterministic edges round 9 hardened, driven by on-disk sentinels so a
browser walk can flip behaviour between page loads without restarting:

  * ``/tmp/lop-r9-ambiguous-<PORT>`` present → the parent command endpoint
    answers the configured ambiguous status (502/504/408) instead of 200, to
    prove the persisted envelope is KEPT for a same-UUID retry.
  * A second session route ``fixture-root-b`` with its own SSE + command, so a
    conversation switch away and back can be shown to PRESERVE session A's
    pending envelope (U1).
  * ``/logout`` serves the daemon's real login HTML (with its inline
    private-storage clear script) so the WebKit-safe logout cleanup runs in the
    browser exactly as in production (U2).

Run:  LO_MOBILE_FIXTURE_PORT=4188 python fixture_r9.py
Login password: fixture-review
"""

import json
import os
import uuid
from pathlib import Path

PORT = int(os.environ.get("LO_MOBILE_FIXTURE_PORT", "4188"))

# Import the committed fixture module's already-built daemon/app and data.
import fixture as base  # noqa: E402

AMBIGUOUS_SENTINEL = Path(f"/tmp/lop-r9-ambiguous-{PORT}")
COMMAND_LOG_A = Path(f"/tmp/lop-r9-command-a-{PORT}.json")
COMMAND_LOG_B = Path(f"/tmp/lop-r9-command-b-{PORT}.json")
SESSION_A = base.SESSION  # "fixture-root"
SESSION_B = "fixture-root-b"

# A second routable session so cross-conversation preservation is testable.
proj_b = base.SessionProjection(
    session_id=SESSION_B,
    pid=4243,
    conversation_name="Second conversation",
    streaming=True,
    version=7,
    transcript=[base.entry("b-1", "user", "Second conversation for switch test.")],
)
base.daemon.capture_subagent_details(proj_b)
base.daemon.session_projections[SESSION_B] = proj_b
record_b = base.SessionRecord(
    pid=4243,
    kind="tui",
    session_id=SESSION_B,
    conversation_name=proj_b.conversation_name,
    cwd=str(Path.cwd()),
    model_label="fixture",
    control_port=2,
    control_key="fixture-b",
)
entry_b = base.SessionEntry(record_b)
entry_b.projection = proj_b
base.daemon.table.entries[record_b.pid] = entry_b


def _log_for(session_id: str) -> Path:
    return COMMAND_LOG_A if session_id == SESSION_A else COMMAND_LOG_B


async def _read_body(receive) -> bytes:
    chunks = []
    while True:
        message = await receive()
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


async def _json(send, status: int, payload: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": payload})


class RoundNine:
    """ASGI wrapper adding round-9 lifecycle edges over the real daemon app."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]

        # Both sessions' command endpoints: honour the ambiguous-status sentinel
        # (post-admission acknowledgement loss), else record a durable 200 ACK.
        for session_id in (SESSION_A, SESSION_B):
            if path == f"/api/sessions/{session_id}/command":
                await self._command(scope, receive, send, session_id)
                return

        # Second session SSE seed (session A's SSE is handled by the base app).
        # A DOM-level diagnostic: prints the engine UA and every private storage
        # key still present, so the browser walk can PROVE (not infer) that
        # logout cleared the scoped storage on the real WebKit engine.
        if path == "/r9-diag":
            html = (
                b"<!doctype html><meta charset=utf-8>"
                b"<body style='font:16px monospace;color:#e9e5db;background:#14110c'>"
                b"<pre id=out></pre><script>"
                b"var keys=[];for(var i=0;i<localStorage.length;i++){var k=localStorage.key(i);"
                b"if(k&&(k.indexOf('lo-mobile-command:')===0||k.indexOf('lo-mobile-draft:')===0))"
                b"keys.push(k);}"
                b"document.getElementById('out').textContent="
                b"'engine='+navigator.userAgent+'\\nprivateKeys='+JSON.stringify(keys);"
                b"</script></body>"
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/html; charset=utf-8")],
                }
            )
            await send({"type": "http.response.body", "body": html})
            return

        if path == f"/api/sessions/{SESSION_B}/events":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/event-stream"),
                        (b"cache-control", b"no-cache"),
                    ],
                }
            )
            body = f"event: projection\ndata: {json.dumps(proj_b.to_json())}\n\n".encode()
            await send({"type": "http.response.body", "body": body, "more_body": True})
            # Hold the stream open briefly; the base disconnect sentinel ends it.
            import asyncio

            while not base.DISCONNECT_SENTINEL.exists():
                await asyncio.sleep(0.1)
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        await self.app(scope, receive, send)

    async def _command(self, scope, receive, send, session_id: str) -> None:
        headers = dict(scope.get("headers", []))
        cookie = headers.get(b"cookie", b"").decode("utf-8", "replace")
        if "lop_mobile=" not in cookie:
            await _json(send, 401, b'{"error":"unauthorized"}')
            return
        body_raw = await _read_body(receive)
        try:
            body = json.loads(body_raw)
            command_id = body["command_id"]
            uuid.UUID(command_id)
            text = body["text"]
            if (
                body.get("op") not in ("steer", "prompt")
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise ValueError
        except (ValueError, TypeError, KeyError):
            await _json(send, 422, b'{"error":"Enter an instruction before sending."}')
            return

        # Ambiguous acknowledgement loss: the daemon may have admitted the UUID
        # but the reply was lost. The envelope must be KEPT for a same-UUID retry.
        if AMBIGUOUS_SENTINEL.exists():
            status = int(AMBIGUOUS_SENTINEL.read_text().strip() or "504")
            # Record that the command WAS admitted, so a later retry with the
            # same UUID coalesces to one delivery — the duplicate-prevention proof.
            log = _log_for(session_id)
            record = {"command_id": command_id, "text": text.strip(), "deliveries": 1}
            if log.exists() and json.loads(log.read_text()).get("command_id") == command_id:
                record = json.loads(log.read_text())
            log.write_text(json.dumps(record, indent=2))
            print(f"r9 ambiguous {status} admitted={command_id} on {session_id}", flush=True)
            await _json(send, status, b'{"error":"session did not answer"}')
            return

        log = _log_for(session_id)
        accepted = {"command_id": command_id, "text": text.strip(), "deliveries": 1}
        if log.exists():
            previous = json.loads(log.read_text())
            if previous.get("command_id") == command_id:
                accepted = previous  # idempotent replay: still one delivery
        log.write_text(json.dumps(accepted, indent=2))
        print(
            f"r9 ack {accepted['command_id']} deliveries={accepted['deliveries']} on {session_id}",
            flush=True,
        )
        await _json(send, 200, b'{"ok":true,"detail":"steering queued"}')


app = RoundNine(base.base_app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
