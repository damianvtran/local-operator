"""Serve the real mobile bundle/API with synthetic sessions for ordering review.

Run with PYTHONPATH=. .venv/bin/python scripts/session_order_mobile.py.
Login at http://127.0.0.1:4198 with the synthetic password `ordering-demo`.
POST /fixture/tick advances activity. Restart the fixture to reset completions.
No runtime scanner or registrant sockets are started.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

import uvicorn
from starlette.responses import JSONResponse
from starlette.routing import Route

import scripts.probe_isolation  # noqa: F401
from local_operator.harness.types import Message
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import SessionProjection, SessionRecord
from local_operator.session.attention import AttentionStore
from local_operator.session.transcript import Transcript

DATA = [
    ("done-new", "Completed newest", False, "complete"),
    ("done-old", "Completed older", False, "complete"),
    ("error", "Failed review", False, "error"),
    ("interrupt", "Interrupted research", False, "interrupted"),
    ("busy-a", "Working Alpha", True, ""),
    ("busy-b", "Working Beta", True, ""),
    ("busy-c", "Working Gamma", True, ""),
]


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4198
    daemon = MobileDaemon(port=port, password="ordering-demo", dial_registrants=False)
    cfg = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    for index, (sid, title, busy, kind) in enumerate(DATA):
        directory = cfg / "sessions" / sid
        transcript = Transcript(directory)
        await transcript.append_message(Message.user(title))
        await transcript.append_message(Message.assistant("Synthetic ordering evidence only."))
        # Explicit immutable fixture dates, shared by before and after trees.
        (directory / "created_at.json").write_text(str(1000 - index))
        os.utime(transcript.path, (1000 - index, 1000 - index))
        record = SessionRecord(
            pid=900000 + index,
            kind="tui",
            session_id=sid,
            conversation_name=title,
            cwd="/synthetic",
            model_label="Demo",
            control_port=1,
            control_key="synthetic",
        )
        entry = SessionEntry(record)
        entry.projection = SessionProjection(
            session_id=sid, pid=record.pid, kind="tui", conversation_name=title, streaming=busy
        )
        daemon.table.entries[record.pid] = entry
        if kind:
            AttentionStore().publish(f"session/{sid}", str(uuid.uuid4()), "result", kind)
    tick = 0

    async def advance(_request):
        nonlocal tick
        sid = ("busy-c", "busy-a", "busy-b")[tick % 3]
        tick += 1
        transcript = cfg / "sessions" / sid / "transcript.jsonl"
        os.utime(transcript, (time.time(), time.time()))
        for entry in daemon.table.entries.values():
            entry.record.heartbeat_at = time.time()
        daemon.table.invalidate_summaries_cache()
        daemon.table.notify_list_changed()
        rows = await daemon.table.summaries()
        return JSONResponse({"tick": tick, "order": [r["session_id"] for r in rows]})

    app = build_app(daemon)
    app.routes.insert(0, Route("/fixture/tick", advance, methods=["POST"]))
    print(f"Synthetic mobile: http://127.0.0.1:{port} password ordering-demo", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
