"""Ordering fixture with a DURABLE-ONLY unread session — the divergent case.

Same shape as ``scripts/session_order_mobile.py`` plus two conversations that
have a transcript on disk and NO live runtime: one carrying an unread completion
and one read. That pair is the case the phone's re-derived key got wrong (the
unread one is ACTIVE on the terminal and the desktop, Previous on the phone),
and it is the case the shared-key fix makes agree.

Run: PYTHONPATH=. .venv/bin/python THIS PORT
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

import scripts.probe_isolation  # noqa: F401  -- FIRST, re-homes HOME/config
from local_operator.harness.types import Message
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import SessionProjection, SessionRecord
from local_operator.session.attention import AttentionStore
from local_operator.session.transcript import Transcript

# (id, title, live-busy, completion-kind, has-live-runtime)
DATA = [
    ("done-new", "Completed newest", False, "complete", True),
    ("done-old", "Completed older", False, "complete", True),
    ("cold-unread", "Finished while away", False, "complete", False),
    ("cold-read", "Read while away", False, "", False),
    ("busy-a", "Working Alpha", True, "", True),
]


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4200
    daemon = MobileDaemon(port=port, password="ordering-demo", dial_registrants=False)
    cfg = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    for index, (sid, title, busy, kind, live) in enumerate(DATA):
        directory = cfg / "sessions" / sid
        transcript = Transcript(directory)
        await transcript.append_message(Message.user(title))
        await transcript.append_message(Message.assistant("Synthetic ordering evidence only."))
        (directory / "created_at.json").write_text(str(1000 - index))
        os.utime(transcript.path, (1000 - index, 1000 - index))
        if not live:
            # DURABLE ONLY: no record, no entry. The phone's own listing reaches
            # it through the durable half, and its unread receipt is the fact the
            # two surfaces used to disagree about.
            if kind:
                AttentionStore().publish(f"session/{sid}", str(uuid.uuid4()), "result", kind)
            continue
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

    async def advance(_request):
        for entry in daemon.table.entries.values():
            entry.record.heartbeat_at = time.time()
        daemon.table.invalidate_summaries_cache()
        daemon.table.notify_list_changed()
        rows = await daemon.table.summaries()
        return JSONResponse(
            {
                "order": [r["session_id"] for r in rows],
                "section": {r["session_id"]: r["section"] for r in rows},
            }
        )

    app = build_app(daemon)
    app.routes.insert(0, Route("/fixture/tick", advance, methods=["POST"]))
    print(f"Synthetic ordering: http://127.0.0.1:{port} password ordering-demo", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
