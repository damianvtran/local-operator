"""Ordering fixture with a DURABLE-ONLY unread session — the divergent case.

WHAT IT IS FOR. The phone's session list used to re-derive its own order key,
and the shape that exposed it is a conversation with a transcript on disk and
NO live runtime that carries an unread completion: the terminal and the desktop
rank it Active, the phone filed it under Previous. This serves the REAL mobile
bundle and API (``MobileDaemon`` + ``build_app``, observer mode, no registrant
sockets) over synthetic sessions that include that pair (one unread, one read),
so the list can be looked at and screenshotted before and after a change to the
ordering or the pin sections. It is ``scripts/session_order_mobile.py`` plus
those two durable-only rows; keep the two in step.

HOW TO RUN IT, from a worktree with its own venv and a built bundle
(``pnpm build`` in ``local_operator/mobile/web``)::

    PYTHONPATH=. .venv/bin/python scripts/session_order_mobile_durable.py 4200

then log in at http://127.0.0.1:4200 with the synthetic password
``ordering-demo``. ``POST /fixture/tick`` republishes heartbeats and returns the
order and sections, so repeated ticks show whether the order is stable.
``scripts.probe_isolation`` re-homes ``HOME`` and the config dir before any app
import, so it never reads or writes the operator's real store. THAT IS ENFORCED,
not assumed: the module raises if anything under ``local_operator`` is already
loaded (``tests/unit/tui/test_visual_capture.py::test_probe_isolation_refuses_a_late_import``),
so moving this line below the app imports makes the script fail at once rather
than re-home a process that has already resolved the real config. Do not rely on
the gallery's import-order test for it: that one inspects scripts which call
``save_capture(app,``, and this fixture captures nothing. Restart it to reset.
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
