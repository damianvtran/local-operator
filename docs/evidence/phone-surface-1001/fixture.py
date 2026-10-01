"""Evidence fixture: two synthetic sessions (live + ended), served from TREE's bundle.
Usage: python fixture.py <tree> <port> <password>   (tree = a dir holding local_operator/)"""
import asyncio, os, sys
tree, port, password = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, tree)
sys.path.insert(0, os.path.join(tree))
import scripts.probe_isolation  # noqa  (re-homes HOME/config; tree has scripts/)
import uvicorn
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import SessionProjection, SessionRecord, TranscriptEntry
import local_operator
assert local_operator.__file__.startswith(tree), local_operator.__file__
async def main():
    # The ended session's DURABLE half: the daemon only lets a send wake a conversation whose directory
    # carries a transcript. The wake itself is stubbed to FAIL (a real one would spawn a runtime): the
    # failed-send copy is exactly what issue #1875 is about.
    import json, time
    from local_operator.harness.types import Message
    from local_operator.paths import config_dir
    import local_operator.mobile.attach_client as dmod
    d = config_dir() / "sessions" / "ev-ended"; d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.jsonl").write_text(json.dumps({"id": "f1", "ts": time.time(), "type": "message", "payload": Message.user("Summarise the quarterly numbers.").model_dump(exclude_defaults=True)}) + "\n")
    async def failing_wake(*a, **k): raise ConnectionError("no host")
    dmod.continue_command = failing_wake
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    for i, (sid, name, ended) in enumerate([("ev-live", "Live session", False), ("ev-ended", "Ended session", True)]):
        proj = SessionProjection(session_id=sid, pid=910000 + i, kind="tui", conversation_name=name, streaming=False, ended=ended,
            transcript=[TranscriptEntry(id=f"{sid}-1", kind="user", text="Summarise the quarterly numbers."),
                        TranscriptEntry(id=f"{sid}-2", kind="assistant", text="Revenue is up 12% on the quarter; costs are flat. The one line worth a second look is the APAC services margin.")], version=3)
        rec = SessionRecord(pid=proj.pid, kind="tui", session_id=sid, conversation_name=name, cwd="/synthetic", model_label="fixture", control_port=1, control_key="fixture")
        entry = SessionEntry(rec); entry.projection = proj; entry.ended = ended
        daemon.session_projections[sid] = proj
        daemon.table.entries[rec.pid] = entry
    print("ready", flush=True)
    await uvicorn.Server(uvicorn.Config(build_app(daemon), host="127.0.0.1", port=port, log_level="warning")).serve()
asyncio.run(main())
