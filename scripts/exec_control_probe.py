"""Attach to a `lop exec --control` run and drive one control op.

A hand-driven supervisor stand-in used to exercise the exec control surface
end to end: it resolves the run's record by session id, reads the control key
out of the 0600 file (the whole authorization model — see
session/runtime/registry.py), dials the loopback socket, and speaks one op.

Not a test fixture and not part of the product: this is the operator-side tool
for the evidence a change to the control surface has to produce, kept next to
the other scripts/ helpers so the next person verifying this path does not
rewrite it from the protocol docs.

    python scripts/exec_control_probe.py <session-id> steer "text"
    python scripts/exec_control_probe.py <session-id> cancel [graceful|immediate]
    python scripts/exec_control_probe.py <session-id> ping
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_operator.paths import config_dir  # noqa: E402
from local_operator.session.runtime import registry  # noqa: E402


def find(session_id: str):
    for record, state in registry.scan(config_dir()):
        if record.session_id == session_id:
            return record, state
    raise SystemExit(f"no record for session {session_id!r}")


async def main() -> int:
    session_id, op = sys.argv[1], sys.argv[2]
    arg = sys.argv[3] if len(sys.argv) > 3 else ""
    record, state = find(session_id)
    print(
        f"[probe] record: pid={record.pid} kind={record.kind} state={state} "
        f"port={record.control_port}",
        flush=True,
    )

    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
    # Dial daemon-class (the default when `client` is omitted) like peer_client:
    # a fire-and-forget supervisor wants no attach accounting.
    writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
    await writer.drain()

    frame: dict[str, Any] = {"op": op, "req": "probe-1"}
    if op == "steer":
        frame["text"] = arg
    elif op == "cancel" and arg:
        frame["mode"] = arg
    writer.write(json.dumps(frame).encode() + b"\n")
    await writer.drain()
    print(f"[probe] sent: {json.dumps(frame)}", flush=True)

    # The first frame back is the unsolicited welcome projection; read until the
    # ack/error carrying our req id.
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=20.0)
        if not line:
            print("[probe] connection closed before ack", flush=True)
            return 1
        try:
            got = json.loads(line)
        except ValueError:
            continue
        if got.get("op") in ("ack", "error") and got.get("req") == "probe-1":
            print(f"[probe] reply: {json.dumps(got)}", flush=True)
            writer.close()
            return 0 if got["op"] == "ack" else 1
        print(f"[probe] ...{got.get('op')}", flush=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
