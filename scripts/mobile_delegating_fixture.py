"""Serve the REAL phone bundle with the delegated-work rows, for capture.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_delegating_fixture.py PORT PASSWORD
The CAPTURE script passes the password, so this file holds no literal.

ONE COLUMN, SIX ROWS, and every row is a different answer to the question the
list could not previously ask — "how much work is running under this session?":

* ``one``    — 1 running. The singular, which is where a count that pluralises
  wrongly reads as a bug before the state itself does.
* ``nine``   — 9 running. The second digit, i.e. a chip that is no longer one
  character wide.
* ``cap``    — 99 running with 12 parked, the widest chip the list can be asked
  to draw. It exists to be LOOKED AT at 360 px: the chip is `shrink-0` and the
  title truncates before it, so this is the row that says whether the count
  costs the name more than it is worth.
* ``parked`` — 0 running, 3 queued. The state that must not read as idle, and
  the one a running-count-only implementation renders as nothing at all.
* ``old``    — a record from a build that does not report counts at all. It must
  render NOTHING (no mark, no chip), never "0 subagents": the phone is not
  allowed to assert "no children" about a session it could not ask.
* ``control`` — 0 running, 0 queued, i.e. a plain resident session. The row the
  others have to be distinguishable FROM.

No runtime scanner and no registrant sockets (``dial_registrants=False``), so
this never touches the operator's live daemon or their sessions. HOME and
LOCAL_OPERATOR_CONFIG_DIR are re-homed by ``scripts.probe_isolation`` on import.

**What the summaries are.** ``_merge_summaries`` builds them, off the same
``SessionRecord`` objects the real daemon holds — so the counts in the frames
come from the record path this change moved the phone onto, not from a
hand-written JSON fixture.
"""

from __future__ import annotations

import asyncio
import sys

import uvicorn

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.session.runtime.types import SessionRecord

#: ``(session_id, pid, name, counts)``. ``counts`` is empty for the row that
#: models a daemon which predates the fields.
ROWS: list[tuple[str, int, str, dict[str, int]]] = [
    ("one", 910001, "Parent with one child", {"subagents_running": 1}),
    ("nine", 910002, "Parent with nine children", {"subagents_running": 9}),
    (
        "cap",
        910003,
        "Parent at capacity, twelve parked behind",
        {"subagents_running": 99, "subagents_queued": 12},
    ),
    ("parked", 910004, "Parent with everything parked", {"subagents_queued": 3}),
    ("old", 910005, "Session on an older daemon", {}),
    ("control", 910006, "Idle session", {"subagents_running": 0, "subagents_queued": 0}),
]


async def main() -> None:
    # The password comes from the capture script (`scripts/mobile_delegating_shot.py`
    # passes the one it types on the login form), so the two cannot disagree about it
    # and no credential-shaped literal lives in this file at all.
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4188
    password = sys.argv[2] if len(sys.argv) > 2 else "fixture-password"
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    for session_id, pid, name, counts in ROWS:
        record = SessionRecord(
            pid=pid,
            kind="tui",
            session_id=session_id,
            conversation_name=name,
            cwd="/synthetic/worktree",
            model_label="fixture/model",
            control_port=1,
            control_key="fixture",
            detached=True,
            **counts,  # type: ignore[arg-type]
        )
        if session_id == "old":
            # An ABSENT field, which is what a record written before they
            # existed carries — the same shape the daemon's own test builds by
            # deleting the attribute.
            del record.subagents_running  # type: ignore[attr-defined]
            del record.subagents_queued  # type: ignore[attr-defined]
        daemon.table.entries[pid] = SessionEntry(record)
    app = build_app(daemon)
    print(f"Fixture phone list on http://127.0.0.1:{port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
