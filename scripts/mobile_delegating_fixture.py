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
import uuid

import uvicorn

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import SessionProjection
from local_operator.session.runtime.types import SessionRecord

#: ``(session_id, pid, name, counts, flags)``. ``counts`` is empty for the row
#: that models a daemon which predates the fields; ``flags`` carries the two
#: states the dot pair has to be told apart FROM (UX round 1, D1).
ROWS: list[tuple[str, int, str, dict[str, int], dict[str, bool]]] = [
    ("one", 910001, "Parent with one child", {"subagents_running": 1}, {}),
    ("nine", 910002, "Parent with nine children", {"subagents_running": 9}, {}),
    (
        "cap",
        910003,
        "Parent at capacity, twelve parked behind",
        {"subagents_running": 99, "subagents_queued": 12},
        {},
    ),
    ("parked", 910004, "Parent with everything parked", {"subagents_queued": 3}, {}),
    ("old", 910005, "Session on an older daemon", {}, {}),
    ("control", 910006, "Idle session", {"subagents_running": 0, "subagents_queued": 0}, {}),
    # THE TWO NEIGHBOURS IN THE SLOT. The design record justifies the PAIR of
    # dots as "a SHAPE distinct from the single unread dot one rung above it, in
    # the same ink", and that comparison is only judgeable if one frame carries
    # all three: the single dot, the spinner that also lives in this slot, and
    # the pair. A streaming row needs a real projection (the summary's
    # ``streaming`` comes from it), and an unread row needs a durable directory
    # plus a real receipt — the merge only asks the attention store about
    # identities the durable scan produced — so both are built below rather than
    # faked in the payload.
    ("unread", 910007, "Parent with an unread outcome", {"subagents_running": 1}, {"unseen": True}),
    (
        "streaming",
        910008,
        "Parent working in its own turn",
        {"subagents_running": 2},
        {"streaming": True},
    ),
]


def _seed_unread(session_id: str) -> None:
    """A session the store LISTS and the attention store has an unread receipt for.

    TWO HALVES, and both are load-bearing. The merge asks the attention store only
    about identities the DURABLE scan produced (``identities = {f"session/{sid}"
    for sid in rows}``), so a receipt with no session directory is invisible; and
    the receipt is what makes the row unread at all. Neither is faked into the
    payload: the directory is the shape the catalogue lists (a ``transcript.jsonl``
    is what makes a directory a conversation), and the receipt goes through the
    real ``AttentionStore`` writer the runtime itself uses.
    """
    from local_operator.paths import config_dir
    from local_operator.session.attention import AttentionStore

    directory = config_dir() / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    AttentionStore().publish(f"session/{session_id}", str(uuid.uuid4()), "a1", "complete")


async def main() -> None:
    # The password comes from the capture script (`scripts.mobile_delegating_shot.py`
    # passes the one it types on the login form), so the two cannot disagree about it
    # and no credential-shaped literal lives in this file at all.
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4188
    password = sys.argv[2] if len(sys.argv) > 2 else "fixture-password"
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    for session_id, pid, name, counts, flags in ROWS:
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
        entry = SessionEntry(record)
        if flags.get("streaming"):
            # The summary's ``streaming`` is read off the projection, so this is
            # the only way the fixture can produce the spinner arm with no
            # registrant to dial (``dial_registrants=False``).
            entry.projection = SessionProjection(
                session_id=session_id,
                pid=pid,
                kind="tui",
                conversation_name=name,
                streaming=True,
                activity="working",
            )
        daemon.table.entries[pid] = entry
        if flags.get("unseen"):
            _seed_unread(session_id)
    app = build_app(daemon)
    print(f"Fixture phone list on http://127.0.0.1:{port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
