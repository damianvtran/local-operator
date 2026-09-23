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
from local_operator.mobile.types import SessionProjection, SubagentRow
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
    # THE DEPTH-2 ROW (design round 2, D1's deferral). The record's count is the
    # WHOLE TREE — the runtime counts over ``comms.nodes()``, which already holds
    # every nested descendant — while the session view's roster header counts
    # DIRECT children only (``SubagentsPanel`` passes ``parentJobId={null}`` and
    # ``AgentRoster`` filters on it). Two direct children, one of which spawned two
    # of its own, is therefore ``4 subagents`` on this list and ``2/2 running`` on
    # the view one tap later. BOTH numbers are real and neither is a bug: they
    # count two populations, which is the deferred finding this row exists to draw
    # in one rig (see the design record's "Not in scope" section).
    (
        "nested",
        910009,
        "Parent with nested descendants",
        {"subagents_running": 4},
        {},
    ),
    # THE ONE-RUNNING-ONE-WAITING ROW (UX round 3). The list chip counts both
    # (``2 subagents``); the session view must agree on the children AND keep the
    # waiting one out of the running lane, which is what the projection's status
    # mapping decides.
    (
        "mixed",
        910010,
        "Parent with one running, one waiting",
        {"subagents_running": 1, "subagents_queued": 1},
        {},
    ),
]

#: The child trees the SESSION VIEW must draw, keyed by session id, in the MOBILE
#: vocabulary the fold emits. ``mobile/projection.py`` maps the runtime's lifecycle
#: status onto these, and the mapping is the thing UX round 3 changed: a
#: capacity-parked child arrives here as ``queued`` and must NOT be drawn as
#: running, because the roster header counts this field. (This fixture serves an
#: ``entry.projection`` directly, which is what the daemon serves for a session it
#: has no relay for — the mapping itself is pinned by its own test in
#: ``tests/unit/mobile/test_projection.py``, not by these frames.)
TREES: dict[str, list[SubagentRow]] = {
    "mixed": [
        SubagentRow(job_id="job-r", label="running child"),
        SubagentRow(job_id="job-q", label="waiting child", status="queued"),
    ],
    "parked": [
        SubagentRow(job_id="job-p1", label="waiting one", status="queued"),
        SubagentRow(job_id="job-p2", label="waiting two", status="queued"),
        SubagentRow(job_id="job-p3", label="waiting three", status="queued"),
    ],
    "nested": [
        SubagentRow(job_id="job-a", label="first child"),
        SubagentRow(job_id="job-b", label="second child"),
        SubagentRow(job_id="job-a1", label="nested one", parent_job_id="job-a"),
        SubagentRow(job_id="job-a2", label="nested two", parent_job_id="job-a"),
    ],
}


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
        elif session_id in TREES:
            entry.projection = SessionProjection(
                session_id=session_id,
                pid=pid,
                kind="tui",
                conversation_name=name,
                activity="waiting",
                subagents=TREES[session_id],
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
