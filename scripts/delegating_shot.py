"""Capture the sidebar's ``delegating`` rung: a parent that still owns children.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/delegating_shot.py \
        OUT.svg [COLSxROWS] [ladder|cursor|unseen]

``ladder`` (the default) is the whole precedence ladder with the new rung in it,
so the frame shows what the state sits BETWEEN and that nothing else moved.
``cursor`` puts the cursor on the delegating row and shows its hover tooltip —
the count lives in the words, and the caret (``›``) is the one glyph that shares
the mark column's neighbourhood, which is the residual visual risk of a new
mark. ``unseen`` isolates the precedence that has to LOSE: a delegating row that
also owns an unread completion keeps the completion mark, because a receipt the
operator has not seen must never be masked by live activity.

**The scenario is built through the real code, not hand-set fields.** Every live
state comes from a discovery record written into an isolated config root and read
back by ``registry.scan`` (the pending gate, the busy turn, the attached session,
the wedged owner and the delegating parent), the armed wake comes from the real
wake index writer, and the unread receipt comes from the real attention store —
so the frames show what a machine in those states renders, not a field somebody
typed in.

**It runs against the PRE-change build too**, which is where the "before" frame
comes from: the two count fields are only passed to ``SessionRow`` when the tree
has them (``_COUNTS`` below), so the same script produces a before/after pair and
the two frames differ only where the change does.

What this cannot prove: that the counts are ever nonzero on a live machine — that
is the runtime's job (``set_subagents``), and this capture is about what the
sidebar DOES with them.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.resume import SessionRow  # noqa: E402
from local_operator.session.attention import AttentionStore  # noqa: E402
from local_operator.session.catalog import decorate_rows  # noqa: E402
from local_operator.session.catalog import entry_for, rank_entries  # noqa: E402
from local_operator.session.runtime import registry  # noqa: E402
from local_operator.session.runtime.types import SessionRecord  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.session_catalog import SidebarSettings  # noqa: E402
from local_operator.wakes.store import write_entry  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The two count fields, or nothing at all on a tree that predates them. This is
#: what lets ONE script produce both halves of the before/after pair.
_COUNTS = (
    {"subagents_running", "subagents_queued"}
    if "subagents_running" in SessionRow._fields
    else set()
)

#: The ids are 12-hex-ish session names so they read like real ones.
GATE_ID = "gate00000001"
BUSY_ID = "busy00000001"
DELEG_ID = "deleg0000001"
ATTACHED_ID = "attach000001"
WAKE_ID = "wake00000001"
IDLE_ID = "idle00000001"
UNSEEN_ID = "unseen000001"
WEDGED_ID = "wedged000001"
COLD_ID = "cold00000001"

#: ``HEARTBEAT_TIMEOUT_S`` is 45 s; this is well past it and past the 205.8 s a
#: live machine measured, and it prints as a stable ``4m``.
QUIET_AGE_S = 243.0


def _counts(**fields: int) -> dict[str, int]:
    """``subagents_running``/``subagents_queued``, but only where they exist."""
    return {name: value for name, value in fields.items() if name in _COUNTS}


def _record(pid: int, session_id: str, name: str, **fields: object) -> None:
    """One discovery record, written as its own runtime would have left it."""
    record = SessionRecord(
        pid=pid,
        kind="daemon",
        session_id=session_id,
        conversation_name=name,
        cwd="/tmp",
        model_label="anthropic/claude-opus-5",
        control_port=12345,
        control_key="k" * 64,
        started=True,
        **fields,  # type: ignore[arg-type]
    )
    directory = registry.run_dir()
    path = directory / f"{pid}.json"
    path.write_text(json.dumps(record.to_json()))
    os.chmod(path, 0o600)
    return None


def _age(pid: int, session_id: str, name: str, age_s: float, **fields: object) -> None:
    """A record whose beat stopped ``age_s`` ago, on a pid that is still alive.

    Written directly rather than through ``RecordPublisher``: ``publish`` stamps
    a fresh heartbeat by design, and a wedged owner is exactly one whose beat
    stopped arriving.
    """
    _record(pid, session_id, name, **fields)
    path = registry.run_dir() / f"{pid}.json"
    payload = json.loads(path.read_text())
    payload["heartbeat_at"] = time.time() - age_s
    path.write_text(json.dumps(payload))
    os.chmod(path, 0o600)


def _receipt(session_id: str) -> dict[str, object]:
    """Publish an UNREAD completion through the real store, and read it back.

    The same two calls ``load_catalog`` makes: the writer stamps the receipt and
    ``state_many`` is what a reader sees. Nothing here sets ``unseen`` by hand.
    """
    store = AttentionStore(registry.config_dir() / "attention.db")
    store.publish(f"session/{session_id}", str(uuid.uuid4()), "a1", "complete")
    return store.state_many([f"session/{session_id}"])[f"session/{session_id}"]


async def main() -> None:
    out = Path(sys.argv[1])
    size = (110, 30)
    surface = "ladder"
    for argument in sys.argv[2:]:
        if "x" in argument:
            cols, rows = argument.split("x")
            size = (int(cols), int(rows))
        else:
            surface = argument

    # Real, live pids: a record is keyed by pid, and a synthetic one classifies
    # as ``stale`` and is reaped mid-capture. None of them may be this script's
    # or the APP's own pid — the app publishes a record for its own session, and
    # a capture that claimed that pid would have it overwritten mid-run
    # (``liveness_shot.py`` records that measurement).
    owners = [subprocess.Popen(["sleep", "300"]) for _ in range(7)]
    try:
        gate, busy, deleg, attached, unseen, wedged, idle_owner = [owner.pid for owner in owners]
        _record(gate, GATE_ID, "Waiting on a person", pending="approval", detached=True)
        _record(busy, BUSY_ID, "Working session", busy=True, detached=True)
        _record(
            deleg,
            DELEG_ID,
            "Parent owning 2 children",
            detached=True,
            **_counts(subagents_running=2, subagents_queued=1),
        )
        # The row the new state has to be distinguishable FROM: an ordinary
        # resident session, whose `●` sits in the same column two rows away.
        _record(idle_owner, IDLE_ID, "Idle session", detached=True)
        # ``detached=False`` is what another terminal watching this session
        # publishes; ``decorate_rows`` maps it to ``live_state="attached"``.
        _record(attached, ATTACHED_ID, "Watched elsewhere", detached=False)
        _record(
            unseen,
            UNSEEN_ID,
            "Unread parent with children",
            detached=True,
            **_counts(subagents_running=1, subagents_queued=0),
        )
        _age(wedged, WEDGED_ID, "Owner stopped reporting", QUIET_AGE_S, busy=True, detached=True)

        root = registry.config_dir()
        # The armed wake, through the store's own writer: ``decorate_rows`` reads
        # it back from the index rather than being handed a count.
        write_entry(
            root,
            WAKE_ID,
            cwd="/tmp",
            schedules=[{"next_due_at": int(time.time() * 1000) + 60_000, "cwd": "/tmp"}],
        )

        now = time.time()
        rows = [
            SessionRow(GATE_ID, now - 10, "Waiting on a person", created_at=now - 10),
            SessionRow(BUSY_ID, now - 20, "Working session", created_at=now - 20),
            SessionRow(DELEG_ID, now - 30, "Parent owning 2 children", created_at=now - 30),
            SessionRow(ATTACHED_ID, now - 40, "Watched elsewhere", created_at=now - 40),
            SessionRow(WAKE_ID, now - 50, "Armed reminder", created_at=now - 50),
            SessionRow(IDLE_ID, now - 60, "Idle session", created_at=now - 60),
            SessionRow(UNSEEN_ID, now - 70, "Unread parent with children", created_at=now - 70),
            SessionRow(WEDGED_ID, now - 80, "Owner stopped reporting", created_at=now - 80),
            SessionRow(COLD_ID, now - 90, "Old conversation", created_at=now - 90),
        ]
        decorated = {row.id: row for row in decorate_rows(root, rows)}
        attention = {UNSEEN_ID: _receipt(UNSEEN_ID)}
        entries = rank_entries(
            [
                entry_for(decorated[row.id], attention.get(row.id))
                for row in rows
                if row.id in decorated
            ]
        )
        print(
            "statuses:",
            [(entry.id, entry.status_code, entry.status) for entry in entries],
            flush=True,
        )
        for entry in entries:
            if entry.id == DELEG_ID:
                print(
                    "delegating row:",
                    repr(entry.status),
                    "mark/ink will be read from the frame",
                    flush=True,
                )

        if surface == "unseen":
            # Isolate the precedence that must LOSE: a delegating row beside its
            # unread twin, so the pair is the whole frame.
            entries = [entry for entry in entries if entry.id in (DELEG_ID, UNSEEN_ID, IDLE_ID)]

        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=size, tooltips=True) as pilot:
            await pilot.pause()
            app._sidebar_settings = SidebarSettings(True, "right")
            await pilot.press("ctrl+b")
            await pilot.pause()
            # Retire BOTH catalog-refresh paths so the real catalog cannot land
            # on top of these rows mid-capture (the sidebar tests' own recipe).
            if app._sidebar_timer is not None:
                app._sidebar_timer.pause()
            app._sidebar_refresh_generation += 1
            sidebar = app._session_sidebar
            sidebar.set_entries(entries)
            await pilot.pause()

            if surface in ("cursor", "unseen"):
                # FOCUS FIRST, or the frame does not contain the glyph it is
                # cited for. The caret is painted only while the list HAS focus
                # (`session_sidebar.py`: `cursor = self.has_focus and entry.id ==
                # self.cursor_id`), so a scenario that sets `cursor_id` and
                # hovers captures the tinted row and no `›` — which is exactly
                # the adjacency this frame exists to show (`›` at column 2's
                # neighbour, the residual risk the module docstring names).
                # Design round 1, D2.
                sidebar.focus()
                sidebar.cursor_id = DELEG_ID
                await pilot.pause()
                row_y = next(
                    y
                    for y in range(1, sidebar.size.height)
                    if (entry := sidebar._entry_at(y)) is not None and entry.id == DELEG_ID
                )
                await pilot.hover("#session-sidebar", offset=(8, row_y))
                await asyncio.sleep(float(app.TOOLTIP_DELAY) + 0.2)
                await pilot.pause()
                sidebar._show_tooltip_now()
                await pilot.pause()

            save_capture(app, str(out))
            print("screenshot:", out, flush=True)
    finally:
        for owner in owners:
            owner.terminate()
            owner.wait(timeout=10)


asyncio.run(main())
