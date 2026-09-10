"""Capture the /resume picker listing a live ``lop exec`` run beside terminal work.

Why this shot exists. Since #804 every ``lop exec`` publishes an ordinary
attachable record, so exec runs have been appearing in this picker via
``decorate_rows(include_live=True)`` — rendered identically to a conversation
the user started. An idle one-shot and the session they were sitting in half an
hour ago both read as "Ready", and an exec record is deliberately ephemeral, so
the indistinguishable row is also the one that can vanish under the cursor.

The frame is the evidence for the tag that fixes it: run this on the branch and
on its base, and the pair shows what the user gains. A test asserting the string
is present cannot show that the column lines up, that the tag reads as metadata
rather than as part of the title, or that untagged rows still align.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/exec_picker_shot.py out.svg [COLSxROWS] [--reap]

``--reap`` captures the SECOND half of the D1 evidence: it opens the picker,
kills the exec run's pid, drives the picker's own ``_tick`` until the record is
gone from the scan, and shoots the frame the user actually sees when a one-shot
ends underneath them. It also prints ``plan_columns`` before and after, because
the stills show the symptom and only the numbers show whether the column moved.

Rows are seeded through the real transcript writer and the real record
publisher, so the picker's own scan/decorate path fills the live state — the
same code the app runs. Nothing here fabricates a ``SessionRow``.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

import scripts.probe_isolation  # noqa: F401 — isolate HOME/config before app imports
from local_operator.harness.types import Message
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import SessionRecord
from local_operator.session.transcript import Transcript
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.session_picker import SessionPickerScreen, plan_columns
from scripts.visual_capture import save_capture
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: (session id, title, record kind, busy). ``kind=None`` means no live record —
#: an ordinary cold conversation, which is most of anyone's list.
SEEDED: list[tuple[str, str, Literal["tui", "exec", "daemon"] | None, bool]] = [
    ("aaaaaaaaaaaa", "Refactor the YAML loader", None, False),
    ("bbbbbbbbbbbb", "nightly release audit", "exec", False),
    ("cccccccccccc", "Draft the migration notes", None, False),
    ("dddddddddddd", "enrichment backfill sweep", "exec", True),
    ("eeeeeeeeeeee", "Debug the flaky pilot test", "tui", False),
]


async def main() -> None:
    argv = [arg for arg in sys.argv[1:] if arg != "--reap"]
    reap = "--reap" in sys.argv
    out = Path(argv[0])
    size = argv[1] if len(argv) > 1 else "100x30"
    cols, rows = (int(part) for part in size.lower().split("x"))
    cfg = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])

    # One sleeping child per live record, so each record names a pid that is
    # really alive and really distinct. Reaped in the finally below; they are
    # bounded by a short sleep as well, so an abandoned run cannot leak them.
    holders = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]) for _ in SEEDED
    ]
    try:
        await _seed_and_shoot(cfg, holders, out, cols, rows, reap=reap)
    finally:
        for holder in holders:
            holder.terminate()
        for holder in holders:
            try:
                holder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                holder.kill()


async def _seed_and_shoot(cfg, holders, out, cols, rows, *, reap: bool = False) -> None:
    for index, (sid, title, kind, busy) in enumerate(SEEDED):
        directory = cfg / "sessions" / sid
        transcript = Transcript(directory)
        await transcript.append_message(Message.user(title))
        stamp = 1_700_000_000 - index * 900
        (directory / "created_at.json").write_text(str(stamp))
        os.utime(transcript.path, (stamp, stamp))
        if kind is None:
            continue
        # A REAL record through the real publisher: this is what makes the
        # picker's own decorate_rows fill live_state and kind, rather than the
        # script asserting the state it wants to draw.
        #
        # A DISTINCT pid per record, and it is load-bearing rather than tidy:
        # ``registry.publish`` keys the file by pid (``<pid>.json``), so three
        # records published under ``os.getpid()`` overwrite each other and the
        # scan returns exactly one. The first version of this script did that
        # and captured a picker with no tag on any row — a plausible frame of
        # the feature apparently not working.
        #
        # The pid must also be ALIVE, because ``registry.scan`` reaps records
        # whose pid is gone. Long-lived helper processes give us real pids we
        # are allowed to point at without inventing liveness.
        registry.publish(
            SessionRecord(
                pid=holders[index].pid,
                kind=kind,
                session_id=sid,
                conversation_name=title,
                cwd=str(cfg),
                model_label="anthropic/claude-opus-5",
                control_port=40000 + index,
                control_key="k",
                busy=busy,
                detached=True,
            ),
            cfg,
        )

    # A resume factory is REQUIRED, not decoration: ``_cmd_resume`` refuses with
    # "resume requires a resume-capable launcher" when it is absent, and the
    # refusal renders as an ordinary notice — so a shot without one captures a
    # plausible frame of the picker never having opened. Both frames of the
    # first before/after pair for this change were exactly that.
    async def resume_factory(resume_id: str | None) -> FakeSession:
        return FakeSession()

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)
    async with app.run_test(size=(cols, rows)) as pilot:
        await pilot.pause()
        # Through the real slash command, so the frame comes from the path a
        # user takes — including ``_overlay_live_state``, which is what fills
        # the live state and the kind this shot exists to show.
        app._cmd_resume("", app._system_notice)
        for _ in range(8):
            await pilot.pause()

        # ASSERT THE PRECONDITION BEFORE TRUSTING A PIXEL. Both of this
        # script's own earlier bugs produced convincing frames of nothing:
        # without a resume factory `_cmd_resume` refuses and the refusal
        # renders as an ordinary notice, and under one shared pid the records
        # collapsed to one. A frame is only evidence once the screen it claims
        # to show is the screen that is up.
        screen = app.screen
        assert isinstance(screen, SessionPickerScreen), f"picker never opened: {type(screen)}"
        seeded_kinds = {row.id: row.kind for row in screen._all}
        for sid, _title, kind, _busy in SEEDED:
            expected = kind or ""
            assert (
                seeded_kinds.get(sid) == expected
            ), f"{sid} scanned back as {seeded_kinds.get(sid)!r}, expected {expected!r}"
        print(f"precondition OK: {seeded_kinds}")

        if not reap:
            save_capture(app, str(out))
            return

        # -- D1: the one-shot ends while the picker is open ------------------
        # The measurement that matters is the COLUMN, not the tag: the reaped
        # row is supposed to lose its own 7 characters. What must not happen is
        # every other name moving because of it.
        before = _columns(screen, cols)
        # EVERY exec record, not just one: the column is reserved for the
        # result set, so it only collapses once the LAST tagged row is gone.
        # Killing one of two leaves the reservation legitimately standing and
        # would capture a frame that proves nothing either way.
        for index, (_sid, _title, kind, _busy) in enumerate(SEEDED):
            if kind != "exec":
                continue
            holders[index].kill()
            holders[index].wait(timeout=5)

        # Drive the picker's OWN timer rather than mutating its rows: `_tick`
        # captures its `before` snapshot as its first statement, so a mutation
        # applied from outside makes the tick compare the new state against
        # itself and the repaint decision under test never runs.
        for _ in range(40):
            screen._tick()
            await pilot.pause()
            if all(row.kind != "exec" for row in screen._all):
                break
        assert all(row.kind != "exec" for row in screen._all), "the record was never reaped"

        after = _columns(screen, cols)
        print(f"plan_columns before reap : {before}")
        print(f"plan_columns after  reap : {after}")
        print(f"COLUMNS UNCHANGED        : {before == after}")
        save_capture(app, str(out))


def _columns(screen, cols: int) -> tuple[int, int, int]:
    """``plan_columns`` as the open picker would compute it right now."""
    rows = screen.visible_rows
    ages = ["1m ago"] * len(rows)
    return plan_columns(
        rows,
        min(cols - 4, 74),
        ages,
        bool(screen.body_matched_ids),
        any(getattr(row, "forked", False) for row in rows),
        True,
        screen._exec_column_latched(rows),
    )


asyncio.run(main())
