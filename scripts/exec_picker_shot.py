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
        scripts/exec_picker_shot.py out.svg [COLSxROWS]

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
    out = Path(sys.argv[1])
    size = sys.argv[2] if len(sys.argv) > 2 else "100x30"
    cols, rows = (int(part) for part in size.lower().split("x"))
    cfg = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])

    # One sleeping child per live record, so each record names a pid that is
    # really alive and really distinct. Reaped in the finally below; they are
    # bounded by a short sleep as well, so an abandoned run cannot leak them.
    holders = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]) for _ in SEEDED
    ]
    try:
        await _seed_and_shoot(cfg, holders, out, cols, rows)
    finally:
        for holder in holders:
            holder.terminate()
        for holder in holders:
            try:
                holder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                holder.kill()


async def _seed_and_shoot(cfg, holders, out, cols, rows) -> None:
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
        save_capture(app, str(out))


asyncio.run(main())
