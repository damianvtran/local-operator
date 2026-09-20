"""Capture the archive/delete FLOWS: receipts, refusals, and the delete result.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/archive_flows_shot.py OUT.svg MODE [COLSxROWS]

``THEME=<palette>`` renders any mode in another palette (``THEME=light`` for the
paper ramp); the default is the app's own.

The picker frames answer "what does the list offer"; these answer the other half
of the design round — **what the feature SAYS**. Every mode drives the real
composer through ``Editor`` (the same path a user types), on an isolated
``HOME``/``LOCAL_OPERATOR_CONFIG_DIR`` with a synthetic session id, so a frame is
a capture of the product's own sentence rather than a hand-built notice.

* ``archive-receipt`` — ``/archive`` on the conversation you are in: the receipt
  that has to name the way back, and (since UX round 1) the chord.
* ``unarchive-receipt`` — ``/unarchive``: the half of the pair a user reads when
  they change their mind.
* ``delete-rehearsal`` — bare ``/delete``: the rehearsal. Since design round 1
  (D2) it names the conversation by TITLE with the id secondary, because the id is
  not on the screen it is typed into.
* ``refusal-live`` — ``/delete yes`` against a session holding a live claim: the
  guard sentence for the state a user reaches by being IN the conversation.
* ``refusal-wake-armed`` — ``/delete yes`` against a conversation with an ARMED
  wake: the sentence UX round 1 (U1) rewrote, because it named an action with no
  door.
* ``dormant-wake-deletes`` — the same store with the wake marked dormant (what
  ``/stop`` leaves behind): the delete SUCCEEDS, which is UX round 1's U2. The
  frame is the receipt, because "deleted" is the evidence that the guard no
  longer traps a stopped conversation.
* ``delete-done`` — a plain ``/delete yes``: the receipt and the landing on a
  fresh conversation.
* ``eviction`` — the 200-cap receipt, naming the conversation the cap put back
  into every list.

The fixture is written through the real writers (``Transcript``-shaped JSONL,
``write_session_title``, ``wakes.store.write_entry``, ``archive_change``), so a
frame cannot agree with a wrong implementation of a store.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.paths import config_dir  # noqa: E402
from local_operator.resume import write_session_title  # noqa: E402
from local_operator.session.archived import ARCHIVED_LIMIT  # noqa: E402
from local_operator.session.cleanup import mark_store  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from tests.unit.tui.test_app_pilot import (  # noqa: E402
    FakeSession,
    _await_session,
    _factory,
)

#: The id the app's own session carries. Real-length hex and NOT a word, because
#: a fixture id that reads as an English word is what made the old copy look
#: truncated in the round-1 frame ("sess").
SESSION = "c3d4e5f6a1b2"
TITLE = "Parser crash on nested frontmatter"
OTHER = "a1b2c3d4e5f6"

#: Sixteen characters of hex, so the sentence is measured at production width.
SECOND_TITLE = "Retention sweep for the analytics ledger"


class _SeededSession(FakeSession):
    """The app's OWN conversation, carrying a production-shaped id.

    ``FakeSession`` answers ``"sess"``, and the delete sentences now name the
    conversation by TITLE with the id beside it (design round 1, D2) — a frame
    whose id is an English word cannot show whether that sentence reads as
    intended, and cannot show the rehearsal naming the conversation at all, since
    the id it prints would not be the one the store holds. This session IS the one
    the store is seeded under, so the rehearsal, the refusals, the receipts and the
    delete result all address the seeded conversation.
    """

    @property
    def session_id(self) -> str:
        return SESSION


def _write_transcript(directory: Path, text: str) -> None:
    (directory / "transcript.jsonl").write_text(
        json.dumps({"type": "message", "payload": {"role": "user", "content": text}}) + "\n",
        encoding="utf-8",
    )


def _seed(root: Path, mode: str) -> None:
    mark_store(root / "sessions")
    for session_id, title in ((SESSION, TITLE), (OTHER, SECOND_TITLE)):
        directory = root / "sessions" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        _write_transcript(directory, f"{title}. Let's pick this up.")
        (directory / "created_at.json").write_text("1700000000", encoding="utf-8")
        write_session_title(directory, title, user_set=True, past_names=[])
    if mode == "archive-receipt":
        return
    if mode == "eviction":
        # One archive per slot, through the store's own writer, so the receipt's
        # eviction names an entry the cap really dropped.
        from local_operator.session.archived import archive_change

        for index in range(ARCHIVED_LIMIT):
            filler = root / "sessions" / f"{index:012x}"
            filler.mkdir(parents=True, exist_ok=True)
            archive_change(root, filler.name, True)
        return
    if mode in {"refusal-wake-armed", "dormant-wake-deletes"}:
        from local_operator.harness.wake import WakeSchedule
        from local_operator.wakes.store import write_entry

        schedule = WakeSchedule(
            id="w1", message="check the parser fix", next_due_at=4_102_444_800_000
        )
        preserve = None
        if mode == "dormant-wake-deletes":
            # WHAT `/stop` LEAVES, written through the same call the stop path
            # makes (`control._mark_wakes_dormant`: schedules kept, `stopped_at`
            # carried through ``preserve``). Hand-writing the JSON would let the
            # frame agree with a wrong idea of what dormancy is.
            preserve = {"stopped_at": 1_700_000_000_000}
        write_entry(root, SESSION, cwd=str(root), schedules=[schedule], preserve=preserve)


async def main() -> None:
    out = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "archive-receipt"
    size = (100, 30)
    if len(sys.argv) > 3 and "x" in sys.argv[3]:
        cols, rows = sys.argv[3].split("x")
        size = (int(cols), int(rows))
    theme = os.environ.get("THEME", "")

    root = Path(config_dir())
    _seed(root, mode)

    if mode == "refusal-live":
        # A live claim, written the way a runtime writes one: the guard reads the
        # pid and asks whether the process is alive, and this process is.
        (root / "sessions" / SESSION / ".session.pid").write_text(
            str(os.getpid()), encoding="utf-8"
        )

    async def resume_factory(_session_id: str | None):
        return _SeededSession()

    app = OperatorApp(lambda: _factory(_SeededSession()), resume_factory=resume_factory)
    async with app.run_test(size=size) as pilot:
        # THE SESSION MUST BE BOUND BEFORE TYPING, or every one of these commands
        # acts on a session-less app: `/delete` answers "nothing saved yet" and
        # the frame shows a sentence about a conversation that does not exist.
        # The band saying "connecting..." in a frame is the tell.
        await _await_session(app, pilot)
        if theme:
            from local_operator.tui import theme as theme_mod

            theme_mod.set_theme(theme)
            app.refresh_css()
            await pilot.pause()
        editor = app.query_one(Editor)
        command = {
            "archive-receipt": "/archive",
            "unarchive-receipt": "/unarchive",
            "delete-rehearsal": "/delete",
            "refusal-live": "/delete yes",
            "refusal-wake-armed": "/delete yes",
            "dormant-wake-deletes": "/delete yes",
            "delete-done": "/delete yes",
            "eviction": "/archive",
        }[mode]
        if mode == "unarchive-receipt":
            from local_operator.session.archived import set_archived

            set_archived(root, SESSION, True)
        if mode == "archive-receipt":
            # A LITTLE HISTORY, so the receipt is not the only thing on screen:
            # the frame has to show WHERE the notice lands, not just its text.
            from local_operator.tui.widgets.assistant import AssistantBlock

            prose = AssistantBlock()
            prose.update_text("The parser fix is in; I left the fuzzer running.")
            app._append_block(prose)
            await pilot.pause()
        editor.text = command
        editor.cursor_location = (0, len(command))
        await pilot.pause()
        if editor._picker.is_open():
            await pilot.press("escape")
            await pilot.pause()
        await pilot.press("enter")
        for _ in range(60):
            await pilot.pause()
            if not getattr(app, "_sidebar_refresh_pending", False):
                break
        await pilot.pause()
        await pilot.pause()
        save_capture(app, out)


if __name__ == "__main__":
    asyncio.run(main())
