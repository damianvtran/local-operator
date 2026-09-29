"""Capture the mesh ``/move --to`` busy refusal notice — the copy-pair surface.

Usage: env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/mesh_move_shot.py OUT.svg [COLSxROWS]

Runs the real ``OperatorApp`` with ``run_session_move`` stubbed to the
PRODUCER's own busy sentence (``mobility._busy_sentence("busy")``) and submits
``/move other1 --to pixel-8``. The stub calls the tree's own function rather
than a copy of its words, so the sentence in the frame is whatever the build
under test says — which is how the before/after pair is made: run this script
from a checkout at the base revision for the "before" frame and from the
feature tree for the "after" one (the pair this script was added with).

The wait is on the notice's own text, then five settle pauses — never a clock.
A notice that has attached but not reflowed photographs its wrap mid-update
(measured: the last row clipped), so the settle is what makes the frame show
the sentence whole.

``run_session_move`` is replaced on the app module because that is the seam the
TUI's own cells patch (``tests/unit/tui/test_remote_open.py``): the frame must
come from the app's real ``_publish_move_result`` path, with only the CLI
subprocess stubbed out.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.network import mobility  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    NoticeBlock,
    TranscriptView,
)
from scripts.visual_capture import (  # noqa: E402
    refuse_flag_shaped_argument,
    save_capture,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


async def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: mesh_move_shot.py OUT.svg [COLSxROWS]")
    # Before it is used as a path: a mistyped flag here writes ``--out.svg``.
    refuse_flag_shaped_argument(sys.argv[1], what="OUT.svg")
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))

    def fake_move(session_id: str, to: str, *, keep: bool = False) -> dict[str, object]:
        return {
            "ok": False,
            "code": "busy",
            "message": mobility._busy_sentence("busy"),
            "session_id": session_id,
            "phase_reached": None,
            "changed": False,
        }

    import local_operator.tui.app as app_mod

    app_mod.run_session_move = fake_move  # type: ignore[assignment]

    def _no_resume(_id: str | None = None) -> object:
        return _factory(FakeSession())

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=size) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app._run_slash_command("/move other1 --to pixel-8")
        texts: list[str] = []
        for _ in range(40):
            await pilot.pause()
            texts = [
                block._text
                for block in app.query_one(TranscriptView).blocks()
                if isinstance(block, NoticeBlock)
            ]
            if any("Could not move" in text for text in texts):
                break
        for _ in range(5):
            await pilot.pause()
        print("notice:", " | ".join(texts))
        print("wrote", save_capture(app, out))


if __name__ == "__main__":
    asyncio.run(main())
