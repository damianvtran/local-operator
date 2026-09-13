"""Capture the two-pane `/resume` picker in the production CSS host.

Evidence for the Telescope redesign: the list and the live conversation
preview, across the widths that cross the stacked/side-by-side breakpoint. The
lightweight test hosts declare no ``CSS_PATH``, so only the real
:class:`OperatorApp` shows what the reader actually sees — see AGENTS.md,
"Visual validation".

Usage:
    python scripts/resume_telescope_shot.py OUT.svg FRAME [COLSxROWS]

FRAME is one of:
    list    the picker with no query, at whatever geometry is asked for
    narrowed
            the picker filtered down to FEWER rows than one page holds, which
            is the case that skips the position counter — and the case a
            design round caught reporting the whole store's size instead of
            the match count.
    query   the picker with a filter typed, whose CURSOR ROW is an exact body
            match — so the frame actually demonstrates the grep context line.
            Design round 2 spent a whole round unable to verify the stacked
            context line because its only stacked query frame's cursor row was
            a FUZZY hit, for which context is suppressed by design.

Frames are written wherever you point them and belong in ``/tmp``, never in the
tree: AGENTS.md forbids committed PR evidence artifacts. The SCRIPT is the
reusable part.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.probe_isolation  # noqa: F401,E402 — isolate HOME/config before app imports
from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.paths import config_dir  # noqa: E402
from local_operator.session.session import Message  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.session_picker import SessionPickerScreen  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

FRAMES = ("list", "query", "narrowed")

#: The query the ``query`` frame types. Chosen because the seeded bodies below
#: contain it as a literal substring, which is what makes the cursor row an
#: EXACT match and so gives it a context line.
QUERY = "picker"

#: A query that matches only the three named sessions, so the list fits one
#: page and the position counter is skipped.
NARROWING_QUERY = "asteroids"

#: Conversations seeded into the isolated store. The first is the one the
#: cursor lands on under ``QUERY``; its body carries the literal query so the
#: context line has something to centre on.
SEEDED: tuple[tuple[str, str, str], ...] = (
    (
        "aa00000000001",
        "Fixing input focus and scroll in /btw command",
        "the /resume picker shows the current name; backfill_session_titles "
        "recovers titles for older sessions so the list reads as names",
    ),
    (
        "bb00000000002",
        "Make an asteroids game in pygame",
        "draw the ship, then the rocks, then the collisions",
    ),
    (
        "cc00000000003",
        "Why does the retention sweep keep the parent",
        "retention evicts the older parent first, which is why the picker "
        "shows children on a machine whose parents are gone",
    ),
)


async def _seed(cfg: Path, count: int) -> None:
    """Write real transcripts through the real writer, newest first."""
    for index, (sid, title, body) in enumerate(SEEDED):
        directory = cfg / "sessions" / sid
        transcript = Transcript(directory)
        await transcript.append_message(Message.user(title))
        await transcript.append_message(Message.assistant(body))
        stamp = 1_700_000_000 - index * 900
        (directory / "created_at.json").write_text(str(stamp))
        os.utime(transcript.path, (stamp, stamp))

    # Filler, so the list is longer than any terminal under test can draw and
    # the frames show a scrolling list — the shape a real store is always in.
    for filler in range(count):
        sid = f"f{filler:011d}"
        directory = cfg / "sessions" / sid
        transcript = Transcript(directory)
        await transcript.append_message(Message.user(f"routine follow-up {filler}"))
        await transcript.append_message(
            Message.assistant(f"handled the {filler} case and moved the picker on")
        )
        stamp = 1_700_000_000 - (len(SEEDED) + filler) * 900
        (directory / "created_at.json").write_text(str(stamp))
        os.utime(transcript.path, (stamp, stamp))


async def _shoot(out: Path, frame: str, cols: int, rows: int) -> None:
    cfg = config_dir()
    await _seed(cfg, max(12, rows * 2))

    # A resume factory is REQUIRED, not decoration: ``_cmd_resume`` refuses
    # with "resume requires a resume-capable launcher" when it is absent, and
    # the refusal renders as an ordinary notice — so a shot without one
    # captures a plausible frame of the picker never having opened.
    async def resume_factory(resume_id: str | None) -> FakeSession:
        return FakeSession()

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)
    async with app.run_test(size=(cols, rows)) as pilot:
        await pilot.pause()
        # Through the real slash command, so the frame comes from the path a
        # user takes — including the ``created_at`` enrichment the preview
        # header's "started X ago" reads.
        app._cmd_resume("", app._system_notice)
        for _ in range(8):
            await pilot.pause()
        # The panes are restyled from Python on the first paint, and the text
        # is wrapped to the width they resolve to — so the FIRST paint is
        # against the pre-restyle geometry and the settled one is a frame
        # later. Repaint once the layout has settled, or the capture shows a
        # body wrapped to the wrong width and a role header left orphaned
        # under it. Measured at 100x30, where the pre-settle paint wrapped a
        # 47-cell line in a 96-cell pane.
        screen_now = app.screen
        if isinstance(screen_now, SessionPickerScreen):
            screen_now._repaint()
        for _ in range(4):
            await pilot.pause()

        # ASSERT THE PRECONDITION BEFORE TRUSTING A PIXEL. A frame is evidence
        # only once the screen it claims to show is the screen that is up.
        screen = app.screen
        assert isinstance(screen, SessionPickerScreen), f"picker never opened: {type(screen)}"
        assert len(screen.visible_rows) >= len(
            SEEDED
        ), f"store seeded {len(screen.visible_rows)} rows, expected at least {len(SEEDED)}"

        if frame == "narrowed":
            for char in NARROWING_QUERY:
                await pilot.press(char)
            for _ in range(6):
                await pilot.pause()
            matches = len(screen.visible_rows)
            budget = screen._layout().list_rows
            # ASSERT THE SHAPE: a frame that still scrolls exercises the
            # counter branch, not the one under review.
            assert 0 < matches <= budget, (
                f"{matches} matches against a {budget}-row page — this frame "
                "does not show the single-page case"
            )
            print(f"narrowed: {matches} matches, page holds {budget}")

        if frame == "query":
            for char in QUERY:
                await pilot.press(char)
            for _ in range(6):
                await pilot.pause()
            # ASSERT THE SHAPE, not just the screen. Context is suppressed by
            # design on a fuzzy hit, so a query frame whose cursor row matched
            # softly demonstrates nothing about the context line.
            assert screen.visible_rows, f"query {QUERY!r} matched nothing"
            cursor = screen.visible_rows[screen.selected_index]
            context = screen._raw_context(cursor)
            assert context is not None, (
                f"cursor row {cursor.id} is not an EXACT body match — "
                "this frame cannot demonstrate the context line"
            )
            assert (
                QUERY in context.lower()
            ), f"context line does not carry the query (D17): {context!r}"
            print(f"context OK: {context[:90]}")

        mode = screen.layout_mode_for_test()
        drawn = len(screen.render_lines_for_test())
        print(
            f"shape: {cols}x{rows} mode={mode} rows_drawn={drawn} "
            f"matching={len(screen.visible_rows)} total={len(screen._all)}"
        )
        print(f"footer: {screen.render_footer_for_test()}")
        preview = screen.render_preview_for_test()
        assert any(line.strip() for line in preview), "preview pane painted nothing"
        print(f"preview[0:3]: {preview[:3]}")

        out.parent.mkdir(parents=True, exist_ok=True)
        save_capture(app, str(out))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("frame", choices=FRAMES)
    parser.add_argument("size", nargs="?", default="160x45")
    args = parser.parse_args()

    cols, _, rows = args.size.partition("x")
    asyncio.run(_shoot(args.out, args.frame, int(cols), int(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
