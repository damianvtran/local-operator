"""Capture the resumed-transcript paging frames in the production CSS host.

Evidence for the "a resumed conversation cannot be scrolled up" defect. The
lightweight test hosts declare no ``CSS_PATH``, so only the real
:class:`OperatorApp` shows what the reader actually sees — see AGENTS.md,
"Visual validation".

Usage:
    python scripts/resume_paging_shot.py OUT.svg FRAME [COLSxROWS]

FRAME is one of:
    resumed      the first frame of a resumed conversation (the bug frame:
                 pre-fix this has no scrollbar and 4 blocks, while the top row
                 promises "older messages above")
    scrolled     the same conversation after scrolling to the top, i.e. what
                 the notice is asking the reader to do
    mount-0..3   consecutive frames across ONE older-page mount, sampled after
                 a ``pause``
    paint-0..3   the SAME mount sampled at the COMPOSITOR REFRESH, which is the
                 moment the terminal is actually written. This is the frame
                 pair that proves the jitter: a ``pause`` drains the whole
                 callback queue and coalesces the entire settle into one
                 observation, so the ``mount-*`` frames show a displaced paint
                 as though it never happened. Pre-fix, ``paint-1`` sits a page
                 below ``paint-0``; post-fix every paint is identical.
    wedged       the head notice after a page whose insert settle was DROPPED
                 and five subsequent clicks. ``call_after_refresh`` is
                 ``post_message``, which returns False on a closing pump, so a
                 view removed under an in-flight page left its paging lease
                 mounted and unreleasable. Pre-fix this frame is frozen — the
                 row still reads "older messages above" and the five clicks
                 moved nothing; post-fix the gate reopens and the clicks page.
    loading      the head notice while an UNMOUNTED lease holds the gate, i.e.
                 ``loading older messages…``. It exists because that copy is
                 the state this fix is most about and it had never been seen
                 on a painted frame: a synthetic session settles inside one
                 paint, so exporting "during" a real load captures the settled
                 row instead (design review round 1, D4). The lease is taken
                 and HELD here rather than the fetch being slowed, so the row
                 is painted from the same ``_reconcile_head_notice`` branch a
                 real in-flight fetch drives.
    unreachable  the head notice in its ``— click to load`` state: more
                 history exists and the frame cannot scroll to it, so the row
                 offers itself as the control. Reached by suppressing the
                 geometry fill, which is what the tall-viewport case does in
                 practice (design review round 1, D3).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.session_presentation import OlderHistoryNotice  # noqa: E402
from local_operator.tui.widgets.transcript import TranscriptView  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

FRAMES = [
    "resumed",
    "scrolled",
    *[f"mount-{i}" for i in range(4)],
    *[f"paint-{i}" for i in range(4)],
    "wedged",
    "loading",
    "unreachable",
]

#: How many times the ``wedged`` frame activates the head notice before the
#: capture. The operator's report was "clicked it and nothing happened", so
#: one click would not distinguish a slow page from a dead control; five
#: establishes that no number of asks recovers the pre-fix state.
_WEDGED_CLICKS = 5


def _history(steps: int = 200, followups: int = 1) -> list[Any]:
    """The operator's shape: one prompt, a long tool run, a short follow-up."""
    rows: list[Any] = [
        SimpleNamespace(
            role="user",
            id="u-0",
            text="Audit every row in the export and tell me which ones fail validation.",
            tool_calls=None,
            content=[],
            custom_type=None,
        )
    ]
    for k in range(steps):
        rows.append(
            SimpleNamespace(
                role="assistant",
                id=f"a-0-{k}",
                text=f"Checking record {k:03d} against the schema.",
                tool_calls=[
                    SimpleNamespace(
                        id=f"call-0-{k}",
                        name="bash",
                        arguments={"command": f"validate --row {k}"},
                    )
                ],
                custom_type=None,
                stop_reason=None,
                provider_payload=None,
            )
        )
        rows.append(
            SimpleNamespace(
                role="tool",
                id=f"t-0-{k}",
                tool_call_id=f"call-0-{k}",
                text=f"exit code: 0\nrow {k:03d}: ok",
                is_error=False,
                provider_payload=None,
                content=[],
                custom_type=None,
            )
        )
    for f in range(followups):
        rows.append(
            SimpleNamespace(
                role="user",
                id=f"fu-{f}",
                text="Thanks — now summarise the failures.",
                tool_calls=None,
                content=[],
                custom_type=None,
            )
        )
        rows.append(
            SimpleNamespace(
                role="assistant",
                id=f"fa-{f}",
                text="Three rows failed validation: 041, 118 and 176.",
                tool_calls=None,
                custom_type=None,
                stop_reason=None,
                provider_payload=None,
            )
        )
    return rows


def grid(value: str) -> tuple[int, int]:
    try:
        columns, rows = (int(n) for n in value.split("x"))
        if not (20 <= columns <= 400 and 10 <= rows <= 150):
            raise ValueError
        return columns, rows
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "grid must be COLSxROWS within 20..400 by 10..150"
        ) from exc


async def capture(path: Path, frame: str, size: tuple[int, int]) -> None:
    session: Any = FakeSession()
    # The jitter frames drop the trailing follow-up on purpose. WITH it, the
    # pre-fix tree cannot scroll at all (that is Defect 1), so a before/after
    # mount pair would be comparing "no scrollbar" against "no jitter" and
    # would prove neither. Without it both trees open on a scrollable frame,
    # and the only difference across the pair is the insert's behaviour.
    jitter = frame.startswith("mount-") or frame.startswith("paint-")
    session._history = _history(followups=0 if jitter else 1)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        for _ in range(60):
            await pilot.pause()
            if app.query_one(TranscriptView).blocks():
                break
        # Let the initial mount AND any scrollability fill settle.
        for _ in range(40):
            await pilot.pause()
        view = app.query_one(TranscriptView)

        if frame == "scrolled":
            view.note_user_scroll()
            view.scroll_home(animate=False)
            for _ in range(12):
                await pilot.pause()
        elif frame == "wedged":
            # Refuse EXACTLY the settle `insert_blocks` schedules, once. That
            # is not a contrivance: it is the same False Textual's
            # `post_message` returns when `_release_sidebar_preparation`
            # removes a transcript under a page that has already flagged its
            # lease mounted. Arming it this way keeps the capture in one
            # process and off the sidebar's whole navigation path.
            real = view.call_after_refresh
            armed = {"hit": False}

            def refuse(callback, *callback_args, **callback_kwargs):
                name = getattr(callback, "__name__", "")
                if not armed["hit"] and name == "settle_then_restore":
                    armed["hit"] = True
                    return False
                return real(callback, *callback_args, **callback_kwargs)

            view.call_after_refresh = refuse  # type: ignore[method-assign]
            app._mount_older_resume_page()
            view.call_after_refresh = real  # type: ignore[method-assign]
            for _ in range(60):
                await pilot.pause()
            if not armed["hit"]:
                raise SystemExit("the settle was never refused; the hazard was not armed")
            notice = app._resume_head_notice
            for _ in range(_WEDGED_CLICKS):
                if isinstance(notice, OlderHistoryNotice):
                    notice.post_message(OlderHistoryNotice.Requested(notice))
                for _ in range(60):
                    await pilot.pause()
            # At the top, where the notice is, so the frame shows the row the
            # reader was clicking rather than the tail they never left.
            #
            # ASK REPEATEDLY, AND ASSERT. One `scroll_home` is enough on the
            # broken branch (nothing loads, so nothing moves) and NOT enough
            # on the fixed one: reaching the head reopens the gate, every
            # recovered page mounts above the reader, and the anchor restore
            # holds their rows — which puts the offset back where it was. The
            # single-shot version therefore captured the "after" frame with
            # the notice 46 rows above the viewport, so the evidence for a
            # fix to a ROW did not contain that row and a reader could not
            # tell repair from removal (design review round 1, D1).
            view.note_user_scroll()
            for _ in range(60):
                view.scroll_home(animate=False)
                await pilot.pause()
                head = app._resume_head_notice
                if view.scroll_y <= 0.5 and head is not None and head.region.y >= 0:
                    break
            for _ in range(4):
                await pilot.pause()
            head = app._resume_head_notice
            # The script must never again silently produce a frame without its
            # subject: a capture that cannot show the row is a failed capture,
            # not a frame to be published with a caption explaining it away.
            if head is None or head.region.y < 0:
                where = None if head is None else head.region.y
                raise SystemExit(
                    f"the head notice is not in the viewport (y={where}); "
                    f"the frame would not show its subject"
                )
        elif frame in ("loading", "unreachable"):
            # Both states are decided by `_reconcile_head_notice` from the
            # CURRENT lease and geometry, so each is reached by establishing
            # that state and letting the production funnel restate the row —
            # never by writing the copy onto the widget, which would prove
            # only that the string exists.
            if frame == "loading":
                # An UNMOUNTED lease is the "a fetch is in flight" state. Held
                # for the capture instead of racing a real fetch: on a
                # synthetic session the round trip settles within one paint,
                # so every attempt to export mid-load exported the settled row.
                lease = app._acquire_paging_lease(app._interaction)
                if lease is None:
                    raise SystemExit("the paging gate was already held; cannot stage `loading`")
            else:
                # `RESUME_UNREACHABLE_NOTICE` needs more-exists AND an
                # unscrollable frame AND no fill in flight. The fill is what
                # normally prevents it, so stand it down and shrink the
                # content back to fit — the same end state a very tall
                # viewport reaches with the fill capped.
                # Strip everything below the notice so the content no longer
                # exceeds the viewport, which is the `not scrollable` half of
                # the state. The deferred head is untouched, so "more exists"
                # stays true — exactly the combination a very tall viewport
                # produces when the fill cannot outgrow the screen.
                for block in list(view.blocks())[1:]:
                    view.remove_block(block)
                # Let the extent shrink before anything reads it. The offset
                # follows on its own once the canvas is one viewport tall;
                # scrolling here instead would move against the OLD extent.
                for _ in range(20):
                    await pilot.pause()
                # Cleared last, so a fill in flight cannot re-arm it while the
                # rows are being taken out from under it.
                app._resume_fill_active = False
            if frame == "loading":
                # The `loading` frame keeps its full transcript, so the notice
                # has to be scrolled back into view. `unreachable` deliberately
                # skips this: its canvas is already one viewport tall (nothing
                # to scroll), and `note_user_scroll` would re-arm the demand
                # this state is defined by the absence of.
                view.note_user_scroll()
                view.scroll_home(animate=False)
                for _ in range(8):
                    await pilot.pause()
            app._reconcile_head_notice()
            for _ in range(8):
                await pilot.pause()
            head = app._resume_head_notice
            # Assert the state actually painted: a frame captioned with a copy
            # it does not contain is worse than no frame (D1's lesson).
            wanted = "loading older messages" if frame == "loading" else "click to load"
            if head is None or wanted not in head.text() or head.region.y < 0:
                raise SystemExit(
                    f"expected {wanted!r} on screen, got "
                    f"{None if head is None else (head.text(), head.region.y)}"
                )
        elif frame.startswith("paint-"):
            # Same parking as `mount-*`, but the SVG is exported from inside
            # the compositor refresh — the moment the terminal is written —
            # rather than after a `pause`. Nothing else can see a displaced
            # frame: a pause drains the callback queue, so by the time it
            # returns the correction has already run.
            view.note_user_scroll()
            view.scroll_to(y=max(8.0, view.max_scroll_y / 2), animate=False)
            for _ in range(20):
                await pilot.pause()
            # EVERY paint of the sequence is captured in THIS ONE run, and the
            # requested index is written to ``path``. Capturing one index per
            # process and comparing across processes is invalid: the parking
            # offset depends on ``max_scroll_y`` at that instant, so two runs
            # can legitimately park at different rows and the "difference"
            # between their frames says nothing about the insert.
            count = len(FRAMES) - FRAMES.index("paint-0")
            wanted = int(frame.split("-")[1])
            screen = app.screen
            original_refresh = screen._compositor_refresh
            seen = {"n": 0}

            def compositor_refresh() -> None:
                original_refresh()
                index = seen["n"]
                seen["n"] += 1
                if index >= count:
                    return
                top = next(
                    (b for b in view.blocks() if b.virtual_region.bottom > view.scroll_y),
                    None,
                )
                g = (top.virtual_region.y - view.scroll_y) if top is not None else 0.0
                out = (
                    path
                    if index == wanted
                    else path.with_name(path.name.replace(f"paint-{wanted}", f"paint-{index}"))
                )
                save_capture(app, out)
                print(
                    f"{out}  PAINT#{index} blocks={len(view.blocks())} "
                    f"virtual_h={view.virtual_size.height} scroll_y={view.scroll_y:.0f} "
                    f"top_gap={g:.0f}"
                )

            screen._compositor_refresh = compositor_refresh  # type: ignore[method-assign]
            try:
                app._mount_older_resume_page()
                for _ in range(16):
                    await pilot.pause()
            finally:
                screen._compositor_refresh = original_refresh  # type: ignore[method-assign]
            if seen["n"] <= wanted:
                raise SystemExit(f"the mount painted fewer than {wanted + 1} frames")
            return
        elif frame.startswith("mount-"):
            # Park the reader MID-transcript, then step one mount frame by frame.
            #
            # Mid-transcript, not in the trigger zone, for two reasons. The
            # anchor restore in `insert_blocks` holds the reader wherever they
            # are, so the jitter is visible at any offset — and parking inside
            # the zone is itself a gesture that mounts a page, which would
            # confound frame 0 with a mount already in flight. `note_user_scroll`
            # still comes first: it releases the tail anchor, without which the
            # mount legitimately drags the viewport to the end and the capture
            # shows tail-following rather than the insert.
            view.note_user_scroll()
            view.scroll_to(y=max(8.0, view.max_scroll_y / 2), animate=False)
            for _ in range(20):
                await pilot.pause()
            index = int(frame.split("-")[1])
            if index:
                app._mount_older_resume_page()
                for _ in range(index):
                    await pilot.pause()

        # The first visible block and its distance to the viewport top: this is
        # the quantity the jitter frames are about, so it is printed with them
        # rather than left to be inferred from the picture.
        top_block = next(
            (b for b in view.blocks() if b.virtual_region.bottom > view.scroll_y), None
        )
        gap = (top_block.virtual_region.y - view.scroll_y) if top_block is not None else 0.0
        geometry = (
            f"blocks={len(view.blocks())} virtual_h={view.virtual_size.height} "
            f"viewport={view.container_size.height} scroll_y={view.scroll_y:.0f} "
            f"max_scroll_y={view.max_scroll_y} scrollbar={bool(view.show_vertical_scrollbar)} "
            f"top_gap={gap:.0f} paging={app._resume_paging}"
        )
        save_capture(app, path)
        print(f"{path}  {geometry}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("frame", choices=FRAMES)
    parser.add_argument("grid", nargs="?", default="120x40", type=grid)
    args = parser.parse_args()
    asyncio.run(capture(args.out, args.frame, args.grid))


if __name__ == "__main__":
    main()
