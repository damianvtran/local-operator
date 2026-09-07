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
from local_operator.tui.widgets.transcript import TranscriptView  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

FRAMES = [
    "resumed",
    "scrolled",
    *[f"mount-{i}" for i in range(4)],
    *[f"paint-{i}" for i in range(4)],
]


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
