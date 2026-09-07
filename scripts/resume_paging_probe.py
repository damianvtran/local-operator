"""Measure resume transcript geometry and page-mount jitter.

Evidence tool for the "resumed transcript cannot be scrolled up" defect. It
answers two questions the stills alone cannot:

1. Is the FIRST frame of a resumed conversation scrollable at all? Prints the
   block count, ``virtual_size`` vs ``container_size``, ``show_vertical_scrollbar``
   and whether the head notice is promising history the reader can reach.
2. Does mounting an older page move the rows under the reader's eyes? Records
   ``scroll_y`` and the visible row strip on every painted frame across the
   mount, so a displaced frame shows up as a row-strip that differs from the
   settled one.

Usage:
    python scripts/resume_paging_probe.py geometry [SHAPE] [COLSxROWS]
    python scripts/resume_paging_probe.py jitter   [SHAPE] [COLSxROWS]
    python scripts/resume_paging_probe.py reader   [SHAPE] [COLSxROWS]

``reader`` is the one to show a human: it prints, for every painted frame of a
page mount, the TEXT of the block at the top of the viewport. A correct insert
prints the same line on every frame; the pre-fix code prints three different
lines and then returns to the first.

SHAPE is one of the conversation shapes in ``SHAPES`` below; ``agentic`` is the
operator's reported case (one prompt, a long run of assistant/tool rows).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture  # noqa: E402

isolate_capture()

from textual.events import MouseScrollUp  # noqa: E402

from local_operator.tui.app import (  # noqa: E402
    RESUME_PAGE_MESSAGES,
    RESUME_PAGE_TRIGGER_ROWS,
    OperatorApp,
)
from local_operator.tui.widgets.transcript import (  # noqa: E402
    NoticeBlock,
    TranscriptView,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


def _user(i: int) -> Any:
    return SimpleNamespace(
        role="user",
        id=f"u-{i}",
        text=f"turn {i:04d}: please check item {i}",
        tool_calls=None,
        content=[],
        custom_type=None,
    )


def _assistant(i: int, k: int) -> Any:
    return SimpleNamespace(
        role="assistant",
        id=f"a-{i}-{k}",
        text=f"Step {k:03d} of turn {i:04d}: inspecting the next candidate row.",
        tool_calls=[
            SimpleNamespace(id=f"call-{i}-{k}", name="bash", arguments={"command": f"echo {k}"})
        ],
        custom_type=None,
        stop_reason=None,
        provider_payload=None,
    )


def _tool(i: int, k: int) -> Any:
    return SimpleNamespace(
        role="tool",
        id=f"t-{i}-{k}",
        tool_call_id=f"call-{i}-{k}",
        text=f"exit code: 0\nitem-{i}-{k}",
        is_error=False,
        provider_payload=None,
        content=[],
        custom_type=None,
    )


def _shape(turns: int, steps: int) -> list[Any]:
    """``turns`` prompts, each followed by ``steps`` assistant/tool pairs."""
    rows: list[Any] = []
    for i in range(turns):
        rows.append(_user(i))
        for k in range(steps):
            rows.append(_assistant(i, k))
            rows.append(_tool(i, k))
    return rows


#: The shapes worth measuring. ``agentic`` is the operator's screenshot: ONE
#: prompt and a long run of tool work, which is the normal shape here.
#:
#: ``followup`` is the shape the forward walk in `_resume_tail_start` actually
#: degenerates on, and it is not exotic: a long agentic turn, then a SHORT
#: recent prompt ("thanks, now do X") whose user row lands INSIDE the last
#: `bound` messages. The forward walk snaps to that late row and renders only
#: the handful of messages after it.
SHAPES = {
    "agentic": lambda: _shape(1, 200),
    "agentic-few": lambda: _shape(5, 100),
    "followup": lambda: _shape(1, 200) + [_user(9), _assistant(9, 0), _tool(9, 0)],
    "ordinary": lambda: _shape(60, 3),
    "short": lambda: _shape(4, 2),
}


def _wheel_to_trigger(view: TranscriptView) -> None:
    """Post ONE real wheel notch at the transcript.

    The widget's own input surface, so it routes through `note_user_scroll`
    and the page-back latch exactly as a hand on a mouse does. One notch is
    enough when the reader is already parked inside the trigger zone, and one
    is the point: the contract under test is "one arrival, one page".
    """
    view.post_message(
        MouseScrollUp(
            widget=view,
            x=5,
            y=5,
            delta_x=0,
            delta_y=-1,
            button=0,
            shift=False,
            meta=False,
            ctrl=False,
        )
    )


def _rows(app: OperatorApp, view: TranscriptView) -> list[str]:
    """The PAINTED rows the transcript occupies on this frame.

    Read from the compositor, not from ``render_lines``: the compositor is what
    was actually put on the terminal, so a frame captured here is a frame the
    reader's eyes saw. Sliced to the transcript's own region so composer and
    status chrome cannot mask a transcript that moved.
    """
    strips = list(app.screen._compositor.render_strips())
    region = view.region
    top, bottom = max(0, region.y), min(len(strips), region.y + region.height)
    return ["".join(seg.text for seg in strips[y]._segments) for y in range(top, bottom)]


def _geometry(app: OperatorApp, view: TranscriptView) -> dict[str, Any]:
    blocks = view.blocks()
    notices = [b.text() for b in blocks if isinstance(b, NoticeBlock)]
    return {
        "blocks": len(blocks),
        "virtual_h": view.virtual_size.height,
        "container_h": view.container_size.height,
        "size_h": view.size.height,
        "max_scroll_y": view.max_scroll_y,
        "scroll_y": view.scroll_y,
        "scrollbar": bool(view.show_vertical_scrollbar),
        "scrollable": view.virtual_size.height > view.container_size.height,
        "pending_head": len(app._resume_pending_head),
        "notices": notices,
    }


async def _boot(history: list[Any], size: tuple[int, int]):
    session = FakeSession()
    session._history = history
    app = OperatorApp(lambda: _factory(session))
    return app, session


async def _settle(pilot, app: OperatorApp, view_holder: dict[str, Any]) -> TranscriptView:
    for _ in range(80):
        await pilot.pause()
        view = app.query_one(TranscriptView)
        if view.blocks():
            view_holder["view"] = view
            for _ in range(10):
                await pilot.pause()
            return view
    raise AssertionError("resume never painted")


async def run_geometry(shape: str, size: tuple[int, int]) -> None:
    history = SHAPES[shape]()
    app, _session = await _boot(history, size)
    async with app.run_test(size=size) as pilot:
        view = await _settle(pilot, app, {})
        geo = _geometry(app, view)
        geo["shape"] = shape
        geo["history"] = len(history)
        geo["grid"] = f"{size[0]}x{size[1]}"
        print(json.dumps(geo, indent=2))


async def run_jitter(shape: str, size: tuple[int, int]) -> None:
    """Record the reader's ANCHOR GAP on every offset the viewport passes.

    The jitter is not visible in a compositor strip diff, because the rows the
    transcript paints between the mount and the restore are the SAME rows —
    what changes is where they sit. The measurable quantity is therefore the
    anchor gap: the distance from the top of the block the reader is looking at
    to the top of the viewport (``anchor.virtual_region.y - scroll_y``). It is
    invariant under a correct insert above the viewport, because the mount adds
    the same amount to the block's absolute position and to the offset — and it
    JUMPS by the inserted extent on any frame where only one of the two moved.

    Sampled from the widget's own ``scroll_y`` watch and its extent hook, so
    every offset the viewport actually rests at is recorded, not just the ones
    a ``pause`` loop happens to land on.
    """
    history = SHAPES[shape]()
    app, _session = await _boot(history, size)
    async with app.run_test(size=size) as pilot:
        view = await _settle(pilot, app, {})
        if not app._resume_pending_head:
            print(json.dumps({"error": "no deferred head to page", "shape": shape}))
            return
        # Park the reader just below the trigger so the notch below lands
        # INSIDE the trigger zone and earns a real page.
        # `note_user_scroll` FIRST: a bare `scroll_to` leaves the tail anchor
        # following, and `_size_updated` then drags the viewport back to the
        # end the moment the mount grows the extent — which measures the tail
        # follow, not the insert. A reader who scrolled up has released it.
        view.note_user_scroll()
        # Just below the trigger row, measured from the CURRENT extent rather
        # than assumed to be near 0: the fill now makes the first frame
        # scrollable and lands the reader on the tail, so a hard-coded `y=6`
        # parks them wherever the tail happens to be and the notch below earns
        # nothing.
        view.scroll_to(y=RESUME_PAGE_TRIGGER_ROWS + 2, animate=False)
        for _ in range(8):
            await pilot.pause()

        # The block under the reader's eyes, identified BEFORE the mount and
        # tracked by identity: its absolute position moves with the insert,
        # and that movement is exactly what the offset must match.
        anchor = next(
            (b for b in view.blocks() if b.virtual_region.bottom > view.scroll_y),
            None,
        )
        if anchor is None:
            print(json.dumps({"error": "no anchor block", "shape": shape}))
            return

        samples: list[dict[str, Any]] = []

        def sample(tag: str) -> None:
            samples.append(
                {
                    "tag": tag,
                    "scroll_y": round(float(view.scroll_y), 2),
                    "anchor_y": int(anchor.virtual_region.y),
                    "gap": round(float(anchor.virtual_region.y - view.scroll_y), 2),
                    "virtual_h": view.virtual_size.height,
                }
            )

        # A REAL GESTURE, not `app._mount_older_resume_page()`.
        #
        # Calling the mount directly bypasses the three layers that sit between
        # a reader and a page — the page-back latch (`_resume_in_zone`), the
        # single-flight requeue, and the scroll animation — and every defect
        # this probe was written to catch lived in exactly that bypassed layer:
        # the fill spending the reader's latch, the cap that could not iterate,
        # and the resume landing at the top instead of the tail. Measurements
        # taken through the direct call were CORRECT and still could not see
        # any of it — a blind spot in the instrument, not in the reading. So
        # the page is earned the way a user earns it.
        pending_before = len(app._resume_pending_head)
        _wheel_to_trigger(view)

        # The baseline is taken AFTER the notch's own travel has landed, and
        # BEFORE the page it earned has mounted. A wheel notch legitimately
        # moves the reader by its own delta — that is the scroll they asked
        # for — and the invariant under test is that the MOUNT adds no motion
        # on top of it. Sampling before the notch folds the gesture's own rows
        # into the verdict and reports a correct insert as displaced.
        for _ in range(2):
            await pilot.pause()
        sample("baseline")
        baseline_gap = samples[0]["gap"]

        original_watch = view.watch_scroll_y
        original_size = view._size_updated
        screen = app.screen
        original_refresh = screen._compositor_refresh

        def watch_scroll_y(old: float, new: float) -> None:
            original_watch(old, new)
            sample("scroll")

        def size_updated(*args: Any, **kwargs: Any) -> bool:
            changed = original_size(*args, **kwargs)
            if changed:
                sample("extent")
            return changed

        def compositor_refresh() -> None:
            # THE moment the terminal is written. Sampled here rather than
            # after a `pause`, because a pause drains the whole callback queue
            # and coalesces the entire settle into one observation — which
            # reports every intermediate paint as if it never happened.
            sample("paint")
            original_refresh()

        view.watch_scroll_y = watch_scroll_y  # type: ignore[method-assign]
        view._size_updated = size_updated  # type: ignore[method-assign]
        screen._compositor_refresh = compositor_refresh  # type: ignore[method-assign]
        try:
            for _ in range(16):
                await pilot.pause()
                sample("frame")
        finally:
            view.watch_scroll_y = original_watch  # type: ignore[method-assign]
            view._size_updated = original_size  # type: ignore[method-assign]
            screen._compositor_refresh = original_refresh  # type: ignore[method-assign]

        sample("settled")
        pages_mounted = (pending_before - len(app._resume_pending_head)) // RESUME_PAGE_MESSAGES

        # PAINTED frames are the ones that matter: a `frame` sample is taken
        # after a `pause`, so the terminal has been written. `extent` samples
        # are taken INSIDE `_size_updated`, before the same frame's re-anchor
        # has run — they show the growth arriving, not a frame the reader saw,
        # so they are reported separately rather than mixed into the verdict.
        painted = [s for s in samples if s["tag"] in ("baseline", "paint", "frame", "settled")]
        painted_excursion = max(abs(s["gap"] - baseline_gap) for s in painted) if painted else 0
        displaced = [s for s in painted if abs(s["gap"] - baseline_gap) > 0.5]
        print(
            json.dumps(
                {
                    "shape": shape,
                    "grid": f"{size[0]}x{size[1]}",
                    "baseline_gap": baseline_gap,
                    "samples": samples,
                    # Proof the gesture actually earned a page: a run where
                    # nothing mounted measures an insert that never happened
                    # and reports a flat, meaningless zero.
                    "pages_mounted": pages_mounted,
                    "painted_frames": len(painted),
                    "displaced_painted_frames": len(displaced),
                    "painted_max_gap_excursion": painted_excursion,
                    "all_sample_max_gap_excursion": (
                        max(abs(s["gap"] - baseline_gap) for s in samples) if samples else 0
                    ),
                    "settled_gap": samples[-1]["gap"],
                },
                indent=2,
            )
        )


async def run_reader(shape: str, size: tuple[int, int]) -> None:
    """Print the block the reader is looking at, on every painted frame.

    The human-legible form of the invariant. ``jitter`` measures the anchor gap
    numerically; this names the row, which is what the reader actually notices
    when it moves — and it needs no interpretation to read as right or wrong.
    """
    history = SHAPES[shape]()
    app, _session = await _boot(history, size)
    async with app.run_test(size=size) as pilot:
        view = await _settle(pilot, app, {})
        if not app._resume_pending_head:
            print(json.dumps({"error": "no deferred head to page", "shape": shape}))
            return
        # Mid-transcript, not in the trigger zone: the zone is itself a gesture
        # that would mount its own page and confound the baseline.
        view.note_user_scroll()
        view.scroll_to(y=max(8.0, view.max_scroll_y / 2), animate=False)
        for _ in range(20):
            await pilot.pause()

        anchor = next((b for b in view.blocks() if b.virtual_region.bottom > view.scroll_y), None)
        if anchor is None:
            print(json.dumps({"error": "no anchor block", "shape": shape}))
            return

        def top_text() -> str:
            top = next((b for b in view.blocks() if b.virtual_region.bottom > view.scroll_y), None)
            text = getattr(top, "text", None)
            if not callable(text):
                return ""
            return str(text())[:44]

        baseline = top_text()
        frames: list[dict[str, Any]] = []
        screen = app.screen
        original_refresh = screen._compositor_refresh

        def compositor_refresh() -> None:
            original_refresh()
            frames.append(
                {
                    "paint": len(frames),
                    "scroll_y": round(float(view.scroll_y), 1),
                    "virtual_h": view.virtual_size.height,
                    "top_row": top_text(),
                    "holds": top_text() == baseline,
                }
            )

        screen._compositor_refresh = compositor_refresh  # type: ignore[method-assign]
        try:
            # A real gesture, for the reason documented in `run_jitter`: the
            # direct mount skips the latch, the requeue and the animation.
            _wheel_to_trigger(view)
            for _ in range(16):
                await pilot.pause()
        finally:
            screen._compositor_refresh = original_refresh  # type: ignore[method-assign]

        print(
            json.dumps(
                {
                    "shape": shape,
                    "grid": f"{size[0]}x{size[1]}",
                    "baseline_top_row": baseline,
                    "frames": frames,
                    "frames_that_moved": sum(1 for f in frames if not f["holds"]),
                },
                indent=2,
            )
        )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "geometry"
    shape = sys.argv[2] if len(sys.argv) > 2 else "agentic"
    grid = sys.argv[3] if len(sys.argv) > 3 else "120x40"
    columns, rows = (int(n) for n in grid.split("x"))
    if mode == "geometry":
        asyncio.run(run_geometry(shape, (columns, rows)))
    elif mode == "jitter":
        asyncio.run(run_jitter(shape, (columns, rows)))
    elif mode == "reader":
        asyncio.run(run_reader(shape, (columns, rows)))
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
