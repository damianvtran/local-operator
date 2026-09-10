"""Capture the two states design round 3 raised against the band's clock.

The switch shot beside this one captures the ADOPTED row; this one captures the
two edges that suppression opened, because neither is visible in that frame:

``blurred.svg``/``blurred-next.svg``
    A row whose phase cannot date itself, on a BLURRED terminal — the state
    where D26 pins the head glyph and the withheld clock leaves nothing else
    that can move (design round 3, D8). Captured as a consecutive PAIR, since
    the finding is that the row does not change: one still cannot show motion,
    and two identical stills are the defect itself.

``shrinking.svg``
    A batch that has shed the calls it was named for, leaving one young
    survivor (design round 3, D9). The frame carries the card's own receipt and
    the band together, which is what makes the number checkable by eye: the
    band must not claim an age the card above it contradicts.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/wait_card_clock_shot.py OUTDIR [COLSxROWS]

Both states are aged on a REAL wall clock. ``pilot.pause()`` yields without
advancing time, so a batch aged with it alone is milliseconds old and a clock
counted from the wrong zero looks correct — which is how D6 survived round 1.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from textual.pilot import Pilot  # noqa: E402

from local_operator.harness.types import (  # noqa: E402
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.tui import animation  # noqa: E402
from local_operator.tui.app import OperatorApp, ToolCard  # noqa: E402
from local_operator.tui.events import ToolEnded, ToolStarted, TurnStarted  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.transcript import WorkingBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: Seconds to age the batch before the frame. Large enough that a clock counted
#: from the wrong zero differs VISIBLY from the right one at the row's own
#: resolution — the number is what the frame is for.
AGE_S = 14.0


def _started(call_id: str, name: str, **args: object) -> ToolStarted:
    return ToolStarted(ToolExecutionStartEvent(tool_call_id=call_id, tool_name=name, args=args))


def _ended(call_id: str, name: str) -> ToolEnded:
    return ToolEnded(
        ToolExecutionEndEvent(
            tool_call_id=call_id,
            tool_name=name,
            result=ToolResult(tool_call_id=call_id, tool_name=name, content=[]),
        )
    )


async def _age(pilot: Pilot[None], seconds: float) -> None:
    """Advance REAL time while the app keeps painting."""
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        await pilot.pause()
        await asyncio.sleep(0.05)


def _band(app: OperatorApp) -> WorkingBlock:
    band = app._working_block
    # Asserted rather than assumed: a missing band prints `None` for the clock,
    # which reads as "no wrong number shown", i.e. as the fix working.
    assert band is not None, "no working line is mounted"
    return band


async def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    size = (120, 24)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))

    # ---- D8: a clockless row on a blurred terminal ------------------------
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app.query_one(Editor).cursor_blink = False
        # THE SHIMMER PIN MUST COME OFF for this capture, and only for this
        # one. `isolate_capture()` sets LOCAL_OPERATOR_NO_SHIMMER=1 so exported
        # frames are reproducible — but that is the OTHER still path, the one
        # D26 deliberately freezes. Leaving it on captures a silenced row whose
        # pinned head is correct, and would read as the D8 defect reproducing
        # after it was fixed. The gate under test is FOCUS.
        os.environ.pop("LOCAL_OPERATOR_NO_SHIMMER", None)
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "await_job", job_id="7a73c97ffc54"))
        await pilot.pause()
        card = [b for b in app._transcript_view().blocks() if isinstance(b, ToolCard)][0]
        # ADOPTED: exactly what a sidebar switch leaves behind. The card clears
        # its own zero, so the phase cannot be dated and the band withholds.
        card.restore(state="running")
        app._refresh_working_activity()
        await _age(pilot, 2.0)

        app._set_animation_focused(False)
        await pilot.pause()
        band = _band(app)
        print("== D8: blurred, clockless row ==")
        print(f"motion_enabled()  : {animation.motion_enabled()}")
        print(f"band label        : {band._activity!r}")
        print(f"band clock        : {band._clock!r}")

        heads: list[str] = []
        save_capture(app, outdir / "blurred.svg")
        heads.append(band._SPINNER[band._still_head])
        # A CONSECUTIVE frame one blurred tick later. The pair is the evidence:
        # the finding is a row that does not change, which one still cannot show.
        await _age(pilot, 1.2)
        save_capture(app, outdir / "blurred-next.svg")
        heads.append(band._SPINNER[band._still_head])
        await _age(pilot, 1.2)
        save_capture(app, outdir / "blurred-next2.svg")
        heads.append(band._SPINNER[band._still_head])
        print(f"heads across 3 frames: {heads}  distinct={len(set(heads))}")
        print(f"clock still withheld : {band._clock!r}")
        app._set_animation_focused(True)
        # Restore the pin for the D9 capture below, which wants the ordinary
        # reproducible frame.
        os.environ["LOCAL_OPERATOR_NO_SHIMMER"] = "1"

    # ---- D9: the band beside a batch that shrank --------------------------
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app.query_one(Editor).cursor_blink = False
        app.post_message(TurnStarted())
        app.post_message(_started("old", "await_job", job_id="7a73c97ffc54"))
        await pilot.pause()
        # AGE the first call for real, so the batch's zero is genuinely old.
        await _age(pilot, AGE_S)

        # The owner refills a slot: `max_parallel_tools` is 8 and the worker
        # refills as slots free, so a batch over eight calls starts its tail
        # after the viewer arrived. This is that tail.
        app.post_message(_started("new", "read", path="local_operator/tui/app.py"))
        await pilot.pause()
        app._refresh_working_activity()
        band = _band(app)
        print()
        print("== D9: batch shrinking to a young survivor ==")
        print(f"2 running   : {band._activity!r}  clock={band._clock!r}")

        # The old call settles. The phase never leaves `running`, so the zero is
        # not re-derived by the phase machinery — the label narrows to `read`.
        app.post_message(_ended("old", "await_job"))
        await pilot.pause()
        app._refresh_working_activity()
        # Let the survivor age WELL past the row's one-second resolution before
        # the frame. At a sub-second age the band and the card can legitimately
        # round to different integers, and a frame whose two numbers differ by
        # one is unreadable as evidence either way \u2014 the finding was a gap of
        # fourteen seconds, so the capture has to make the agreement obvious.
        shed_age = 4.0
        await _age(pilot, shed_age)
        app._refresh_working_activity()
        band = _band(app)
        cards = [b for b in app._transcript_view().blocks() if isinstance(b, ToolCard)]
        survivor = [c for c in cards if c.tool_name == "read"][0]
        real = time.monotonic() - (survivor.started_at or time.monotonic())
        # Paint both rows at the SAME instant before exporting. The card and the
        # band run independent one-second tickers, so a frame sampled between
        # their ticks can show two integers a second apart for one underlying
        # value \u2014 a paint-cadence artifact that is present on base too, and that
        # would muddy a frame whose whole subject is whether the two numbers
        # agree. This forces the state the next tick reaches anyway; it does not
        # change either clock's zero.
        band._paint()
        survivor._refresh_row()
        await pilot.pause()
        save_capture(app, outdir / "shrinking.svg")
        print(f"1 survivor  : {band._activity!r}  clock={band._clock!r}")
        print(f"survivor real age    : {real:.1f}s")
        print(f"batch age (old zero) : {AGE_S + shed_age:.1f}s   <- what the band used to say")
        print(f"overstatement        : {AGE_S + shed_age - real:.1f}s")


asyncio.run(main())
