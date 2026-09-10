"""Capture the reported frame itself: a sidebar switch into a parked tool.

The ladder shot next to this one compares STATES; this one captures the actual
reported surface, through real owners over real sockets and the production
sidebar path, so the evidence is the frame the operator sees rather than a
reconstruction of it.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/wait_card_switch_shot.py OUTDIR [COLSxROWS]

Writes ``<outdir>/switch.svg`` (the moment after the switch, tool still
parked) and ``<outdir>/settled.svg`` (the same row once the tool returns) —
the pair that shows whether the row moved when its outcome arrived.

The tool is deliberately AGED on a real wall clock before the switch (see
``AGE_S``), because the band under the row keys its elapsed clock to the phase
and the phase changes at the moment the viewer arrives. Without a visible age
the band's number and the tool's true age are indistinguishable in the frame,
which is exactly how D6 survived round 1.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

# pytest's own fixture implementation, usable outside a test: `live_owners`
# patches through it, and borrowing the real class rather than a hand-rolled
# double means this script exercises the same patching (and the same undo) the
# test suite does.
from _pytest.monkeypatch import MonkeyPatch  # noqa: E402

from local_operator.tui.app import OperatorApp, ToolCard  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from tests.e2e.harness import wait_for_adoption  # noqa: E402
from tests.unit.harness.test_comms import DEADLOCK_GUARD_S, MAX_PUMP_TURNS  # noqa: E402
from tests.unit.tui.test_sidebar_live_tool_card import (  # noqa: E402
    _parking_tool,
    _switch,
    live_owners,
)

#: Seconds to let the tool genuinely execute before the viewer switches in.
#: Large enough that a clock restarted at the switch reads visibly differently
#: from the tool's real age — the whole point of the capture.
AGE_S = 12.0


async def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    size = (120, 24)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))

    import os
    import tempfile

    config = Path(tempfile.mkdtemp(prefix="wait-card-shot-")) / "config"
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(config)
    released = asyncio.Event()
    monkey = MonkeyPatch()
    try:
        async with live_owners(config, ["home00", "busy01"], _parking_tool(released), monkey) as r:
            app = OperatorApp(lambda: r("home00"), resume_factory=r)
            async with app.run_test(size=size) as pilot:
                await wait_for_adoption(app, pilot)
                app.query_one(Editor).cursor_blink = False
                app._set_sidebar_open(True)
                if app._sidebar_timer is not None:
                    app._sidebar_timer.pause()

                # Visit while idle, leave: what files the unanswered call into
                # the hidden viewer's live history, which is the replay this
                # whole surface is about.
                await _switch(app, "busy01", pilot)
                await _switch(app, "home00", pilot)

                owner = r.servers["busy01"]._handle._session  # type: ignore[attr-defined]
                turn = asyncio.create_task(owner.prompt("await the fix"))
                for _ in range(MAX_PUMP_TURNS):
                    await pilot.pause()
                    state = owner.frontend_state
                    if state.streaming and any(
                        e.get("type") == "tool_execution_start" for e in state.live_events
                    ):
                        break
                # AGE the call on a real clock while the viewer is elsewhere.
                # `pilot.pause()` alone yields without advancing wall time, so
                # the tool would be milliseconds old at the switch and a band
                # clock counting from the wrong zero would look correct.
                aged_from = time.monotonic()
                while time.monotonic() - aged_from < AGE_S:
                    await pilot.pause()
                    await asyncio.sleep(0.05)

                await _switch(app, "busy01", pilot)
                app._set_sidebar_open(False)
                await pilot.pause()
                real_age = time.monotonic() - aged_from
                save_capture(app, outdir / "switch.svg")
                cards = [b for b in app._transcript_view().blocks() if isinstance(b, ToolCard)]
                print("switch  :", cards[0]._state, [t for t, _s in cards[0]._status_runs()])
                # The D6 evidence: the tool's REAL age beside what the band says
                # about it. A number here that is not `real_age` is the finding.
                band = app._working_block
                # Asserted, not assumed: a switch that left no band mounted
                # would otherwise print `None` and read as "no clock shown",
                # i.e. as the fix working.
                assert band is not None, "no working line is mounted after the switch"
                print(f"tool real age    : {real_age:.1f}s")
                print(f"band label       : {band._activity!r} (phase {band._phase!r})")
                print(f"band clock shown : {band._clock!r}")
                print(f"card dates itself: {cards[0].started_at is not None}")

                released.set()
                for _ in range(MAX_PUMP_TURNS):
                    await pilot.pause()
                    if cards[0]._state not in ("running", "waiting"):
                        break
                for _ in range(20):
                    await pilot.pause()
                save_capture(app, outdir / "settled.svg")
                print("settled :", cards[0]._state, [t for t, _s in cards[0]._status_runs()])
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(turn, DEADLOCK_GUARD_S)
    finally:
        released.set()
        # Same discipline the fixture keeps: this script patches module globals,
        # so it puts them back rather than leaving a mutated interpreter.
        monkey.undo()


asyncio.run(main())
