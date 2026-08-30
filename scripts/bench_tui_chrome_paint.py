"""Count the invalidation a running subagent page costs per spinner tick.

Wall-clock is useless as a regression signal for this work. The machine that
motivated it was at loadavg 260-307 on 14 cores from unrelated harnesses, and a
fixed work quantum measured there held CPU time stable (81.6 -> 101.5 ms) while
WALL time inflated 4.6-7.8x. So this benchmark reports **call counts**, which
are load-invariant: how many ``messages.Layout`` the screen receives, how many
compositor reflows those cause, how many ``messages.Update`` are posted, and how
many times each chrome row is rewritten.

Those four numbers are the actual subject of the chrome-paint fixes. A tick that
posts no Layout cannot reflow the screen, and a row that is not rewritten cannot
emit escape sequences to the terminal, whatever the box happens to be doing.

Run from the checkout with its interpreter::

    env -u NO_COLOR TERM=xterm-256color \
      .venv/bin/python scripts/bench_tui_chrome_paint.py --json out.json

``--json`` writes the same figures as a machine-readable record so a before and
an after run can be diffed rather than eyeballed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textual import messages  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock  # noqa: E402
from scripts.benchmark_residual_child_lag import CanonicalFakeSession  # noqa: E402
from tests.unit.tui.test_app_pilot import _factory  # noqa: E402

#: Transcript depth behind the page. The reflow cost this measures is a
#: function of how many widgets the compositor re-arranges, so an empty
#: transcript would report a win that the reported session never sees.
BLOCKS = 160
#: Driven spinner ticks. Enough that the per-tick averages are stable and short
#: enough that the whole run finishes on a loaded box.
TICKS = 50


class Probe:
    """Counts screen invalidation without changing any of it.

    Every hook wraps and delegates: the app under measurement must behave
    exactly as it does unmeasured, or the counts describe the probe.
    """

    def __init__(self, app: OperatorApp) -> None:
        self.app = app
        self.counts: Counter[str] = Counter()
        self._install()

    def _install(self) -> None:
        screen = self.app.screen

        original_post = screen.post_message

        def post_message(message: Any) -> Any:
            # Layout is the expensive one and the point of the exercise: it
            # clears every ancestor's arrangement cache on the way in, then
            # makes the screen re-arrange and the compositor reflow.
            if isinstance(message, messages.Layout):
                self.counts["layout_msgs"] += 1
            elif isinstance(message, messages.Update):
                self.counts["update_msgs"] += 1
            return original_post(message)

        screen.post_message = post_message  # type: ignore[method-assign]

        original_reflow = screen._compositor.reflow

        def reflow(*args: Any, **kwargs: Any) -> Any:
            self.counts["reflow"] += 1
            return original_reflow(*args, **kwargs)

        screen._compositor.reflow = reflow  # type: ignore[method-assign]

        original_refresh_layout = screen._refresh_layout

        def refresh_layout(*args: Any, **kwargs: Any) -> Any:
            self.counts["refresh_layout"] += 1
            return original_refresh_layout(*args, **kwargs)

        screen._refresh_layout = refresh_layout  # type: ignore[method-assign]

    def watch_chrome(self, view: Any) -> None:
        """Count rewrites of each chrome row, and ticks, on the open page.

        Per-ROW rather than per-paint: the fixes turn one paint that rewrote
        three rows into one that rewrites the row the spinner actually moved,
        so a paint count alone would report no change at all.
        """
        for name, label in (
            ("_title", "title_updates"),
            ("_breadcrumb", "breadcrumb_updates"),
            ("_rule", "rule_updates"),
        ):
            widget = getattr(view, name)
            original = widget.update

            def update(*args: Any, _label: str = label, _original: Any = original, **kw: Any):
                self.counts[_label] += 1
                return _original(*args, **kw)

            widget.update = update

        original_paint = view._paint_chrome

        def paint_chrome(*args: Any, **kwargs: Any) -> Any:
            self.counts["paint_chrome"] += 1
            return original_paint(*args, **kwargs)

        view._paint_chrome = paint_chrome

        original_tick = view._tick

        def tick(*args: Any, **kwargs: Any) -> Any:
            self.counts["ticks"] += 1
            return original_tick(*args, **kwargs)

        view._tick = tick


async def run(ticks: int, blocks: int) -> dict[str, Any]:
    session = CanonicalFakeSession(running=True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is session:
                break
        view_transcript = app._transcript_view()
        with view_transcript.batch_append():
            for index in range(blocks):
                view_transcript.append_block(
                    NoticeBlock(f"retained event {index} " + "word " * 12, "info")
                )
        await pilot.pause()

        app._open_subagent_view("active-child")
        await pilot.pause()
        await pilot.pause()
        child = app._subagent_view
        if child is None:  # pragma: no cover - the fixture always opens
            raise RuntimeError("subagent view did not open")

        # Counting starts AFTER the page has settled, so the open's own layout
        # is not charged to the steady state this measures.
        probe = Probe(app)
        probe.watch_chrome(child)

        started = time.perf_counter()
        for _ in range(ticks):
            # The timer is driven directly rather than slept on: a 0.08 s
            # interval under a loaded box does not fire on schedule, and the
            # subject is the work per tick, not the scheduler's punctuality.
            child._tick()
            await pilot.pause()
        wall = time.perf_counter() - started

    counts = dict(probe.counts)
    per_tick = {f"{key}_per_tick": round(value / ticks, 4) for key, value in counts.items()}
    return {
        "ticks": ticks,
        "blocks": blocks,
        "counts": counts,
        "per_tick": per_tick,
        # Reported but deliberately NOT the headline: see the module docstring.
        "wall_s": round(wall, 3),
    }


async def run_focus(seconds: float) -> dict[str, Any]:
    """Bytes an idle session writes to its terminal, focused vs blurred.

    This one IS time-based, because the quantity is a RATE and there is no
    count that stands in for it — but it is a rate of BYTES, not of frames, so
    a loaded box makes it conservative (fewer timer firings) rather than
    noisy in the direction of the claim.

    The app is driven with the splash up and a turn running, which is the
    shape of a session sitting in a window the user has tabbed away from: the
    welcome pulse, the tip rotation and the working line's shimmer are exactly
    what a blurred terminal should stop paying for.
    """
    import os

    os.environ.pop("LOCAL_OPERATOR_NO_SHIMMER", None)

    from local_operator.tui.animation import reset_animation_focus
    from local_operator.tui.widgets.transcript import WorkingBlock

    session = CanonicalFakeSession(running=True)
    app = OperatorApp(lambda: _factory(session))
    written = {"n": 0}

    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        working = WorkingBlock("thinking")
        app._append_block(working)
        app._working_block = working
        await pilot.pause()

        # Counted at ``App._display``, which is where a compositor update is
        # handed over to be written, and the bytes are the update's OWN
        # rendered segments. `run_test` is headless, so `_display` returns
        # before touching the driver and a hook on `driver.write` counts
        # nothing at all — the update still describes exactly what a real
        # terminal would receive, which is the quantity in question. This is
        # the same seam `scripts/probe_spinner_invalidation.py` measures at.
        console = app.console
        original_display = app._display

        def display(screen_arg: Any, renderable: Any) -> None:
            if renderable is not None:
                try:
                    written["n"] += len(renderable.render_segments(console).encode())
                except Exception:
                    pass
            return original_display(screen_arg, renderable)

        app._display = display  # type: ignore[method-assign]

        reset_animation_focus()
        app._set_animation_focused(True)
        written["n"] = 0
        await asyncio.sleep(seconds)
        focused_bytes = written["n"]

        app._set_animation_focused(False)
        await pilot.pause()
        written["n"] = 0
        await asyncio.sleep(seconds)
        blurred_bytes = written["n"]

        reset_animation_focus()

    return {
        "window_s": seconds,
        "focused_bytes_per_s": round(focused_bytes / seconds),
        "blurred_bytes_per_s": round(blurred_bytes / seconds),
        "reduction": (round(focused_bytes / blurred_bytes, 1) if blurred_bytes else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write the record here")
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--blocks", type=int, default=BLOCKS)
    parser.add_argument(
        "--focus",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="also measure focused-vs-blurred terminal output over this window",
    )
    args = parser.parse_args()

    record = asyncio.run(run(args.ticks, args.blocks))
    counts = record["counts"]
    print(f"driven spinner ticks={record['ticks']} transcript_blocks={record['blocks']}")
    for key in (
        "layout_msgs",
        "reflow",
        "refresh_layout",
        "update_msgs",
        "paint_chrome",
        "title_updates",
        "breadcrumb_updates",
        "rule_updates",
    ):
        total = counts.get(key, 0)
        print(f"  {key:<20} total={total:<8} per_tick={total / record['ticks']:.3f}")
    print(f"  {'wall (not a signal)':<20} {record['wall_s']:.3f}s")
    if args.focus > 0:
        focus = asyncio.run(run_focus(args.focus))
        record["focus"] = focus
        print(f"\nterminal output, {focus['window_s']}s windows, splash + running turn")
        print(f"  focused  {focus['focused_bytes_per_s']:>8} bytes/s")
        print(f"  blurred  {focus['blurred_bytes_per_s']:>8} bytes/s")
        print(f"  reduction {focus['reduction']}x")
    if args.json is not None:
        args.json.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
