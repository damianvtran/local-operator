"""Characterise Esc-to-stop latency for the word-caret escape coalescing.

Issue #370's fix has to hold an `escape` briefly to see whether an arrow
follows it. The question this answers is what that hold COSTS the app's stop
key, because Esc-to-stop going perceptibly slower would be a bad trade for a
caret movement.

The answer is that it costs one message-pump turn, because the parser has
already resolved the timing ambiguity (see the escape-coalescing block in
`editor.py`). This measures that under a deliberately loaded event loop, and
compares the three candidate deferral primitives so the choice is evidence
rather than preference.

    cd <worktree>
    PYTHONPATH=$PWD env -u NO_COLOR TERM=xterm-256color \
        ~/local-operator/.venv/bin/python scripts/word_caret_latency.py
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual import events  # noqa: E402
from textual._xterm_parser import XTermParser  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

TRIALS = 25


async def _hog() -> None:
    """A busy loop, so latency is measured against a contended pump."""
    try:
        while True:
            sum(range(300_000))
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        pass


async def _measure_deferral(mechanism: str, trials: int = 40) -> list[float]:
    """Return schedule-to-callback latencies (ms) for one deferral primitive.

    Measured by awaiting the callback directly rather than polling with
    ``pilot.pause()``, because a polling loop's own sleep interval dominates
    the number being measured and hides the difference between the two.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    latencies: list[float] = []
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        hog = asyncio.create_task(_hog())
        for _ in range(trials):
            done = asyncio.Event()
            start = time.perf_counter()
            captured: list[float] = []

            def _fire(_c=captured, _d=done, _s=start) -> None:
                _c.append((time.perf_counter() - _s) * 1000)
                _d.set()

            if mechanism == "call_later":
                editor.call_later(_fire)
            else:
                editor.call_after_refresh(_fire)
            try:
                await asyncio.wait_for(done.wait(), 2)
            except asyncio.TimeoutError:  # pragma: no cover - defensive
                pass
            latencies.extend(captured)

        hog.cancel()
        try:
            await hog
        except asyncio.CancelledError:
            pass
    return latencies


async def _measure(mechanism: str) -> tuple[int, float, float]:
    """Return (misordered_trials, max_ms, mean_ms) for one deferral primitive."""
    order: list[str] = []
    original = Editor._on_key

    async def _spy(self, event):  # type: ignore[no-untyped-def]
        order.append(event.key)
        return await original(self, event)

    Editor._on_key = _spy  # type: ignore[method-assign]
    try:
        app = OperatorApp(lambda: _factory(FakeSession()))
        misordered = 0
        latencies: list[float] = []
        async with app.run_test(size=(100, 24)) as pilot:
            for _ in range(200):
                if app._session is not None:
                    break
                await pilot.pause()
                await asyncio.sleep(0.01)
            editor = app.query_one(Editor)
            editor.focus()
            editor.text = "alpha beta gamma delta"
            await pilot.pause()

            hog = asyncio.create_task(_hog())
            for _ in range(TRIALS):
                order.clear()
                fired: list[float] = []

                def _fire(_fired=fired) -> None:
                    _fired.append(time.perf_counter())

                def _defer(action, _m=mechanism) -> None:  # type: ignore[no-untyped-def]
                    if _m == "call_later":
                        editor.call_later(_fire)
                    else:
                        editor.call_after_refresh(_fire)

                editor._defer_escape = _defer  # type: ignore[method-assign]

                start = time.perf_counter()
                parser = XTermParser()
                for event in list(parser.feed("\x1b\x1b[D")) + list(parser.feed("")):
                    if isinstance(event, events.Key):
                        event.set_sender(app)
                        app._driver.send_message(event)  # type: ignore[union-attr]
                for _ in range(60):
                    await pilot.pause()
                    await asyncio.sleep(0.002)
                    if fired:
                        break

                if order[:2] != ["escape", "left"]:
                    misordered += 1
                if fired:
                    latencies.append((fired[0] - start) * 1000)

            hog.cancel()
            try:
                await hog
            except asyncio.CancelledError:
                pass
        return misordered, max(latencies), sum(latencies) / len(latencies)
    finally:
        Editor._on_key = original  # type: ignore[method-assign]


async def main() -> None:
    # The premise: the parser holds a lone Esc itself, and emits the chord's two
    # events from a single pass. This is why one pump turn is sufficient.
    assert list(XTermParser().feed("\x1b")) == []
    chord = [e.key for e in XTermParser().feed("\x1b\x1b[D") if isinstance(e, events.Key)]
    print("parser: feed('\\x1b') -> []   (held until ESCAPE_DELAY expires)")
    print(f"parser: feed('\\x1b\\x1b[D') -> {chord}   (one pass, queued back to back)")
    print()
    print("Ordering: does the queued arrow always overtake the deferred escape?")
    for mechanism in ("call_after_refresh", "call_later"):
        misordered, _, _ = await _measure(mechanism)
        print(f"  {mechanism:18}: misordered {misordered}/{TRIALS}")
    print()

    print("Esc-to-stop latency (schedule -> callback), loop under sustained load:")
    print("  BEFORE (wall-clock ESCAPE_DELAY window): 100.00 ms by construction")
    for mechanism in ("call_after_refresh", "call_later"):
        latencies = sorted(await _measure_deferral(mechanism))
        median = statistics.median(latencies)
        p95 = latencies[int(0.95 * len(latencies)) - 1]
        print(
            f"  AFTER  ({mechanism:18}): "
            f"median {median:6.2f} ms  p95 {p95:7.2f} ms  max {max(latencies):7.2f} ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
