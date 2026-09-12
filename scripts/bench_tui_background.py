"""Measure input injection -> compositor-painted text under synthetic child load.

Run with the worktree interpreter, e.g. ``.venv/bin/python
scripts/bench_tui_background.py --children 0 10 50 --output /tmp/baseline.json``.
No provider or retained session is used. Wall latency is an observation, NEVER
an assertion suitable for CI: the host can deschedule this process. CPU gaps,
operation counts and positive/negative canaries distinguish that from loop work.
The paint boundary is Textual's actual compositor update handed to _display,
not pilot.press/pause and not the physical terminal's presentation latency.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import string
import sys
import threading
import time
from pathlib import Path
from typing import Any

_source_parser = argparse.ArgumentParser(add_help=False)
_source_parser.add_argument("--source-root", default=str(Path(__file__).resolve().parents[1]))
_source_args, _ = _source_parser.parse_known_args()
# The small baseline archive may omit test helpers. Fixtures belong to this
# benchmark's checkout; application imports below MUST resolve the selected
# source root (asserted in the provenance record).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests  # noqa: E402,F401

sys.path.insert(0, str(Path(_source_args.source_root).resolve()))
import scripts.probe_isolation  # noqa: E402,F401

# Isolation normally disables shimmer for stills. The timing workload must keep
# production animation enabled, or the measurement removes the work it studies.
os.environ.pop("LOCAL_OPERATOR_NO_SHIMMER", None)

from textual._compositor import ChopsUpdate, LayoutUpdate  # noqa: E402
from textual.events import Key, MouseScrollUp  # noqa: E402

from local_operator.harness.jobs import AsyncJob, AsyncJobManager  # noqa: E402
from local_operator.harness.types import SubagentProgressEvent  # noqa: E402
from local_operator.session.attached import AttachedSession  # noqa: E402
from local_operator.session.frontend_state import (  # noqa: E402
    FrontendSessionState,
    FrontendStateStore,
    FrontendUpdate,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import SubagentProgress  # noqa: E402
from local_operator.tui.widgets import subagent_panel  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock  # noqa: E402
from scripts.benchmark_residual_child_lag import Timings, _events  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


def summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "n": len(values),
        "p50_ms": statistics.median(values) * 1000 if values else 0,
        "p95_ms": ordered[int((len(values) - 1) * 0.95)] * 1000 if values else 0,
        "max_ms": max(values, default=0) * 1000,
    }


class LoadSession(FakeSession):
    """Real job ledger, state projection and canonical/raw event fanout."""

    def __init__(self, children: int, *, attached: bool = False) -> None:
        super().__init__()
        self.jobs = AsyncJobManager(on_job_change=self.changed)
        self._source_jobs = self.jobs
        self.pending = False
        self.delivered = 0
        self.edges_sent = 0
        self.raw: Any = None
        self._frontend_state_store: FrontendStateStore = FrontendStateStore(
            FrontendSessionState(session_id=self.session_id, epoch="benchmark")
        )
        for i in range(children):
            self.jobs._jobs[f"benchmark-child-{i}"] = AsyncJob(
                id=f"benchmark-child-{i}",
                type="task",
                label=f"Synthetic coder {i}",
                status="running",
                start_time=time.time() - 60,
                started_at=time.time() - 60,
                trajectory=_events(100),
                agent_role="coder",
                prompt="Synthetic load only",
            )
        self._frontend_state_store.refresh_jobs(self)
        self.viewer = None
        if attached:
            self.owns_runtime = False
            self.outcome_is_synchronous = False
            self.runtime_locality = "this-machine"

            async def never():
                raise AssertionError("benchmark must not take over a live runtime")

            self.viewer = AttachedSession(
                config_dir=Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]),
                session_id=self.session_id,
                takeover_factory=never,
            )
            self.viewer._install_frontend(self._frontend_state_store.state)
            self.jobs = self.viewer.jobs
            self._subagent_comms = self.viewer._subagent_comms
            assert self.viewer._frontend_store is not None
            self._frontend_state_store = self.viewer._frontend_store

    @property
    def session_id(self) -> str:
        return "benchmark-parent"

    @property
    def frontend_state(self):
        return self._frontend_state_store.state

    def subscribe_frontend(self, callback):
        return self._frontend_state_store.subscribe(callback)

    def refresh_frontend_usage(self) -> None:
        pass

    def changed(self) -> None:
        if not self.pending:
            self.pending = True
            asyncio.get_running_loop().call_later(0.05, self.flush)

    def flush(self) -> None:
        if self.viewer is not None:
            return
        self.pending = False
        self._frontend_state_store.refresh_jobs(self)

    def edge(self, index: int) -> None:
        self.edges_sent += 1
        if self.viewer is not None:
            # Real AttachedSession ingestion, including facade refresh and local
            # subscriber fanout. The socket codec has already produced this dict
            # at the production callback boundary; no I/O or models are faked here.
            for _ in range(max(1, len(self.jobs.list()))):
                update = FrontendUpdate(
                    epoch="benchmark",
                    sequence=self._frontend_state_store._state.sequence + 1,
                    changes={"activity_phase": f"edge-{index}-{self.delivered}"},
                )
                self.viewer._on_frontend_update(update.model_dump(mode="json"))
                self.delivered += 1
            return
        for job in self.jobs.list():
            event = SubagentProgressEvent(job_id=job.id, label=job.label, progress=f"edge-{index}")
            self._source_jobs._progress_fn(job.id)(event.progress)
            self._frontend_state_store.observe_event(self, event)
            if self.raw is not None:
                self.raw(event)
            self.delivered += 1


class PaintProbe:
    """Observe only rendered updates; never force an extra render to find text."""

    def __init__(self, app: OperatorApp) -> None:
        self.pending: list[tuple[str, float]] = []
        self.samples: list[float] = []
        self.frames = 0
        self.scroll_pending: list[tuple[float, float]] = []
        self.scroll_samples: list[float] = []
        self.scroll_frames: list[float] = []
        original = app._display

        def display(screen, renderable) -> None:
            original(screen, renderable)
            if renderable is None:
                return
            # Consume the exact composed update, not widget.text (which can be
            # current while the screen still contains the previous character).
            region = app._editor().region
            lines = []
            if isinstance(renderable, LayoutUpdate):
                for y, strips in enumerate(renderable.strips, renderable.region.y):
                    if region.y <= y < region.bottom:
                        lines.append("".join(strip.text for strip in strips))
            elif isinstance(renderable, ChopsUpdate):
                for y in sorted({span[0] for span in renderable.spans}):
                    if region.y <= y < region.bottom:
                        lines.append(
                            "".join(
                                strip.text
                                for strip in renderable.chops[y].values()
                                if strip is not None
                            )
                        )
            else:
                raise AssertionError(f"unrecognized compositor update: {type(renderable)}")
            text = "".join(lines).replace(" ", "")
            self.frames += 1
            now = time.perf_counter()
            offset = float(app._transcript_view().scroll_y)
            self.scroll_frames.append(offset)
            for previous, started in self.scroll_pending[:]:
                if offset < previous:
                    self.scroll_samples.append(now - started)
                    self.scroll_pending.remove((previous, started))
            for marker, started in self.pending[:]:
                if marker in text:
                    self.samples.append(now - started)
                    self.pending.remove((marker, started))

        app._display = display

    def inject(self, app: OperatorApp, prefix: str, char: str) -> None:
        self.pending.append((prefix[-8:], time.perf_counter()))
        app.post_message(Key(char, char))


async def loop_probe(
    stop: asyncio.Event,
    wall: list[float],
    cpu: list[float],
    ready: asyncio.Event | None = None,
) -> None:
    previous = time.perf_counter(), time.thread_time()
    if ready is not None:
        ready.set()
    while not stop.is_set():
        await asyncio.sleep(0.005)
        now = time.perf_counter(), time.thread_time()
        wall.append(now[0] - previous[0])
        cpu.append(now[1] - previous[1])
        previous = now


async def loop_canaries() -> dict[str, Any]:
    readings = {}
    for mode in ("cpu", "blocking_io", "idle"):
        wall: list[float] = []
        cpu: list[float] = []
        ready, stop = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(loop_probe(stop, wall, cpu, ready))
        await ready.wait()
        if mode == "cpu":
            started = time.thread_time()
            while time.thread_time() - started < 0.025:
                pass
        elif mode == "blocking_io":
            time.sleep(0.06)
        await asyncio.sleep(0.02)
        stop.set()
        await task
        readings[mode] = {"wall": summary(wall), "cpu": summary(cpu)}
    assert readings["cpu"]["cpu"]["max_ms"] >= 20
    assert readings["blocking_io"]["wall"]["max_ms"] >= 50
    assert readings["blocking_io"]["cpu"]["max_ms"] < readings["cpu"]["cpu"]["max_ms"]
    assert readings["idle"]["cpu"]["max_ms"] < readings["cpu"]["cpu"]["max_ms"]
    return readings


async def scenario(children: int, args: argparse.Namespace) -> dict[str, Any]:
    session = LoadSession(children, attached=args.attached)
    app = OperatorApp(lambda: _factory(session))
    timings = Timings()
    loop_thread = threading.get_ident()
    stats_threads: list[int] = []
    original_stats = subagent_panel.job_stats

    def slow_stats(*a: Any, **kw: Any):
        stats_threads.append(threading.get_ident())
        time.sleep(args.slow_stats_ms / 1000)
        return original_stats(*a, **kw)

    if args.slow_stats_ms:
        subagent_panel.job_stats = slow_stats
    wall_gaps: list[float] = []
    cpu_gaps: list[float] = []
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is session:
                break
        assert app._session is session
        transcript = app._transcript_view()
        with transcript.batch_append():
            for i in range(args.blocks):
                transcript.append_block(NoticeBlock(f"Retained event {i}: " + "word " * 12))
        await pilot.pause()
        app._editor().focus()
        if args.scroll:
            transcript.scroll_to(y=transcript.max_scroll_y / 2, immediate=True, animate=False)
            await pilot.pause()
        scroll_start = float(transcript.scroll_y)
        session.raw = lambda e: app.on_subagent_progress(
            SubagentProgress(e.job_id, e.label, e.progress)
        )
        for owner, method, name in [
            (session._frontend_state_store, "refresh_jobs", "store.refresh_jobs"),
            (session._frontend_state_store, "apply_update", "store.apply_update"),
            (app, "_apply_frontend_state", "app.apply_frontend_state"),
            (app, "_refresh_band", "app.refresh_band"),
        ]:
            timings.wrap(owner, method, name)
        paint = PaintProbe(app)
        stop = asyncio.Event()

        finished_keys = threading.Event()

        async def stream() -> None:
            index = 0
            # Keep offering load until EVERY key has been injected. Otherwise
            # the fast build spends most input samples idle after it finishes
            # the same fixed edge count, manufacturing an apparent latency win.
            while index < args.edges or not finished_keys.is_set():
                await asyncio.sleep(0.05)
                session.edge(index)
                index += 1

        probe_task = asyncio.create_task(loop_probe(stop, wall_gaps, cpu_gaps))
        stream_task = asyncio.create_task(stream())
        # A separate OS thread injects at its own cadence: a blocked event loop
        # must not postpone the START of the measurement and hide its own stall.
        typed = "".join(random.Random(42).choices(string.ascii_letters, k=args.samples))
        started = time.perf_counter()

        def keys() -> None:
            prefix = ""
            for char in typed:
                time.sleep(0.08)
                prefix += char
                paint.inject(app, prefix, char)
                if args.scroll and len(prefix) % 10 == 0:
                    paint.scroll_pending.append((float(transcript.scroll_y), time.perf_counter()))
                    transcript.post_message(
                        MouseScrollUp(transcript, 4, 4, 0, -1, 0, False, False, False)
                    )
            finished_keys.set()

        thread = threading.Thread(target=keys, daemon=True)
        thread.start()
        await asyncio.to_thread(thread.join)
        await stream_task
        session.flush()
        await pilot.pause()
        stop.set()
        await probe_task
        elapsed = time.perf_counter() - started
        assert app._editor().text == typed, app._editor().text
        assert len(paint.samples) == args.samples, (len(paint.samples), paint.pending)
        assert (
            session.delivered
            == (max(1, children) if args.attached else children) * session.edges_sent
        )
        if not args.attached:
            assert all(
                j.latest_details == {"progress": f"edge-{session.edges_sent - 1}"}
                for j in session.jobs.list()
            )
        else:
            assert (
                session.frontend_state.activity_phase
                == f"edge-{session.edges_sent - 1}-{session.delivered - 1}"
            )
        if args.scroll:
            assert len(paint.scroll_samples) == args.samples // 10, paint.scroll_pending
            assert transcript.scroll_y < scroll_start
            # Repeated frames must never snap toward the tail between wheels.
            assert all(
                b <= a for a, b in zip(paint.scroll_frames, paint.scroll_frames[1:])
            ), paint.scroll_frames
        if args.slow_stats_ms and children:
            assert stats_threads and loop_thread not in stats_threads
        if args.capture:
            root = Path(args.capture)
            root.mkdir(parents=True, exist_ok=True)
            save_capture(app, str(root / f"children-{children}.svg"))
        result = {
            "children": children,
            "input_to_compositor_paint": summary(paint.samples),
            "wheel_to_compositor_paint": summary(paint.scroll_samples),
            "scroll_offsets_per_frame": paint.scroll_frames if args.scroll else [],
            "slow_stats_calls": len(stats_threads),
            "slow_stats_ran_on_loop": loop_thread in stats_threads,
            "loop_wall_gap": summary(wall_gaps),
            "loop_cpu_gap": summary(cpu_gaps),
            "operations": {k: summary(v) for k, v in timings.values.items()},
            "frames": paint.frames,
            "elapsed_s": elapsed,
            "achieved_events_per_s": session.delivered / elapsed,
            "events_delivered": session.delivered,
            "bursts_delivered": session.edges_sent,
            "correctness": (
                f"{args.samples} characters painted; every progress edge delivered; "
                "latest job state matches"
            ),
        }
    subagent_panel.job_stats = original_stats
    return result


async def canaries() -> dict[str, Any]:
    app = OperatorApp(lambda: _factory(LoadSession(0)))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._editor().focus()
        paint = PaintProbe(app)
        paint.pending.append(("IMPOSSIBLE-MARKER", time.perf_counter()))
        paint.inject(app, "a", "a")
        await pilot.pause()
        assert len(paint.samples) == 1
        clean = paint.samples[0]

        def delayed_key() -> None:
            time.sleep(0.02)
            paint.inject(app, "ab", "b")

        thread = threading.Thread(target=delayed_key)
        thread.start()
        # Deliberate regression: CPU-only timing cannot see blocking I/O, but
        # input-to-painted-frame WALL latency must count time asleep on the loop.
        time.sleep(0.18)
        await asyncio.to_thread(thread.join)
        await pilot.pause()
        assert len(paint.samples) == 2
        assert [item[0] for item in paint.pending] == ["IMPOSSIBLE-MARKER"]
        blocked = paint.samples[1]
        assert blocked > 0.12, blocked
        return {
            "clean_ms": clean * 1000,
            "blocked_ms": blocked * 1000,
            "absent_marker_detected": False,
            "painted_characters": app._editor().text,
        }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--children", type=int, nargs="+", default=[0, 10, 50])
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--blocks", type=int, default=500)
    parser.add_argument(
        "--edges",
        type=int,
        default=3,
        help="minimum bursts; load continues until all keys injected",
    )
    parser.add_argument("--source-root", default=_source_args.source_root)
    parser.add_argument("--slow-stats-ms", type=float, default=0)
    parser.add_argument("--scroll", action="store_true")
    parser.add_argument("--attached", action="store_true")
    parser.add_argument("--capture")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checks = await canaries()
    checks["loop_probe"] = await loop_canaries()
    print(json.dumps({"canaries": checks}), flush=True)
    results = []
    for count in args.children:
        result = await scenario(count, args)
        results.append(result)
        print(json.dumps(result), flush=True)
    sources = {}
    for cls in (OperatorApp, AttachedSession, FrontendStateStore):
        filename = sys.modules[cls.__module__].__file__
        assert filename is not None
        path = Path(filename).resolve()
        assert path.is_relative_to(Path(args.source_root).resolve()), path
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "modules": sources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.executable,
    }
    Path(args.output).write_text(
        json.dumps(
            {"args": vars(args), "provenance": provenance, "canaries": checks, "results": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
