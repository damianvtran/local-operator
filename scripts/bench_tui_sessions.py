"""Measure the TUI's render path against a variable number of ATTACHED SESSIONS.

Run with the worktree interpreter, e.g.::

    .venv/bin/python scripts/bench_tui_sessions.py \
        --sessions 0 2 6 12 --output /tmp/tui-sessions.json

WHY THIS EXISTS. ``scripts/bench_tui_background.py`` varies ``--children`` *within*
one session, so the claim that the TUI's render cost grows with the number of
*attached sessions* had no instrument behind it — it was a linear extrapolation
of a single-session constant (``/tmp/tui-coupling-audit.md`` §7). This harness
varies the axis that claim is about: N sessions the TUI holds a viewer for, each
streaming its own canonical deltas onto the one asyncio loop, while input is
injected and the compositor's own paint boundary is timed.

WHAT IS MODELLED.
* Each session is a real ``AttachedSession`` (a viewer: ``owns_runtime = False``)
  with its own ``FrontendStateStore``, seeded with a realistic roster —
  ``--children`` running task jobs, each retaining ``--rows`` trajectory rows.
* Each session's frames come from the PRODUCTION writer path: a real
  ``AsyncJobManager`` roster plus ``FrontendStateStore.refresh_jobs`` →
  ``mutate``, which is what builds the ``changes['jobs']`` summaries and the
  trailing ``job_trajectory_appends``. They are delivered at the producer's own
  50 ms coalescing cadence (``session/session.py``) onto the decoded-callback
  boundary the socket codec hands the viewer —
  ``AttachedSession._on_frontend_update`` (``attach_client.py``) — which is the
  start point the neighbouring harness documents too.
* Session 0 is also the app's CURRENT session, so the current-session arm of the
  coupling (``OperatorApp._on_frontend_update`` → ``_apply_frontend_state``, which
  reads the whole state) is live. Sessions 1..N-1 are registered as leased
  sidebar sources, so the per-source fan-out and the retention predicate that
  clones the state for a boolean are live too. Those are exactly the three
  mechanisms the audit ranks as growing with attached sessions.
* The frames are BUILT BEFORE the measurement window and recycled, with only the
  sequence rewritten. Rebuilding the owner's roster on this loop for every
  delivery would charge the viewer for work production does in ANOTHER PROCESS;
  what the follower pays for depends on the retained window's SIZE, not on row
  identity, and a recycled frame still appends one row per child. The build cost
  is reported separately as ``pool_build_s``.
* No socket, no runtime, no provider and no retained session is used: nothing
  here can reach the operator's live sessions.

WHAT THE NUMBERS MEAN, AND WHAT THEY DO NOT.
This host is heavily shared, so wall time is dominated by descheduling: a
wall-clock percentile is an OBSERVATION, never a CI ceiling, and the same code on
an idle box produces much smaller numbers. CPU is ``time.thread_time()`` on the
loop thread (AGENTS.md, "If you must measure, measure CPU, not wall time"), which
is load-robust but blind to a pure blocking sleep — which is why the loop
canaries assert BOTH a CPU and a wall reading rather than one. Achieved frames
per second counts completed compositor updates, NOT terminal presentation.
Use ``--continuous`` to maintain redraw demand: natural-demand counts are not
render capacity. The CPU accounting ratio is not an idle-host FPS prediction.
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
# Fixtures (`tests.*`) belong to this benchmark's checkout; the APPLICATION must
# resolve the tree `--source-root` names, which is the mechanism the before/after
# comparison rests on. Both are added, the selected root first, and the
# provenance block below asserts which tree the app was actually imported from —
# an editable install resolves exactly ONE root, so without that assertion a
# before-run could silently measure the after tree and report the change as a
# no-op.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tests  # noqa: E402,F401

sys.path.insert(0, str(Path(_source_args.source_root).resolve()))
import scripts.probe_isolation  # noqa: E402,F401

# Older source roots only scrub CMUX variables in probe_isolation. A benchmark
# launched by a runtime must not inherit adoption/provider flags into its app.
for _key in tuple(os.environ):
    if _key.startswith(("CMUX_", "LOP_")):
        os.environ.pop(_key)

# Isolation normally disables shimmer for stills. The timing workload must keep
# production animation enabled, or the measurement removes the work it studies.
os.environ.pop("LOCAL_OPERATOR_NO_SHIMMER", None)

from textual._compositor import ChopsUpdate, LayoutUpdate  # noqa: E402
from textual.events import Key  # noqa: E402

from local_operator.harness.jobs import (  # noqa: E402
    TRAJECTORY_SEQ_KEY,
    AsyncJob,
    AsyncJobManager,
)
from local_operator.session.attached import AttachedSession  # noqa: E402
from local_operator.session.frontend_state import (  # noqa: E402
    _TRAJECTORY_CAP,
    FrontendModelSpec,
    FrontendSessionState,
    FrontendStateStore,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.session_interaction import SessionInteraction  # noqa: E402
from scripts.benchmark_residual_child_lag import _events  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The producer's real coalescing cadence: one canonical `jobs` delta per attached
#: session every 50 ms while a roster moves (``session/session.py:6095``, whose
#: comment is the source of the 20/s-per-session figure the audit derives). Kept
#: as the sweep's delivery interval so the load matches production rather than a
#: rate chosen to flatter the numbers.
DELTA_INTERVAL_S = 0.05

#: Input is injected from a SEPARATE OS THREAD at this cadence, so a blocked loop
#: cannot postpone the start of the measurement and thereby hide its own stall.
KEY_INTERVAL_S = 0.08

#: Deterministic alphabet seed and length: every session count types the same
#: string, so a latency difference between cells is load, not input.
KEY_SEED = 42

#: Bound on the streaming window in delta rounds (400 × 50 ms = 20 s at cadence).
#: A backstop only, and deliberately expressed in ROUNDS rather than wall seconds:
#: on a saturated loop each round costs far more wall time than its 50 ms sleep
#: (measured at ~8x on this host), so a wall bound would cut a run off in the
#: middle of a measurement while a round bound scales with the load.
DEFAULT_MAX_ROUNDS = 400

#: How many compositor updates the drain loop waits for after the last key is
#: queued. Bounded in loop TURNS rather than seconds because a turn count
#: survives contention that a wall-clock budget does not (AGENTS.md, "Wait on the
#: event, never on the clock").
DRAIN_TURNS = 400

#: Sites entered once per delivered delta at the TOP level, used for the
#: attributed per-delta total. Nested spans are excluded on purpose:
#: `FrontendStateStore.apply_update` runs inside
#: `AttachedSession._on_frontend_update`, and `OperatorApp._apply_frontend_state`
#: inside `_apply_pending_frontend_state`, so summing every label would count
#: that work twice and report a per-delta cost larger than the whole loop burned
#: — a number that reads as a finding and is an accounting error.
TOP_LEVEL_DELTA_SITES = (
    "AttachedSession._on_frontend_update",
    "OperatorApp._source_frontend_changed",
    "OperatorApp._apply_pending_frontend_state",
)


def summary(values: list[float]) -> dict[str, float | int]:
    """Percentiles of a sample, in milliseconds (the shape the sibling harness uses)."""
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values) * 1000 if values else 0.0,
        "p50_ms": statistics.median(values) * 1000 if values else 0.0,
        "p95_ms": ordered[int((len(values) - 1) * 0.95)] * 1000 if values else 0.0,
        "max_ms": max(values, default=0.0) * 1000,
    }


class CpuSpans:
    """Accumulate ``time.thread_time()`` per wrapped call, on the loop thread.

    WHY thread_time AND NOT perf_counter. A wall sample of a callback on this
    host is dominated by the OS not scheduling the process, so a wall-based
    per-delta cost would move with the weather rather than with the code. Thread
    CPU counts only while this thread really ran, which is what "the loop paid
    for this delta" means. Its blind spot is a callback that blocks in a syscall
    (the CPU clock does not advance while asleep) — the loop canaries below exist
    to prove that reading is still paired with a wall one, and every number this
    class produces is quoted as CPU in the report.
    """

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {}
        self.calls: dict[str, int] = {}

    def wrap(self, owner: Any, name: str, label: str) -> None:
        original = getattr(owner, name)

        def measured(*args: Any, **kwargs: Any) -> Any:
            started = time.thread_time()
            try:
                return original(*args, **kwargs)
            finally:
                self.samples.setdefault(label, []).append(time.thread_time() - started)
                self.calls[label] = self.calls.get(label, 0) + 1

        setattr(owner, name, measured)

    def report(self, deltas: int) -> dict[str, Any]:
        """Per-site CPU, divided once by every delivered delta.

        ``cpu_us_per_delta`` is the attributable cost of ONE canonical delta, so
        it is the column a fix to the delta path can move. Read it beside
        ``calls``: a site called once per CURRENT-session delta (the app's own
        handlers) is divided by every session's deltas, so its per-delta figure
        reads N times smaller than its per-call figure — ``mean_cpu_us`` is the
        per-call number for those sites.
        """
        out: dict[str, Any] = {}
        for label, values in self.samples.items():
            ordered = sorted(values)
            total = sum(values)
            out[label] = {
                "calls": self.calls[label],
                "total_cpu_ms": total * 1000,
                "mean_cpu_us": (statistics.fmean(values) * 1e6) if values else 0.0,
                "p50_cpu_us": statistics.median(values) * 1e6 if values else 0.0,
                "p95_cpu_us": ordered[int((len(ordered) - 1) * 0.95)] * 1e6 if ordered else 0.0,
                # Attributed per DELIVERED delta rather than per call: the point of
                # the sweep is what one attached session costs the loop per beat,
                # and the fan-out sites are called once per delta per source.
                "cpu_us_per_delta": (total / deltas * 1e6) if deltas else 0.0,
            }
        return out


def _appended_row(wave: int, index: int) -> dict[str, Any]:
    """One relayed child event, in the shape the shared ``_events`` fixture writes."""
    return {
        "type": "message_update",
        "message": {"role": "assistant", "id": f"m{wave}-{index}"},
        "delta": f"child {index} wave {wave}: " + "z" * 240,
    }


class ProducerRoster:
    """The OWNER-side half of one attached session.

    This object is never handed to the app: it exists so the frames the viewers
    reduce are produced by the production writer path (``refresh_jobs`` →
    ``mutate``) rather than by a hand-built dict that would silently drift from
    the wire shape. Only the slice ``FrontendStateStore.refresh_jobs`` reads is
    exposed — ``jobs``, ``model``, ``_subagent_comms``.
    """

    def __init__(self, session_id: str, children: int, rows: int, *, capped: bool = True) -> None:
        self.session_id = session_id
        self.capped = capped
        self.relayed = rows
        self.model = None
        # No lineage graph: a synthetic roster has no nested subagent tree, and
        # `_jobs` folds lineage only when a comms ledger is present.
        self._subagent_comms = None
        self.manager = AsyncJobManager(on_job_change=lambda: None)
        # Fixed-width wall timestamps keep before/after frame sizes comparable
        # without changing the elapsed-time presentation into a synthetic age.
        now = float(int(time.time()))
        self.children: list[AsyncJob] = []
        for index in range(children):
            job = AsyncJob(
                id=f"{session_id}-child-{index}",
                type="task",
                label=f"Synthetic child {index}",
                status="running",
                start_time=now - 60,
                started_at=now - 60,
                # A `task` job with no events is `None`, not `[]`: the roster
                # reader has to be able to tell "no events recorded" from "this
                # job type has none", and `--rows 0` exercises that shape.
                trajectory=[
                    {**row, TRAJECTORY_SEQ_KEY: sequence}
                    for sequence, row in enumerate(_events(((rows + 4) // 5) * 5)[:rows])
                ]
                or None,
                agent_role="coder",
                prompt="Synthetic load only",
            )
            # Registered directly into the ledger rather than through
            # `manager.register`, because registration schedules a real runner
            # coroutine — there is no child to run here, and the roster row is all
            # the writer path reads.
            self.manager._jobs[job.id] = job
            self.children.append(job)
        self.jobs = self.manager
        self.store = FrontendStateStore(
            FrontendSessionState(
                session_id=session_id,
                epoch=f"bench-{session_id}",
                # The band and the splash read the label off this spec, so a
                # snapshot without one would repaint them empty and measure a
                # screen production never shows.
                selected_model=FrontendModelSpec(provider="test", model_id="model"),
            )
        )
        # The seed every viewer installs its frontend from, in the same call the
        # runtime makes on the 50 ms roster coalescer.
        self.store.refresh_jobs(self)

    def tick(self, wave: int) -> dict[str, Any]:
        """Append one relayed event per child; return one canonical jobs delta.

        Mirror ``subagent.relay``: stamps advance independently of retained
        length and the oldest rows are removed at the cap. The current producer
        emits a full replacement when that trim breaks its prefix proof. The
        optional uncapped variant isolates suffix cost but is NOT a faithful
        long-running runtime workload.
        """
        for index, job in enumerate(self.children):
            # The runner mutates the SAME list. Replacing it per tick defeats
            # the producer's identity-based memo and measures the wrong path.
            rows = job.trajectory if job.trajectory is not None else []
            rows.append({**_appended_row(wave, index), TRAJECTORY_SEQ_KEY: self.relayed})
            if self.capped and len(rows) > _TRAJECTORY_CAP:
                del rows[: len(rows) - _TRAJECTORY_CAP]
            job.trajectory = rows
        self.relayed += 1
        update = self.store.refresh_jobs(self)
        if update is None:
            raise AssertionError("a moving roster must publish a delta")
        payload = update.model_dump(mode="json")
        if not payload.get("job_trajectory_appends"):
            # Replacement rows also travel in job_trajectory_appends; requiring
            # the body detects an accidentally summary-only timing workload.
            raise AssertionError("expected a delta carrying job_trajectory_appends")
        return payload


def build_pool(roster: ProducerRoster, size: int) -> list[dict[str, Any]]:
    """Pre-build `size` production-shaped delta frames for one session.

    Built before the measurement window on purpose: the producer's rebuild is
    owner-process work, and running it on the viewer's loop would charge the TUI
    for work production does in another process — the harness would then measure
    its own generator. The cost is reported as `pool_build_s` so the reader can
    see what was moved out of the window rather than taking it on trust.
    """
    return [roster.tick(index) for index in range(size)]


def build_viewer(session_id: str, config_dir: Path, seed_state: FrontendSessionState) -> Any:
    """A real viewer for one synthetic owner, installed from its attach seed."""

    def never() -> Any:
        # The takeover is the one thing this harness must never do: it would
        # turn a synthetic session into an attempt on a real one.
        raise AssertionError("the benchmark must not take over a live runtime")

    viewer = AttachedSession(
        config_dir=config_dir,
        session_id=session_id,
        takeover_factory=never,
    )
    viewer._install_frontend(seed_state)
    if viewer._frontend_store is None:
        raise AssertionError("the frontend store must be installed before any delta")
    return viewer


class AttachedFacade(FakeSession):
    """The app's CURRENT session, backed by a real attached viewer.

    The current session is a distinct arm of the coupling: its deltas reach
    ``OperatorApp._on_frontend_update`` → ``_apply_frontend_state``, which reads
    the whole state. A harness that only parked N viewers as sidebar sources
    would omit that arm entirely, and the number it published would be the cost
    of N-1 sessions.
    """

    # Viewer role (SessionProtocol): this fake owns no loop and learns outcomes
    # over the wire. `_is_viewer` reads these rather than the concrete class.
    owns_runtime = False
    outcome_is_synchronous = False
    runtime_locality = "this-machine"

    def __init__(self, viewer: Any) -> None:
        super().__init__()
        self.viewer = viewer
        self.jobs = viewer.jobs
        self._subagent_comms = viewer._subagent_comms

    @property
    def session_id(self) -> str:
        return self.viewer.session_id

    @property
    def frontend_state(self) -> Any:
        return self.viewer._frontend_store.state

    @property
    def has_running_job(self) -> bool:
        # Baseline trees retain their historical clone cost rather than silently
        # gaining the production narrow-read optimization in this facade.
        narrow = getattr(self.viewer, "has_running_job", None)
        if isinstance(narrow, bool):
            return narrow
        return any(job.status == "running" for job in self.frontend_state.jobs)

    def subscribe_frontend(self, callback: Any) -> Any:
        return self.viewer._frontend_store.subscribe(callback)

    def refresh_frontend_usage(self) -> None:
        """Satisfied by the viewer: nothing to refresh from the app's side."""


class PaintProbe:
    """Observe only rendered updates; never force an extra render to find text.

    The same paint-boundary technique ``scripts/bench_tui_background.py`` uses,
    and kept for the reason it states there: the measurement target is the frame
    Textual's compositor actually handed to ``_display``, not the round trip
    through ``pilot.press``/``pause`` and not the physical terminal's
    presentation latency. Frames are counted here, so the achieved frame rate is
    counted at the same boundary the latency is measured at.
    """

    def __init__(self, app: OperatorApp, *, continuous: bool = False) -> None:
        self.continuous = continuous
        self.frame_times: list[float] = []
        self.message_queue_peak = 0
        self.pending: list[tuple[str, float]] = []
        self.samples: list[float] = []
        self.frames = 0
        original = app._display

        def display(screen: Any, renderable: Any) -> None:
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
            self.frame_times.append(now)
            self.message_queue_peak = max(self.message_queue_peak, app._message_queue.qsize())
            if self.continuous:
                # Demand another real frame without bypassing the refresh timer
                # or doing a synchronous render inside the measuring probe.
                app.call_later(app.screen.refresh)
            for marker, started in self.pending[:]:
                if marker in text:
                    self.samples.append(now - started)
                    self.pending.remove((marker, started))

        app._display = display

    def inject(self, app: OperatorApp, prefix: str, char: str) -> None:
        """Queue one character and start its clock at the injection, not the paint."""
        self.pending.append((prefix[-8:], time.perf_counter()))
        app.post_message(Key(char, char))


async def loop_probe(
    stop: asyncio.Event,
    wall: list[float],
    cpu: list[float],
    ready: asyncio.Event | None = None,
) -> None:
    """Sample the loop's wall and CPU gap between wakes, independently."""
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
    """Prove the gap probe can see loop work, a blocking sleep, and idleness."""
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
    # Three separate facts, and all three are needed: the CPU clock must see CPU
    # work, the wall clock must see a blocking sleep the CPU clock is blind to,
    # and an idle loop must show neither. An instrument failing any of them would
    # report a cost it cannot attribute.
    assert readings["cpu"]["cpu"]["max_ms"] >= 20
    assert readings["blocking_io"]["wall"]["max_ms"] >= 50
    assert readings["blocking_io"]["cpu"]["max_ms"] < readings["cpu"]["cpu"]["max_ms"]
    assert readings["idle"]["cpu"]["max_ms"] < readings["cpu"]["cpu"]["max_ms"]
    return readings


async def canaries() -> dict[str, Any]:
    """Prove the paint probe sees a painted character, a stall, and nothing else.

    The NEGATIVE half is the load-bearing one: an absent marker must stay
    undetected, or a probe that matched anything would report a paint for input
    the app never rendered and every latency below would be a fiction.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
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
        # A deliberate regression, kept because it is the only thing separating
        # this probe from one that measures the injection call: CPU-only timing
        # cannot see a blocking sleep on the loop, but input-to-painted-frame WALL
        # latency must count the time the loop spent asleep.
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


async def scenario(count: int, args: argparse.Namespace) -> dict[str, Any]:
    """Measure one session count: boot, attach N viewers, stream, inject, report."""
    config_dir = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    # 12-hex ids, like the other benches: they are directory-shaped, so nothing
    # downstream has to special-case a synthetic conversation.
    ids = [f"{index + 1:012x}" for index in range(count)]

    # Build the owner side first, install each viewer from its attach seed, and
    # only then pre-build the delta pool. The order matters: `refresh_jobs` moves
    # the producer's sequence forward, so a pool built before the install would
    # leave every viewer expecting a sequence the producer has already passed and
    # every delta would be refused by the follower's exact-sequence check.
    rosters = [
        ProducerRoster(session_id, args.children, args.rows, capped=args.workload == "capped")
        for session_id in ids
    ]
    viewers = [
        build_viewer(session_id, config_dir, roster.store.state)
        for session_id, roster in zip(ids, rosters, strict=True)
    ]
    viewers_seed_rows = [
        [list(job.trajectory) for job in viewer._frontend_store.state.jobs] for viewer in viewers
    ]
    pool_started = time.perf_counter()
    pool = [build_pool(roster, args.pool) for roster in rosters]
    pool_build_s = time.perf_counter() - pool_started
    # The delivered frame's own size, reported so a reader can hold this run
    # against another one. A per-delta cost that moved because the FIXTURE moved
    # is not a cost that moved, and the audit's reference frame is 17.8 KiB —
    # quote the query, not just the digit.
    frame_bytes = sorted(len(json.dumps(frame)) for frame in pool[0]) if pool else []

    facade = AttachedFacade(viewers[0]) if count else FakeSession()
    app = OperatorApp(lambda: _factory(facade))
    wall_gaps: list[float] = []
    cpu_gaps: list[float] = []
    delivered = [0] * count
    # Each viewer's next expected sequence, read off the store it installed: the
    # follower refuses anything that is not exactly `sequence + 1`, so a rewrite
    # that guessed would fail loudly rather than skew a number.
    next_sequence = [viewer._frontend_store.state.sequence + 1 for viewer in viewers]

    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is facade:
                break
        assert app._session is facade

        # Sessions 1..N-1 become leased sidebar sources, exactly as the sidebar
        # registers them (``app._sidebar_sources[id] = SessionInteraction(remote)``
        # then ``_watch_source_frontend``), so the per-delta fan-out and the
        # retention predicate are real rather than simulated.
        sources: list[SessionInteraction] = []
        for session_id, viewer in list(zip(ids, viewers, strict=True))[1:]:
            source = SessionInteraction(viewer)
            # RETENTION, and it is the load model rather than a convenience. A
            # source with no presentation and no retention is RELEASABLE, so the
            # idle path would dispose it and the sweep would silently measure
            # fewer attached sessions than it reports. `approve_all` with a
            # running child is the arm the audit measures ("each one costs a deep
            # state copy per owner delta forever"): the viewer is retained by the
            # owner's live turn. Releasing it is the MITIGATION, not the load.
            source.draft.approve_all = True
            source.parked_at = time.monotonic()
            app._sidebar_sources[session_id] = source
            app._watch_source_frontend(source)
            sources.append(source)

        app._editor().focus()
        spans = CpuSpans()
        for viewer in viewers:
            spans.wrap(viewer, "_on_frontend_update", "AttachedSession._on_frontend_update")
            spans.wrap(viewer._frontend_store, "apply_update", "FrontendStateStore.apply_update")
        if sources:
            spans.wrap(app, "_source_frontend_changed", "OperatorApp._source_frontend_changed")
        if count:
            spans.wrap(
                app,
                "_apply_pending_frontend_state",
                "OperatorApp._apply_pending_frontend_state",
            )
            spans.wrap(app, "_apply_frontend_state", "OperatorApp._apply_frontend_state")

        paint = PaintProbe(app, continuous=args.continuous)
        if args.continuous:
            app.screen.refresh()
        stop = asyncio.Event()
        delivery_lags: list[float] = []
        pending_rounds: list[int] = []

        async def stream() -> None:
            """Offer one delta round per cadence until the last key has PAINTED.

            The load must outlive the last INJECTION, not merely the last sample
            request: a loop that stopped streaming while the final characters were
            still queued would time those samples in the quiet it had just
            created, and report a latency the loaded frame never had. The stop
            condition is therefore a loop-side fact (`len(paint.samples)`) plus
            the injecting thread's liveness, with a round budget as the backstop.
            """
            round_index = 0
            deadline = time.perf_counter()
            while (
                round_index < args.deltas
                or key_thread.is_alive()
                or len(paint.samples) < args.samples
            ) and round_index < args.max_rounds:
                # Fixed deadlines avoid understating offered load when reduction
                # is slow; sleeping after each round would throttle the producer.
                deadline += DELTA_INTERVAL_S
                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
                lag = max(0.0, time.perf_counter() - deadline)
                delivery_lags.append(lag)
                # Virtual queue: frames are generated from a bounded reusable
                # pool, never an unbounded task backlog. Report overdue offered
                # rounds instead of hiding overload behind successful delivery.
                pending_rounds.append(
                    min(args.max_rounds - round_index, 1 + int(lag / DELTA_INTERVAL_S))
                )
                for view, viewer in enumerate(viewers):
                    payload = pool[view][round_index % args.pool]
                    # Only the sequence is rewritten. The frame's CONTENT is a
                    # frozen production frame; regenerating the owner's roster per
                    # delivery would put owner work on this loop (see build_pool).
                    payload["sequence"] = next_sequence[view]
                    next_sequence[view] += 1
                    viewer._on_frontend_update(payload)
                    delivered[view] += 1
                paint.message_queue_peak = max(paint.message_queue_peak, app._message_queue.qsize())
                round_index += 1
            rounds_done[0] = round_index
            stream_finished[0] = time.perf_counter()
            stream_cpu_finished[0] = time.thread_time()

        rounds_done = [0]
        stream_finished = [0.0]
        stream_cpu_finished = [0.0]
        probe_task = asyncio.create_task(loop_probe(stop, wall_gaps, cpu_gaps))
        typed = "".join(random.Random(KEY_SEED).choices(string.ascii_letters, k=args.samples))

        def keys() -> None:
            prefix = ""
            for char in typed:
                time.sleep(KEY_INTERVAL_S)
                prefix += char
                paint.inject(app, prefix, char)

        key_thread = threading.Thread(target=keys, daemon=True, name="bench-keys")
        paint.frames = 0
        cpu_started = time.thread_time()
        started = time.perf_counter()
        stream_task = asyncio.create_task(stream())
        key_thread.start()
        # `to_thread` and not a bare `join`: a bare join would block the loop
        # thread for the whole injection window, which would deliver every delta
        # only AFTER the typing finished and measure an idle loop against a load
        # that had not started yet.
        await asyncio.to_thread(key_thread.join)
        for _ in range(DRAIN_TURNS):
            await pilot.pause()
            if len(paint.samples) >= args.samples:
                break
        await stream_task
        stop.set()
        await probe_task
        # End at the last delivered round, not the later pilot/probe drain.
        # Counting quiet frames after load stopped would inflate loaded FPS.
        finished = stream_finished[0]
        elapsed = finished - started
        cpu_used = stream_cpu_finished[0] - cpu_started
        paint.continuous = False
        measured_frames = [stamp for stamp in paint.frame_times if started <= stamp <= finished]
        frame_gaps = [right - left for left, right in zip(measured_frames, measured_frames[1:])]
        frames_painted = len(measured_frames)

        if args.capture:
            root = Path(args.capture)
            root.mkdir(parents=True, exist_ok=True)
            save_capture(app, str(root / f"sessions-{count}.svg"))
            await pilot.pause()
            save_capture(app, str(root / f"sessions-{count}-settled.svg"))

        # -- correctness. Every assertion here is a way the instrument could return
        # -- a plausible readout while measuring something else, so they run while
        # -- the app is still up (after `run_test` exits there is no editor to read).
        assert app._editor().text == typed, app._editor().text
        assert len(paint.samples) == args.samples, (len(paint.samples), paint.pending)
        if count:
            assert all(value > 0 for value in delivered), delivered
            for viewer, session_id, value in zip(viewers, ids, delivered, strict=True):
                store = viewer._frontend_store
                assert store is not None
                # The deltas really reached the reducer: the follower's sequence
                # advanced by exactly the number of frames delivered. A delta
                # refused by the exact-sequence check would leave this short, so a
                # sweep that silently delivered nothing cannot pass as a
                # measurement.
                assert store.state.sequence == value + 1, (
                    session_id,
                    store.state.sequence,
                    value,
                )
                seed_jobs = rosters[ids.index(session_id)].store.state.jobs
                # Length alone cannot detect a dropped/reordered append. Rebuild
                # the expected tail from the actual offered pool for every job.
                for job_index, job in enumerate(store.state.jobs):
                    expected = list(viewers_seed_rows[ids.index(session_id)][job_index])
                    for tick in range(value):
                        frame = pool[ids.index(session_id)][tick % args.pool]
                        if job.id in frame.get("job_trajectory_replacements", ()):
                            expected = []
                        expected.extend(frame.get("job_trajectory_appends", {}).get(job.id, []))
                        expected = expected[-_TRAJECTORY_CAP:]
                    assert list(job.trajectory) == expected, (session_id, job.id)
                    assert job.status == seed_jobs[job_index].status
            if args.children and args.rows:
                first = viewers[0]._frontend_store.state.jobs[0]
                # Every append landed, and the reducer's own cap is what bounds the
                # window: `rows + deltas` rows, clipped at `_TRAJECTORY_CAP`. At
                # the default `rows=500` the cap is already binding, which is the
                # shape the audit measured (a full retained window, one appended
                # row per child per delta) rather than an empty roster whose
                # deltas are cheap.
                assert len(first.trajectory) == min(args.rows + delivered[0], _TRAJECTORY_CAP), (
                    len(first.trajectory),
                    args.rows,
                    delivered[0],
                )

    total_deltas = sum(delivered)
    result = {
        "sessions": count,
        "workload": args.workload,
        # WHAT THIS INSTRUMENT DOES NOT MODEL, carried with every cell so a
        # reader cannot quote a number without its caveat. The delivery boundary
        # below is the decoded callback, which sits AFTER the transport's own
        # projection: `RuntimeServer._relay_frontend_to_on_loop` runs
        # `filter_update_trajectories` against that connection's watched jobs, so
        # a real parked connection never receives the trajectories this fixture
        # hands over unconditionally. Measured separately on the wire, a parked
        # capped rotation is ~3.8 KB/delta (constant) against ~207 KB for a
        # connection watching all six children. This harness is therefore the
        # WATCHED-ALL lane: right for the reducer and the render loop, and an
        # overstatement of ordinary parked fan-in.
        "transport": {
            "lane": "watched_all_detail",
            "ingest_boundary": "AttachedSession._on_frontend_update (decoded callback)",
            "watched_job_filter_applied": False,
            "parked_fan_in_modelled": False,
        },
        "replacement_jobs_per_frame": (
            [len(frame.get("job_trajectory_replacements", ())) for frame in pool[0]] if pool else []
        ),
        "children": args.children,
        "retained_rows": args.rows,
        "retained_roster_rows_per_session": args.children * args.rows,
        "delta_frame_bytes": {
            "min": frame_bytes[0] if frame_bytes else 0,
            "median": statistics.median(frame_bytes) if frame_bytes else 0,
            "max": frame_bytes[-1] if frame_bytes else 0,
        },
        "input_to_compositor_paint": summary(paint.samples),
        "loop_wall_gap": summary(wall_gaps),
        "loop_cpu_gap": summary(cpu_gaps),
        "window_s": elapsed,
        "measurement_window": "stream_start_to_last_delivered_round_no_drain",
        "render_demand": "continuously_dirty" if args.continuous else "natural_demand",
        "compositor_frame_gap": summary(frame_gaps),
        "frames_painted": frames_painted,
        "achieved_fps_wall": frames_painted / elapsed if elapsed else 0.0,
        "loop_cpu_s": cpu_used,
        # An accounting ratio, NOT idle-host FPS: Textual timers and offered
        # demand still use wall time, which cannot be divided away.
        "frames_per_loop_cpu_second": frames_painted / cpu_used if cpu_used else 0.0,
        "loop_cpu_share_of_wall": cpu_used / elapsed if elapsed else 0.0,
        "offered_interval_ms": DELTA_INTERVAL_S * 1000,
        "delivery_deadline_lag": summary(delivery_lags),
        "pending_offered_rounds_peak": max(pending_rounds, default=0),
        "pending_rounds_capacity": args.max_rounds,
        "queued_input_at_finish": len(paint.pending),
        "textual_message_queue_peak_observed": paint.message_queue_peak,
        "deltas_delivered": total_deltas,
        "deltas_per_session": delivered,
        "deltas_per_s": total_deltas / elapsed if elapsed else 0.0,
        "cpu": spans.report(total_deltas),
        # The attributable part of the loop's per-delta cost: the top-level
        # delta sites only (see TOP_LEVEL_DELTA_SITES). Deliberately NOT the whole
        # loop gap — the loop also paints and runs its own timers — so this is
        # the half a fix to the delta path can move, and the gap probe is the
        # half it cannot.
        "attributed_cpu_ms_per_delta": (
            sum(sum(spans.samples.get(label, ())) for label in TOP_LEVEL_DELTA_SITES)
            * 1000
            / total_deltas
            if total_deltas
            else 0.0
        ),
        "rounds": rounds_done[0],
        "round_budget_exhausted": rounds_done[0] >= args.max_rounds,
        "pool_size": args.pool,
        "pool_build_s": pool_build_s,
        "correctness": (
            f"{args.samples} characters painted; {total_deltas} canonical deltas applied "
            f"across {count} attached sessions; every sequence advanced in step"
        ),
    }
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions",
        type=int,
        nargs="+",
        default=[0, 2, 6, 12],
        help="attached-session counts to sweep; 0 is the no-load baseline",
    )
    parser.add_argument("--children", type=int, default=6, help="running children per session")
    parser.add_argument(
        "--workload",
        choices=("capped", "suffix"),
        default="capped",
        help="capped mirrors runtime eviction; suffix isolates uncapped append cost only",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=500,
        help="retained trajectory rows per child; 500 is the reducer's own cap",
    )
    parser.add_argument("--samples", type=int, default=120, help="injected characters")
    parser.add_argument(
        "--deltas",
        type=int,
        default=20,
        help="minimum delta rounds per session; load continues until every key has painted",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=8,
        help="pre-built delta frames per session, then recycled (see build_pool)",
    )
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--source-root", default=_source_args.source_root)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="keep compositor dirty; natural-demand frame counts are not capacity FPS",
    )
    parser.add_argument("--capture")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checks = await canaries()
    checks["loop_probe"] = await loop_canaries()
    print(json.dumps({"canaries": checks}), flush=True)
    results = []
    for count in args.sessions:
        result = await scenario(count, args)
        results.append(result)
        print(json.dumps(result), flush=True)

    sources: dict[str, str] = {}
    for cls in (OperatorApp, AttachedSession, FrontendStateStore):
        filename = sys.modules[cls.__module__].__file__
        assert filename is not None
        path = Path(filename).resolve()
        # The proof that `--source-root` selected the tree it names. Without it a
        # "before" run against a pristine checkout could silently measure this
        # tree and report the change under test as a no-op.
        assert path.is_relative_to(Path(args.source_root).resolve()), path
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        load = [round(value, 2) for value in os.getloadavg()]
    except OSError:  # pragma: no cover - not every platform has a load average
        load = []
    Path(args.output).write_text(
        json.dumps(
            {
                "args": vars(args),
                "provenance": {
                    "modules": sources,
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "python": sys.executable,
                    "load_average": load,
                },
                "canaries": checks,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
