"""Offline benchmark: what N concurrent subagents cost the loop they share.

WHY THIS EXISTS. Every subagent is a full ``Session`` running on the PARENT's
event loop, so any per-event cost in the child pipeline (trajectory records,
frontend-state folds, parent relays, transcript writes) is paid N times over on
one thread. Operators reported children that run visibly slower than a parent
doing the same work, and parents that go unresponsive while waiting on a large
fan-out. This script reproduces that shape with NO provider and NO credentials:
a scripted stream that emits reasoning, text and a tool call per model turn, at
a fixed per-token pacing, so the only thing that varies between arms is the
harness itself.

What it reports, per fan-out size N:

* ``solo_turn_ms``   -- the median wall time of one scripted model turn in a
  single top-level session (the "parent speed" baseline);
* ``child_turn_ms``  -- the same, measured inside each of N concurrent children;
* ``slowdown``       -- child / solo. 1.0 means a child is as fast as a parent;
* ``loop_lag_ms``    -- p50/p99/max of a 10 ms heartbeat's lateness on the shared
  loop while the fan-out runs (what a parent's ``wait``/UI experiences);
* ``launch_ms``      -- time from the first launch until every child has started;
* ``settle_ms``      -- time from the last child's final message until the
  parent's settled event for it fires;
* ``cpu_s``          -- process CPU for the whole fan-out (host-noise tolerant);
* ``step_gap_ms``    -- p50/p95/max of the NON-TOOL overhead of one step: the
  wall time from a model call's final stream event to the SAME conversation's
  next provider request (tool execution is the scripted ``todo`` no-op, so this
  is request assembly, relay, persistence and loop contention). Reported for
  the solo parent and for the children, which is the "child vs parent step"
  the operator asked about;
* ``hub_wake_ms``    -- with ``--hub``, every child's tool call is a ``hub``
  note to its parent; this is the time from that delivery until a parked
  parent-side waiter on the arrival signal actually runs.

``--runtime`` (on by default) attaches the production runtime host
(``ServingSessionHandle``) to the parent, which is what a runtime-hosted
conversation -- every desktop/phone session -- runs per root event.
``--history N`` seeds N settled children into the parent's registry first: a
long-lived parent carries up to ``MAX_RECORDS`` of them, and a per-event cost
that is linear in the registry is invisible on an empty one.

Run it from a worktree with that worktree's interpreter, ISOLATED (the children
write real session directories under the config dir):

    ISO=$(mktemp -d)
    env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" \
      PATH="$PATH" TERM=xterm-256color \
      .venv/bin/python scripts/bench_subagent_fanout.py --children 1 8 24

Wall figures are observations on a shared host, never CI assertions; compare
arms with ``cpu_s`` and ``slowdown`` measured back to back on the same host.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import os
import pstats
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

if not os.environ.get("LOCAL_OPERATOR_CONFIG_DIR"):
    raise SystemExit(
        "refusing to run un-isolated: set HOME and LOCAL_OPERATOR_CONFIG_DIR (see docstring)"
    )

import local_operator  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamReasoningDelta,
    StreamTextDelta,
    StreamToolCallDelta,
    Usage,
)
from local_operator.session.session import Session  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402

MODEL = ModelSpec(provider="test", model_id="bench", context_window=200_000)


class ScriptedStream:
    """A provider that thinks, talks, calls one tool, and finishes after K turns.

    ``turn_times`` collects each model call's wall time keyed by the session
    directory the request belongs to, so solo and child turns are measured by
    the same clock at the same boundary.
    """

    def __init__(self, *, turns: int, reasoning: int, text: int, pace_s: float, tool: str) -> None:
        self.turns = turns
        self.reasoning = reasoning
        self.text = text
        self.pace_s = pace_s
        self.tool = tool
        self.turn_times: list[float] = []
        #: Non-tool overhead per step, see ``step_gap_ms``: keyed by the
        #: conversation's first message id, which is unique per session and
        #: stable across its requests, so solo and child gaps are measured by
        #: the same clock at the same two boundaries.
        self.step_gaps: list[float] = []
        self.final_ends: list[float] = []
        self._last_end: dict[str, float] = {}

    def __call__(self, request: ChatRequest, signal: Any = None):
        now = time.perf_counter()
        key = str(getattr(request.messages[0], "id", "") if request.messages else "")
        ended = self._last_end.pop(key, None)
        if ended is not None:
            self.step_gaps.append(now - ended)
        assistant_turns = sum(
            1 for m in request.messages if isinstance(m, Message) and m.role == "assistant"
        )
        return self._gen(assistant_turns, key)

    async def _gen(self, done: int, key: str = ""):
        start = time.perf_counter()
        # Token pacing: a real stream yields to the loop between chunks, and
        # the loop's other tenants run in those gaps. Batched sleeps keep the
        # timer count sane at high N while preserving the per-chunk yield.
        for i in range(self.reasoning):
            yield StreamReasoningDelta(delta="thinking about it ")
            if i % 10 == 9:
                await asyncio.sleep(self.pace_s)
            else:
                await asyncio.sleep(0)
        for i in range(self.text):
            yield StreamTextDelta(delta="word ")
            if i % 10 == 9:
                await asyncio.sleep(self.pace_s)
            else:
                await asyncio.sleep(0)
        usage = Usage(input_tokens=20_000 + done * 1500, output_tokens=self.text + self.reasoning)
        if done + 1 >= self.turns:
            self.turn_times.append(time.perf_counter() - start)
            self.final_ends.append(time.perf_counter())
            yield StreamEndEvent(stop_reason="stop", usage=usage)
            return
        if self.tool == "bash":
            args: dict[str, Any] = {"command": "true"}
        elif self.tool == "hub":
            args = {"message": f"progress note {done}"}
        else:
            args = {"op": "view"}
        yield StreamToolCallDelta(
            index=0, id=f"call-{done}", name=self.tool, argument_delta=json.dumps(args)
        )
        self.turn_times.append(time.perf_counter() - start)
        self._last_end[key] = time.perf_counter()
        yield StreamEndEvent(stop_reason="toolUse", usage=usage)

    # Session-owned forks share this script (and its clock) with the parent.
    def fork(self, _conversation: str) -> "ScriptedStream":
        return self

    async def close(self) -> None:  # pragma: no cover - fork protocol
        return None


def _tools(names: list[str]):
    from local_operator.harness.types import ToolContext
    from local_operator.tools.registry import create_tools

    return create_tools(ToolContext(cwd=os.getcwd()), enabled=names)


async def _heartbeat(stop: asyncio.Event, lags: list[float], period: float = 0.01) -> None:
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        due = loop.time() + period
        await asyncio.sleep(period)
        lags.append(max(0.0, loop.time() - due))


def _revision() -> dict[str, Any]:
    """``git rev-parse HEAD`` of the tree this script runs from, plus dirtiness."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": dirty}


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def _dist_ms(xs: list[float]) -> dict[str, float]:
    return {
        "p50": _pct(xs, 0.5) * 1000,
        "p95": _pct(xs, 0.95) * 1000,
        "max": max(xs, default=0.0) * 1000,
        "n": len(xs),
    }


def _seed_history(parent: Session, root: Path, count: int) -> None:
    """Give the parent ``count`` settled children, as a long-lived parent has.

    Real transcript files, because the registry's roster probes each one's
    existence; synthetic content, because no retained user session is read.
    """
    if count <= 0:
        return
    rows = []
    for index in range(count):
        directory = root / f"history-{id(parent)}-{index}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "transcript.jsonl").write_text("{}\n")
        rows.append(
            {
                "job_id": f"hist-{index}",
                "label": f"earlier child {index}",
                "session_dir": str(directory),
                "prompt": "an earlier delegated task " * 40,
                "outcome": "completed",
                "settled_at": time.time() - 60 - index,
                "result_text": "done " * 200,
            }
        )
    parent.subagent_comms.restore(rows)


def _attach_runtime(parent: Session, root: Path) -> Any:
    """The production runtime host, subscribed exactly as ``RuntimeServer`` does."""
    from local_operator.session.runtime.serving import ServingSessionHandle

    handle = ServingSessionHandle(
        parent, asyncio.get_running_loop(), cwd=str(root), install_gates=False
    )
    pushes = [0]

    def _push() -> None:
        pushes[0] += 1

    handle.subscribe(_push)
    return handle, pushes


async def run_solo(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    # ``hub`` exists only inside a child (its target is the parent), so the
    # solo baseline takes the same no-op step with ``todo`` instead.
    tool = "todo" if args.tool == "hub" else args.tool
    stream = ScriptedStream(
        turns=args.turns, reasoning=args.reasoning, text=args.text, pace_s=args.pace, tool=tool
    )
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=_tools([tool]),
        transcript=Transcript(root / "solo"),
        system_blocks_provider=lambda *a, **k: ["bench"],
        cwd=str(root),
        yolo=True,
    )
    await session.async_init()
    # The same host the fan-out parent gets, so a parent step and a child step
    # are compared under the same per-event runtime work.
    handle, _pushes = _attach_runtime(session, root) if args.runtime else (None, [0])
    start = time.perf_counter()
    await session.prompt("do the scripted work")
    wall = time.perf_counter() - start
    if handle is not None:
        await handle.dispose()
    else:
        await session.dispose()
    return {
        "solo_turn_ms": statistics.median(stream.turn_times) * 1000,
        "solo_wall_s": wall,
        "solo_step_gap_ms": _dist_ms(stream.step_gaps),
    }


async def run_fanout(n: int, args: argparse.Namespace, root: Path) -> dict[str, Any]:
    stream = ScriptedStream(
        turns=args.turns, reasoning=args.reasoning, text=args.text, pace_s=args.pace, tool=args.tool
    )
    parent = Session(
        model=MODEL,
        stream_fn=stream,
        tools=_tools([args.tool]),
        transcript=Transcript(root / f"parent-{n}"),
        system_blocks_provider=lambda *a, **k: ["bench"],
        cwd=str(root),
        yolo=True,
    )
    await parent.async_init()
    parent.jobs.set_max_running(max(n, 1))
    _seed_history(parent, root, args.history)
    handle, pushes = _attach_runtime(parent, root) if args.runtime else (None, [0])
    # Hub latency: stamp every arrival and let a parked waiter record when it
    # actually ran. This is the wedge shape where a child's ``hub`` note sat
    # unread while the parent loop was busy projecting the roster.
    hub_marks: list[float] = []
    hub_wakes: list[float] = []
    arrival = parent._peer_arrival

    class _StampedArrival:
        """Delegates to the real signal; stamps the instant of each arrival.

        A wrapper because ``_PeerArrival`` is slotted, and the session reads
        the attribute on every ``queue_aside`` -- so the stamp is taken at
        exactly the delivery point production uses.
        """

        # Spelled out rather than ``__getattr__``: ``ToolContext`` validates
        # this field against a runtime-checkable protocol, which inspects the
        # class for these members.
        def event(self) -> asyncio.Event:
            return arrival.event()

        def count(self) -> int:
            return arrival.count()

        def arrivals(self) -> Any:
            return arrival.arrivals()

        def mark(self, *a: Any, **k: Any) -> None:
            hub_marks.append(time.perf_counter())
            arrival.mark(*a, **k)

    parent._peer_arrival = _StampedArrival()  # type: ignore[assignment]

    async def _hub_waiter(done: asyncio.Event) -> None:
        seen = 0
        while not done.is_set():
            event = arrival.event()
            waiter = asyncio.ensure_future(event.wait())
            stopper = asyncio.ensure_future(done.wait())
            await asyncio.wait({waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)
            for pending in (waiter, stopper):
                pending.cancel()
            now = time.perf_counter()
            for stamp in hub_marks[seen:]:
                hub_wakes.append(now - stamp)
            seen = len(hub_marks)
            event.clear()

    hub_done = asyncio.Event()
    hub_task = asyncio.create_task(_hub_waiter(hub_done))
    # An attached viewer: every real parent has one (TUI, desktop, phone), and
    # a subscriber is what arms the parent's 50 ms roster coalescer -- the
    # path that re-projects every child's trajectory while the fan-out runs.
    frames = 0

    def _on_update(_update: Any) -> None:
        nonlocal frames
        frames += 1

    subscription = parent.subscribe_frontend(_on_update) if args.viewer else None
    stop = asyncio.Event()
    lags: list[float] = []
    beat = asyncio.create_task(_heartbeat(stop, lags))
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    ids = [
        parent._launch_subagent(label=f"child-{i}", prompt="do the scripted work") for i in range(n)
    ]
    # Launch-to-running: every child has a trajectory once its runner started.
    while any(getattr(parent.jobs.get(j), "trajectory", None) is None for j in ids):
        await asyncio.sleep(0.005)
    launch = time.perf_counter() - t0
    woke: list[float] = []

    async def _wait_one(job_id: str) -> None:
        await parent.jobs.settled_event(job_id).wait()
        woke.append(time.perf_counter())

    await asyncio.gather(*(_wait_one(j) for j in ids))
    wall = time.perf_counter() - t0
    settle = (max(woke) - max(stream.final_ends)) if woke and stream.final_ends else 0.0
    hub_done.set()
    await hub_task
    cpu = time.process_time() - cpu0
    stop.set()
    await beat
    statuses = [getattr(parent.jobs.get(j), "status", "gone") for j in ids]
    if subscription is not None:
        subscription.unsubscribe()
    if handle is not None:
        await handle.dispose()
    else:
        await parent.dispose()
    return {
        "children": n,
        "history": args.history,
        "runtime": bool(args.runtime),
        "step_gap_ms": _dist_ms(stream.step_gaps),
        "settle_ms": settle * 1000,
        "hub_wake_ms": _dist_ms(hub_wakes),
        "runtime_pushes": pushes[0],
        "child_turn_ms": statistics.median(stream.turn_times) * 1000,
        "child_turn_p90_ms": _pct(stream.turn_times, 0.9) * 1000,
        "launch_ms": launch * 1000,
        "wall_s": wall,
        "cpu_s": cpu,
        "loop_lag_ms": {
            "p50": _pct(lags, 0.5) * 1000,
            "p99": _pct(lags, 0.99) * 1000,
            "max": max(lags, default=0.0) * 1000,
        },
        "statuses": sorted(set(str(s) for s in statuses)),
        "viewer_frames": frames,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--children", type=int, nargs="+", default=[1, 8, 24])
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--reasoning", type=int, default=200)
    parser.add_argument("--text", type=int, default=200)
    parser.add_argument("--pace", type=float, default=0.002, help="seconds per 10 stream chunks")
    parser.add_argument("--tool", default="todo")
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="attach a frontend subscriber to the parent (the realistic case)",
    )
    parser.add_argument(
        "--runtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="attach the production runtime host (ServingSessionHandle) to the parent",
    )
    parser.add_argument(
        "--history", type=int, default=0, help="settled children seeded into the parent first"
    )
    parser.add_argument("--profile", help="write a cProfile of the largest fan-out here")
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--repeat", type=int, default=1, help="run each fan-out size this many times"
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        solo = await run_solo(args, root)
        rows = []
        for n in [size for size in args.children for _ in range(max(1, args.repeat))]:
            prof = cProfile.Profile() if args.profile and n == max(args.children) else None
            if prof:
                prof.enable()
            row = await run_fanout(n, args, root)
            if prof:
                prof.disable()
                prof.dump_stats(args.profile)
            row["slowdown"] = row["child_turn_ms"] / solo["solo_turn_ms"]
            solo_gap = solo["solo_step_gap_ms"]["p50"]
            row["step_gap_ratio"] = row["step_gap_ms"]["p50"] / solo_gap if solo_gap else 0.0
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = {
        "label": args.label,
        "source": str(REPO),
        # Self-attesting arm identity: an A/B arm is whichever tree this script
        # file sits in (``sys.path.insert(0, REPO)`` beats the venv's editable
        # finder), and a worktree path stops identifying a commit the moment
        # the worktree is deleted. The SHA, a dirty flag and the module path
        # actually imported make every result file provable on its own.
        "revision": _revision(),
        "imported_from": str(Path(local_operator.__file__).resolve().parent),
        "params": {k: v for k, v in vars(args).items() if k not in {"output", "profile"}},
        "solo": solo,
        "fanout": rows,
        "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }
    print(json.dumps(solo))
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
    if args.profile:
        stats = pstats.Stats(args.profile)
        stats.sort_stats("cumulative").print_stats(40)


if __name__ == "__main__":
    asyncio.run(main())
