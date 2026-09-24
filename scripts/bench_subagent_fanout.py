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
* ``cpu_s``          -- process CPU for the whole fan-out (host-noise tolerant).

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

    def __call__(self, request: ChatRequest, signal: Any = None):
        assistant_turns = sum(
            1 for m in request.messages if isinstance(m, Message) and m.role == "assistant"
        )
        return self._gen(assistant_turns)

    async def _gen(self, done: int):
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
            yield StreamEndEvent(stop_reason="stop", usage=usage)
            return
        args = {"command": "true"} if self.tool == "bash" else {"op": "view"}
        yield StreamToolCallDelta(
            index=0, id=f"call-{done}", name=self.tool, argument_delta=json.dumps(args)
        )
        self.turn_times.append(time.perf_counter() - start)
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


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


async def run_solo(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    stream = ScriptedStream(
        turns=args.turns, reasoning=args.reasoning, text=args.text, pace_s=args.pace, tool=args.tool
    )
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=_tools([args.tool]),
        transcript=Transcript(root / "solo"),
        system_blocks_provider=lambda *a, **k: ["bench"],
        cwd=str(root),
        yolo=True,
    )
    await session.async_init()
    start = time.perf_counter()
    await session.prompt("do the scripted work")
    wall = time.perf_counter() - start
    await session.dispose()
    return {"solo_turn_ms": statistics.median(stream.turn_times) * 1000, "solo_wall_s": wall}


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
    await asyncio.gather(*(parent.jobs.settled_event(j).wait() for j in ids))
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    stop.set()
    await beat
    statuses = [getattr(parent.jobs.get(j), "status", "gone") for j in ids]
    if subscription is not None:
        subscription.unsubscribe()
    await parent.dispose()
    return {
        "children": n,
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
    parser.add_argument("--profile", help="write a cProfile of the largest fan-out here")
    parser.add_argument("--label", default="")
    parser.add_argument("--output")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        solo = await run_solo(args, root)
        rows = []
        for n in args.children:
            prof = cProfile.Profile() if args.profile and n == max(args.children) else None
            if prof:
                prof.enable()
            row = await run_fanout(n, args, root)
            if prof:
                prof.disable()
                prof.dump_stats(args.profile)
            row["slowdown"] = row["child_turn_ms"] / solo["solo_turn_ms"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = {
        "label": args.label,
        "source": str(REPO),
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
