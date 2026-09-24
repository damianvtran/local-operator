"""Submit -> admission -> turn start, on a real runtime, under the loads a fleet carries.

WHY THIS EXISTS
===============
The operator's report is "I press send and it takes a few seconds before the
message registers and starts processing" — on a host with ~20 live runtimes,
load average 20-100, and parents holding 8-16 in-process child lanes. The
existing rigs measure the pieces one at a time on a quiet runtime
(``bench_ttft.py``: warm send on an idle runtime; ``bench_attach_latency.py``:
the attach) and they pass: the echo and the acknowledgement are already local
and immediate. What they cannot show is what a LOADED runtime does between the
wire frame arriving and the turn starting, which is where the seconds are.

WHAT IT DRIVES
==============
The RUNTIME runs in a CHILD PROCESS of this script (``--child``), exactly the
production shape: its own interpreter and GIL, a real ``Session`` over a real
transcript, ``ServingSessionHandle`` and ``RuntimeServer.start()`` (the serving
plane on its own thread, the workload on the session loop). The VIEWER is the
production ``AttachedSession`` in this process, dialled over the real socket —
the object every front end (TUI, desktop daemon, mobile daemon) uses to send.
The only double is the provider: a paced scripted stream, so every number is
local-operator's own overhead.

Hops timestamped (``time.perf_counter`` is ``mach_absolute_time`` on macOS, so
it is comparable across the two processes):

  rpc_ms        viewer submit -> the runtime's control dispatch sees the frame
  session_prompt_ms / pipeline_ms / append_ms
                the same frame entering ``Session.prompt``, entering the turn
                pipeline (turn lock held), and reaching the durable user-row
                append — the hops between the RPC and the echo
  admit_ms      viewer submit -> the viewer holds the owner's receipt
  echo_ms       viewer submit -> the viewer sees the user ``message_start``
  start_ms      viewer submit -> the viewer sees ``agent_start`` (the working signal)
  request_ms    viewer submit -> the runtime's provider stream is CALLED
  token_ms      viewer submit -> the viewer sees the first text delta

Conditions (``--condition``), each a real shape from the live fleet:

  idle      nothing else running
  lanes     N in-process child lanes (``--lanes``) streaming and stepping
            tools on the parent's loop while the probe is sent
  roster    ``lanes`` plus a restored roster of ``--roster`` settled children
            with a ~13 KB ``effective_prompt`` each — the measured shape of the
            fleet's 1-4.4 MB ``subagent-roster.v1.json`` sidecars
  (any) + ``--burn K`` spawns K pure-Python busy loops owned by this script
            (reaped at exit) to reproduce host CPU contention

ISOLATION
=========
Refuses to run without ``LOCAL_OPERATOR_CONFIG_DIR``; the child writes only
under that root. ``LOP_*``/``CMUX_*`` are stripped from the child's env. Every
process this script starts is killed by exact pid at exit.

    ISO=$(mktemp -d)
    env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" PATH="$PATH" \\
      .venv/bin/python scripts/bench_send_admission.py --condition roster --lanes 12 --probes 8

Wall figures are observations on a shared host; compare arms interleaved on the
same host, and read ``loop_lag`` and the child's CPU beside them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

if not os.environ.get("LOCAL_OPERATOR_CONFIG_DIR"):
    raise SystemExit("refusing to run un-isolated: set HOME and LOCAL_OPERATOR_CONFIG_DIR")

CONFIG_DIR = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
SESSION_ID = "benchsend0001"
#: A launch prompt the size the fleet's roster records carry (effective_prompt
#: p50 across 54 live sidecars is dominated by a ~12.9 KB team/role preamble).
_PREAMBLE = ("You are a coder on this team. " * 440)[:12_900]


# ---------------------------------------------------------------------------
# child: the runtime
# ---------------------------------------------------------------------------


class PacedStream:
    """Text deltas at a fixed pace; lanes also call one tool per turn.

    ``lane_turns`` bounds a lane's run so a lane keeps stepping (tool, text,
    tool, ...) for the whole measurement, the way a delegated child does.
    """

    def __init__(self, marks: list[tuple[str, str, float]], lane_turns: int) -> None:
        self.marks = marks
        self.lane_turns = lane_turns

    def fork(self, _conversation: str) -> "PacedStream":
        return self

    async def close(self) -> None:  # pragma: no cover - fork protocol
        return None

    def __call__(self, request: Any, signal: Any = None) -> Any:
        from local_operator.harness.types import Message

        users = [m for m in request.messages if isinstance(m, Message) and m.role == "user"]
        text = users[-1].text if users else ""
        done = sum(1 for m in request.messages if isinstance(m, Message) and m.role == "assistant")
        if text.startswith("probe-") or text.startswith("long-"):
            self.marks.append(("request", text.split()[0], time.perf_counter()))
        return self._gen(text, done)

    async def _gen(self, text: str, done: int):
        from local_operator.harness.types import (
            StreamEndEvent,
            StreamTextDelta,
            StreamToolCallDelta,
            Usage,
        )

        lane = text.startswith(_PREAMBLE[:40])
        pieces = 400 if text.startswith("long-") else 40
        for i in range(pieces):
            yield StreamTextDelta(delta="word ")
            await asyncio.sleep(0.004 if i % 5 == 4 else 0)
        usage = Usage(input_tokens=20_000 + done * 1500, output_tokens=pieces)
        if lane and done + 1 < self.lane_turns:
            yield StreamToolCallDelta(
                index=0, id=f"call-{done}", name="todo", argument_delta=json.dumps({"op": "view"})
            )
            yield StreamEndEvent(stop_reason="toolUse", usage=usage)
            return
        yield StreamEndEvent(stop_reason="stop", usage=usage)


def _seed_roster(directory: Path, count: int) -> None:
    """Write a settled-children sidecar shaped like the fleet's largest ones."""
    from local_operator.session.session import (
        _SUBAGENT_ROSTER_VERSION,
        SUBAGENT_ROSTER_SIDECAR,
        _write_roster_sidecar,
    )

    records = []
    for i in range(count):
        job_id = f"old{i:05d}"
        records.append(
            {
                "job_id": job_id,
                "label": f"settled child {i}",
                "parent_job_id": None,
                "prompt": "Review the change and report findings. " * 12,
                "effective_prompt": _PREAMBLE + " Review the change and report findings.",
                "launch_message_id": f"subagent-launch:{job_id}",
                "agent_role": "reviewer",
                "effort": "",
                "restricted": False,
                "session_dir": str(directory.parent / f"child-{job_id}"),
                "outcome": "completed",
                "cut_off_cause": "",
                "result_text": "ok " * 160,
                "error_text": None,
                "paused": False,
                "settled_at": time.time() - 3600,
                "attempt_aliases": [],
                "prior_launch_prompts": {},
            }
        )
    directory.mkdir(parents=True, exist_ok=True)
    _write_roster_sidecar(
        directory / SUBAGENT_ROSTER_SIDECAR,
        {
            "version": _SUBAGENT_ROSTER_VERSION,
            "generation": 1,
            "jobs": [],
            "records": records,
            "accounting": [],
        },
    )


class _Sampler:
    """Wall-clock stack sampler of ONE thread, from inside the process.

    ``sys._current_frames`` needs the GIL, so a sample lands at the next moment
    the sampler gets it — which attributes the loop thread's time to the frame
    it was in when the GIL was taken from it. That is the attribution wanted:
    "what was the loop doing while the probe waited".
    """

    def __init__(self, thread_id: int, period: float = 0.002) -> None:
        self.thread_id = thread_id
        self.period = period
        self.counts: dict[str, int] = {}
        self.leaf: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bench-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(2)

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            frame = sys._current_frames().get(self.thread_id)
            seen: set[str] = set()
            first = True
            while frame is not None:
                code = frame.f_code
                if "local_operator" in code.co_filename or first:
                    key = (
                        f"{Path(code.co_filename).name}:{code.co_name}"
                        if "local_operator" in code.co_filename
                        else f"<{Path(code.co_filename).name}:{code.co_name}>"
                    )
                    if first:
                        self.leaf[key] = self.leaf.get(key, 0) + 1
                        first = False
                    if key not in seen:
                        seen.add(key)
                        self.counts[key] = self.counts.get(key, 0) + 1
                frame = frame.f_back


async def _child_main(args: argparse.Namespace) -> None:
    from local_operator.harness.types import AgentStartEvent, MessageStartEvent, ModelSpec
    from local_operator.session.runtime import server as server_module
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.harness.types import ToolContext
    from local_operator.tools.registry import create_tools

    marks: list[tuple[str, str, float]] = []
    directory = CONFIG_DIR / "sessions" / SESSION_ID
    if args.roster:
        _seed_roster(directory, args.roster)
    stream = PacedStream(marks, args.lane_turns)
    session = Session(
        model=ModelSpec(provider="test", model_id="bench", context_window=200_000),
        stream_fn=stream,
        tools=create_tools(ToolContext(cwd=str(directory)), enabled=["todo"]),
        transcript=Transcript(directory),
        system_blocks_provider=lambda *a, **k: ["bench"],
        cwd=str(directory),
        yolo=True,
    )
    await session.async_init()
    session.jobs.set_max_running(max(1, args.lanes))

    def on_event(event: Any) -> None:
        if isinstance(event, MessageStartEvent) and getattr(event.message, "role", "") == "user":
            marks.append(("emit_user", event.message.text.split()[0], time.perf_counter()))
        elif isinstance(event, AgentStartEvent):
            marks.append(("emit_start", "", time.perf_counter()))

    session.subscribe(on_event)

    original_dispatch = server_module.RuntimeServer._dispatch

    async def dispatch(self: Any, op: str, frame: dict[str, Any], **kw: Any) -> Any:
        if op in ("prompt", "steer"):
            marks.append(("rpc", str(frame.get("text", "")).split()[0], time.perf_counter()))
        return await original_dispatch(self, op, frame, **kw)

    server_module.RuntimeServer._dispatch = dispatch  # type: ignore[method-assign]

    # Hop marks INSIDE the admission path, wrapped at the class so the
    # production objects are otherwise untouched: when the handle's body starts
    # on the session loop, when Session.prompt starts, when it holds the turn
    # lock, and when the user row is durably appended.
    from local_operator.session.runtime import serving as serving_module
    from local_operator.session.transcript import Transcript as _Transcript

    def _wrap(owner: Any, name: str, kind: str, pick: Any) -> None:
        original = getattr(owner, name)

        async def wrapped(self: Any, *a: Any, **kw: Any) -> Any:
            tag = pick(a, kw)
            if tag:
                marks.append((kind, tag, time.perf_counter()))
            return await original(self, *a, **kw)

        wrapped.__wrapped__ = original  # type: ignore[attr-defined]
        setattr(owner, name, wrapped)

    def _probe_text(a: Any, kw: Any) -> str:
        text = str(a[0] if a else kw.get("text", ""))
        return text.split()[0] if text.startswith(("probe-", "long-")) else ""

    _wrap(Session, "prompt", "session_prompt", _probe_text)
    _wrap(Session, "_run_turn_pipeline", "pipeline", lambda a, kw: _pipeline_tag(a))
    _wrap(
        _Transcript,
        "append_messages",
        "append",
        lambda a, kw: _pipeline_tag((list(a[0]) if a else [],)),
    )
    del serving_module

    # Per-call CPU of the loop's recurring publishers, so a share of loop
    # samples can be turned into "N calls x M ms each" (thread_time: the
    # honest instrument on a loaded host — it excludes time not scheduled).
    from local_operator.mobile.projection import ProjectionFold
    from local_operator.session.frontend_state import FrontendStateStore

    call_cpu: dict[str, list[float]] = {}

    def _time_sync(owner: Any, name: str) -> None:
        original = getattr(owner, name)

        def timed(self: Any, *a: Any, **kw: Any) -> Any:
            c0 = time.thread_time()
            try:
                return original(self, *a, **kw)
            finally:
                call_cpu.setdefault(name, []).append((time.thread_time() - c0) * 1000)

        setattr(owner, name, timed)

    if os.environ.get("BENCH_PROFILE_REFRESH_JOBS"):
        # Deterministic profile of ONE publisher's calls, aggregated across the
        # run: which function inside it eats the budget.
        import cProfile

        profiler = cProfile.Profile()
        original_refresh = FrontendStateStore.refresh_jobs

        def profiled(self: Any, *a: Any, **kw: Any) -> Any:
            profiler.enable()
            try:
                return original_refresh(self, *a, **kw)
            finally:
                profiler.disable()

        FrontendStateStore.refresh_jobs = profiled  # type: ignore[method-assign]
        import atexit

        atexit.register(lambda: profiler.dump_stats(os.environ["BENCH_PROFILE_REFRESH_JOBS"]))
    _time_sync(FrontendStateStore, "refresh_jobs")
    _time_sync(FrontendStateStore, "refresh_from_session")
    _time_sync(ProjectionFold, "set_subagent_details")

    loop = asyncio.get_running_loop()
    handle = ServingSessionHandle(session, loop, cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    server.start()
    await server.wait_until_published()
    (directory / ".session.pid").write_text(str(os.getpid()))

    lags: list[float] = []

    async def heartbeat() -> None:
        while True:
            due = loop.time() + 0.01
            await asyncio.sleep(0.01)
            lags.append(max(0.0, loop.time() - due))

    beat = asyncio.create_task(heartbeat())
    for i in range(args.lanes):
        session._launch_subagent(label=f"lane-{i}", prompt=_PREAMBLE + f" lane {i} work")
    sampler = _Sampler(threading.get_ident()) if args.sample else None
    if sampler:
        sampler.start()
    print("READY", flush=True)
    cpu0 = time.process_time()
    # The parent closes our stdin when the probes are done.
    await loop.run_in_executor(None, sys.stdin.read)
    cpu = time.process_time() - cpu0
    if sampler:
        sampler.stop()
    beat.cancel()
    report = {
        "marks": marks,
        "loop_lag_ms": _pcts([x * 1000 for x in lags]),
        "child_cpu_s": cpu,
        "roster_writes": getattr(session, "_subagent_roster_written_generation", None),
        "call_cpu_ms": {
            name: {**_pcts(values), "total": round(sum(values), 1)}
            for name, values in call_cpu.items()
        },
    }
    if sampler:
        total = sum(sampler.leaf.values()) or 1
        report["sample_total"] = total
        report["sample_inclusive"] = sorted(sampler.counts.items(), key=lambda kv: -kv[1])[:70]
        report["sample_leaf"] = sorted(sampler.leaf.items(), key=lambda kv: -kv[1])[:20]
    Path(args.report).write_text(json.dumps(report))
    if os.environ.get("BENCH_PROFILE_REFRESH_JOBS"):
        import atexit

        atexit._run_exitfuncs()
    os._exit(0)  # the lanes are still stepping; the parent owns our lifetime


# ---------------------------------------------------------------------------
# parent: the viewer
# ---------------------------------------------------------------------------


def _pipeline_tag(args: Any) -> str:
    """The probe tag of the user message a pipeline/append call carries, if any."""
    for message in args[0] if args else []:
        text = str(getattr(message, "text", "") or "")
        if getattr(message, "role", "") == "user" and text.startswith(("probe-", "long-")):
            return text.split()[0]
    return ""


def _pcts(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)

    return {"p50": at(0.5), "p95": at(0.95), "max": round(ordered[-1], 1), "n": len(ordered)}


def _child_env() -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("LOP_", "CMUX_", "XPC_FLAGS"))
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


async def _wait_record(timeout: float = 60.0) -> Any:
    from local_operator.session.runtime import registry

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for record, _state in registry.scan(CONFIG_DIR):
            if getattr(record, "session_id", "") == SESSION_ID:
                return record
        await asyncio.sleep(0.05)
    raise RuntimeError("runtime never published")


async def _parent_main(args: argparse.Namespace) -> dict[str, Any]:
    from local_operator.harness.types import (
        AgentStartEvent,
        MessageStartEvent,
        MessageUpdateEvent,
    )
    from local_operator.session.attached import AttachedSession

    burners: list[subprocess.Popen[bytes]] = []
    report_path = CONFIG_DIR / f"child-report-{uuid.uuid4().hex[:8]}.json"
    child_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--report",
        str(report_path),
        "--lanes",
        str(args.lanes if args.condition != "idle" else 0),
        "--roster",
        str(args.roster if args.condition == "roster" else 0),
        "--lane-turns",
        str(args.lane_turns),
    ] + (["--sample"] if args.sample else [])
    child = subprocess.Popen(
        child_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=_child_env()
    )
    try:
        assert child.stdout is not None
        line = await asyncio.to_thread(child.stdout.readline)
        if b"READY" not in line:
            raise RuntimeError(f"child did not start: {line!r}")
        record = await _wait_record()
        for _ in range(args.burn):
            burners.append(
                subprocess.Popen([sys.executable, "-c", "while True: pass"], env=_child_env())
            )

        async def never() -> Any:
            raise RuntimeError("never take over")

        viewer = await AttachedSession.connect(
            record, SESSION_ID, config_dir=CONFIG_DIR, takeover_factory=never
        )
        seen: list[tuple[str, str, float]] = []

        def on_event(event: Any) -> None:
            now = time.perf_counter()
            if isinstance(event, MessageStartEvent) and getattr(event.message, "role", "") == "user":
                seen.append(("echo", event.message.text.split()[0], now))
            elif isinstance(event, AgentStartEvent):
                seen.append(("start", "", now))
            elif isinstance(event, MessageUpdateEvent):
                seen.append(("delta", "", now))

        viewer.subscribe(on_event)
        # Let the lanes get going: the point is a probe against a BUSY loop.
        await asyncio.sleep(args.settle)
        rows: list[dict[str, float]] = []
        for index in range(args.probes):
            while viewer.is_streaming:
                await asyncio.sleep(0.02)
            await asyncio.sleep(args.gap)
            tag = f"probe-{index}"
            since = len(seen)
            t0 = time.perf_counter()
            await viewer.prompt(tag + " hello")
            admitted = time.perf_counter()
            row: dict[str, float] = {"t0": t0, "admit_ms": (admitted - t0) * 1000}
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                window = seen[since:]
                echo = next((s for s in window if s[0] == "echo" and s[1] == tag), None)
                if echo is not None:
                    after = [s for s in window if s[2] >= echo[2] - 0.5]
                    start = next((s for s in after if s[0] == "start"), None)
                    delta = next((s for s in after if s[0] == "delta" and start and s[2] >= start[2]), None)
                    if start is not None and delta is not None:
                        row["echo_ms"] = (echo[2] - t0) * 1000
                        row["start_ms"] = (start[2] - t0) * 1000
                        row["token_ms"] = (delta[2] - t0) * 1000
                        break
                await asyncio.sleep(0.005)
            row["tag"] = index  # type: ignore[assignment]
            rows.append(row)
        await viewer.dispose()
    finally:
        for proc in burners:
            proc.kill()
        for proc in burners:
            proc.wait()
        if child.stdin is not None:
            child.stdin.close()
        try:
            await asyncio.to_thread(child.wait, 60)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    child_report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for row in rows:
        tag = f"probe-{int(row['tag'])}"
        for kind, name, ts in child_report.get("marks", []):
            if name == tag and ts >= row["t0"]:
                key = {
                    "rpc": "rpc_ms",
                    "session_prompt": "session_prompt_ms",
                    "pipeline": "pipeline_ms",
                    "append": "append_ms",
                    "request": "request_ms",
                    "emit_user": "emit_ms",
                }.get(kind)
                if key and key not in row:
                    row[key] = (ts - row["t0"]) * 1000
    summary = {
        key: _pcts([r[key] for r in rows if key in r])
        for key in (
            "rpc_ms",
            "session_prompt_ms",
            "pipeline_ms",
            "append_ms",
            "emit_ms",
            "admit_ms",
            "echo_ms",
            "start_ms",
            "request_ms",
            "token_ms",
        )
    }
    return {
        "condition": args.condition,
        "lanes": args.lanes if args.condition != "idle" else 0,
        "roster": args.roster if args.condition == "roster" else 0,
        "burn": args.burn,
        "load": os.getloadavg()[0],
        "summary": summary,
        "loop_lag_ms": child_report.get("loop_lag_ms"),
        "child_cpu_s": child_report.get("child_cpu_s"),
        "call_cpu_ms": child_report.get("call_cpu_ms"),
        "sample_total": child_report.get("sample_total"),
        "sample_inclusive": child_report.get("sample_inclusive"),
        "sample_leaf": child_report.get("sample_leaf"),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("idle", "lanes", "roster"), default="idle")
    parser.add_argument("--lanes", type=int, default=12)
    parser.add_argument("--roster", type=int, default=240)
    parser.add_argument("--lane-turns", type=int, default=400)
    parser.add_argument("--burn", type=int, default=0, help="background busy-loop processes")
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--gap", type=float, default=0.3, help="idle gap before each probe")
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--sample", action="store_true", help="stack-sample the runtime loop")
    parser.add_argument("--json", default="")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--report", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    import logging

    logging.disable(logging.WARNING)
    if args.child:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        asyncio.run(_child_main(args))
        return 0
    out = asyncio.run(_parent_main(args))
    brief = {k: v for k, v in out.items() if k not in ("rows", "sample_inclusive", "sample_leaf")}
    print(json.dumps(brief))
    if args.sample:
        print("-- runtime loop, inclusive samples --")
        for name, count in out.get("sample_inclusive") or []:
            print(f"  {count:6d} {name}")
        print("-- runtime loop, leaf --")
        for name, count in out.get("sample_leaf") or []:
            print(f"  {count:6d} {name}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
