"""Model / effort switch latency on a real runtime, under a long manager's load.

WHY THIS EXISTS
===============
The operator's report: on a long-running manager session (110 MB transcript,
2.7 MB ``subagent-roster.v1.json``, usually mid-turn with many lanes) the
desktop's ``/model`` request outlives the app's 20 s request deadline. A switch
changes one spec field the NEXT provider call reads, so its latency should be a
constant, independent of transcript size, roster size or whether a turn is in
flight. This rig measures what the switch actually costs end to end and where.

WHAT IT DRIVES
==============
The production shape of ``scripts/bench_send_admission.py`` (whose child
building blocks it reuses): the RUNTIME runs in a child process — a real
``Session`` over a real transcript, ``ServingSessionHandle`` and
``RuntimeServer.start()`` — and the viewer is the production
``AttachedSession`` in this process, dialled over the real socket. The provider
is a paced scripted stream, so every number is local-operator's own overhead.

Probes, each sent over the viewer's real ``AttachClient``:

  set_model     ``set_model`` op (the phone sheet / desktop picker / peer)
  set_effort    ``set_effort`` op
  slash_model   ``slash_result`` ``/model <p>/<id>`` (the desktop command path)

Conditions:

  idle      nothing running; the floor
  busy      ``--lanes`` child lanes streaming + a seeded ``--roster`` + a
            seeded transcript of ``--transcript-mb`` + the parent mid-turn
  parked    ``busy`` plus one ``prompt`` sent on the SAME connection while the
            parent is mid-turn (it parks until the turn lock frees, which is
            what a desktop send during a long manager turn does), then the
            switch — the connection's op chain is what this cell measures

Per probe the child records when the dispatcher SAW the op (``rpc_ms``; both
``_dispatch`` and ``_dispatch_payload``, so ``slash_result`` is covered), when
the handle's body STARTED on the session loop (``body_ms``), when the spec
build and the credential read STARTED on their threads (``build_ms``,
``access_ms``: the gap from ``body_ms`` is executor queueing), and the CPU each
sub-step took; the viewer records the round trip (``ack_ms``).

The child also samples its own loop: ``loop_lag_ms`` is a 5 ms heartbeat's
lateness, and ``stall_offenders_ms`` attributes every stalled sample to the
innermost ``local_operator`` frame then on the loop thread, so a late
``rpc_ms`` can be traced to the synchronous work that held the loop. A
``selectors.py:select`` row is the loop waiting for the GIL after its poll
returned, i.e. worker threads' CPU, not loop work. Under an isolated ``HOME``
``resolve.py:_macos_defaults_language`` appears too: with no
``.GlobalPreferences.plist`` the language probe falls back to a subprocess,
which a real home does not.

ISOLATION
=========
Refuses to run without ``LOCAL_OPERATOR_CONFIG_DIR``; writes only under it.
``LOP_*``/``CMUX_*``/``XPC_FLAGS`` are stripped from the child; the child is
reaped by exact pid.

    ISO=$(mktemp -d)
    env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" PATH="$PATH" \\
      .venv/bin/python scripts/bench_model_switch.py --condition parked
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
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import bench_send_admission as base  # noqa: E402  (refuses un-isolated)

CONFIG_DIR = base.CONFIG_DIR
SESSION_ID = base.SESSION_ID
_LADDER = ("low", "medium", "high")


async def _seed_transcript(directory: Path, megabytes: int) -> None:
    """Append ``megabytes`` of settled turns so the transcript is manager-sized.

    Written through the real ``Transcript`` so the rows are the production
    shape (and the session's boot reads them as it would a real one).
    """
    if megabytes <= 0:
        return
    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    body = "historic tool output line. " * 400  # ~10.8 KB per row

    if True:
        written = 0
        target = megabytes * 1024 * 1024
        while written < target:
            batch = []
            for i in range(50):
                batch.append(Message.user(f"old question {written} {i}"))
                batch.append(Message.assistant(body))
                written += len(body) + 64
            await transcript.append_messages(batch)


async def _child_main(args: argparse.Namespace) -> None:
    from local_operator.harness.types import ModelSpec, ToolContext
    from local_operator.model import configure
    from local_operator.session.runtime import server as server_module
    from local_operator.session.runtime import serving as serving_module
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.tools.registry import create_tools

    marks: list[tuple[str, str, float]] = []
    cpu: dict[str, list[float]] = {}
    directory = CONFIG_DIR / "sessions" / SESSION_ID
    if args.roster:
        base._seed_roster(directory, args.roster)
    await _seed_transcript(directory, args.transcript_mb)

    # The ``test`` provider carries no reasoning ladder; give it one so
    # ``set_effort`` and the effort carry run their real branches.
    original_build = configure.build_model_spec

    def build(provider: str, model_id: str, *a: Any, **kw: Any) -> Any:
        # When the build STARTS (on whichever thread runs it): the gap from the
        # body mark is the executor queue wait a saturated pool imposes.
        marks.append(("build", current_tag[0], time.perf_counter()))
        spec = original_build(provider, model_id, *a, **kw)
        if provider == "test":
            spec = spec.model_copy(
                update={"reasoning_efforts": _LADDER, "reasoning_default_effort": "medium"}
            )
        return spec

    configure.build_model_spec = build  # type: ignore[assignment]

    # The switch's second worker-thread hop: the credential read behind the
    # access claim. ``access`` marks when it STARTS on its thread, ``accessed``
    # when it returns; with ``build`` this splits the switch into queue waits.
    from local_operator.providers import model_access as access_module

    original_access = access_module.serving_model_access_here

    def access(*a: Any, **kw: Any) -> Any:
        marks.append(("access", current_tag[0], time.perf_counter()))
        try:
            return original_access(*a, **kw)
        finally:
            marks.append(("accessed", current_tag[0], time.perf_counter()))

    access_module.serving_model_access_here = access  # type: ignore[assignment]

    stream = base.PacedStream(marks, args.lane_turns)
    session = Session(
        model=ModelSpec(
            provider="test",
            model_id="bench-a",
            context_window=200_000,
            reasoning_efforts=_LADDER,
            reasoning_default_effort="medium",
            reasoning_effort="medium",
        ),
        stream_fn=stream,
        tools=create_tools(ToolContext(cwd=str(directory)), enabled=["todo"]),
        transcript=Transcript(directory),
        system_blocks_provider=lambda *a, **k: ["bench"],
        cwd=str(directory),
        yolo=True,
    )
    await session.async_init()
    session.jobs.set_max_running(max(1, args.lanes))

    original_dispatch = server_module.RuntimeServer._dispatch

    async def dispatch(self: Any, op: str, frame: dict[str, Any], **kw: Any) -> Any:
        tag = str(frame.get("tag") or "")
        if tag:
            marks.append(("rpc", tag, time.perf_counter()))
        return await original_dispatch(self, op, frame, **kw)

    server_module.RuntimeServer._dispatch = dispatch  # type: ignore[method-assign]

    # ``slash_result`` answers with a payload, so it is dispatched through
    # ``_dispatch_payload`` and never reaches ``_dispatch``: without this mark
    # the desktop's own ``/model`` route had no dispatcher-receipt time.
    original_payload = server_module.RuntimeServer._dispatch_payload

    async def dispatch_payload(
        self: Any, op: str, frame: dict[str, Any], *a: Any, **kw: Any
    ) -> Any:
        tag = str(frame.get("tag") or "")
        if tag:
            marks.append(("rpc", tag, time.perf_counter()))
        return await original_payload(self, op, frame, *a, **kw)

    server_module.RuntimeServer._dispatch_payload = dispatch_payload  # type: ignore[method-assign]

    # Body-start marks: the first statement the handle runs ON the session loop.
    current_tag: list[str] = [""]

    def _body_mark(name: str) -> None:
        original = getattr(serving_module.ServingSessionHandle, name)
        inner = getattr(original, "__wrapped__", None)
        if inner is None:
            return

        async def body(self: Any, *a: Any, **kw: Any) -> Any:
            marks.append(("body", current_tag[0], time.perf_counter()))
            return await inner(self, *a, **kw)

        setattr(serving_module.ServingSessionHandle, name, serving_module._on_session_loop(body))

    for name in ("set_model_effort", "set_effort", "run_slash_authoritative"):
        _body_mark(name)

    def _time_sync(owner: Any, name: str) -> None:
        original = getattr(owner, name, None)
        if original is None:
            return

        def timed(self: Any, *a: Any, **kw: Any) -> Any:
            c0 = time.perf_counter()
            try:
                return original(self, *a, **kw)
            finally:
                cpu.setdefault(name, []).append((time.perf_counter() - c0) * 1000)

        setattr(owner, name, timed)

    for name in ("_refresh_roster", "_refresh_state", "publish_model_access"):
        _time_sync(ServingSessionHandle, name)
    _time_sync(Session, "set_model")
    # The parts of ``Session.set_model`` (wall, on the loop): which one scales.
    for part in (
        "refresh_frontend_state",
        "_rebuild_effort_tier_tools",
        "_spawn_selected_model_write",
        "_journal_knobs_if_selection_changed",
        "_spawn_model_change",
        "_drop_fallback_pin",
    ):
        _time_sync(Session, part)

    loop = asyncio.get_running_loop()
    handle = ServingSessionHandle(session, loop, cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    server.start()
    await server.wait_until_published()
    (directory / ".session.pid").write_text(str(os.getpid()))
    for i in range(args.lanes):
        session._launch_subagent(label=f"lane-{i}", prompt=base._PREAMBLE + f" lane {i} work")

    # LOOP-STALL SAMPLER. A frame the dispatcher sees late was sent while the
    # session loop was running something synchronous. A watchdog thread samples
    # the loop thread's stack whenever the loop's heartbeat is older than
    # ``STALL_MS``, and attributes each stalled sample to the innermost
    # ``local_operator`` frame on that stack — the code that held the loop.
    stall_ms = 20.0
    beat = [time.perf_counter()]
    loop_ident = threading.get_ident()
    offenders: dict[str, float] = {}
    lags: list[float] = []

    async def heartbeat() -> None:
        while True:
            t = time.perf_counter()
            lags.append((t - beat[0]) * 1000)
            beat[0] = t
            await asyncio.sleep(0.005)

    def watchdog() -> None:
        import traceback

        period = 0.005
        last = time.perf_counter()
        while True:
            time.sleep(period)
            # Weighted by the REAL time since the previous sample, not the
            # nominal period: under load this thread is GIL-starved too, and a
            # fixed weight would under-count exactly the longest stalls.
            now = time.perf_counter()
            weight, last = (now - last) * 1000, now
            age = (now - beat[0]) * 1000
            if age < stall_ms:
                continue
            frame = sys._current_frames().get(loop_ident)
            if frame is None:
                continue
            stack = traceback.extract_stack(frame)
            ours = [f for f in stack if "/local_operator/" in f.filename]
            where = ours[-1] if ours else stack[-1]
            outer = ours[0] if ours else stack[0]
            key = (
                f"{Path(where.filename).name}:{where.name}"
                f" <- {Path(outer.filename).name}:{outer.name}"
            )
            offenders[key] = offenders.get(key, 0.0) + weight

    heartbeat_task = asyncio.ensure_future(heartbeat())
    threading.Thread(target=watchdog, daemon=True).start()

    # A control line from the parent sets the tag the next body mark carries.
    def control() -> None:
        for line in sys.stdin:
            current_tag[0] = line.strip()

    threading.Thread(target=control, daemon=True).start()
    print("READY", flush=True)
    stopped = asyncio.Event()
    loop.add_signal_handler(signal.SIGTERM, stopped.set)
    await stopped.wait()
    heartbeat_task.cancel()
    Path(args.report).write_text(
        json.dumps(
            {
                "marks": marks,
                "cpu_ms": {k: base._pcts(v) for k, v in cpu.items()},
                "loop_lag_ms": base._pcts(lags),
                "stall_offenders_ms": dict(sorted(offenders.items(), key=lambda kv: -kv[1])[:15]),
            }
        )
    )
    os._exit(0)


async def _parent_main(args: argparse.Namespace) -> dict[str, Any]:
    from local_operator.session.attached import AttachedSession

    report_path = CONFIG_DIR / "switch-report.json"
    busy = args.condition != "idle"
    child = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--report",
            str(report_path),
            "--lanes",
            str(args.lanes if busy else 0),
            "--roster",
            str(args.roster if busy else 0),
            "--transcript-mb",
            str(args.transcript_mb if busy else 0),
            "--lane-turns",
            str(args.lane_turns),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=base._child_env(),
    )
    rows: list[dict[str, Any]] = []
    try:
        assert child.stdout is not None and child.stdin is not None
        line = await asyncio.to_thread(child.stdout.readline)
        if b"READY" not in line:
            raise RuntimeError(f"child did not start: {line!r}")
        record = await base._wait_record(timeout=180)

        async def never() -> Any:
            raise RuntimeError("never take over")

        viewer = await AttachedSession.connect(
            record, SESSION_ID, config_dir=CONFIG_DIR, takeover_factory=never
        )
        client = viewer._client
        assert client is not None
        await asyncio.sleep(args.settle)
        parked: asyncio.Task[Any] | None = None
        if busy:
            # The parent mid-turn: a long paced answer holds the turn lock.
            await client.prompt("long-turn " + "x")
        if args.condition == "parked":
            # Sent while the turn runs, on the SAME connection: it parks inside
            # the handle until the turn lock frees, holding the op chain.
            await asyncio.sleep(0.3)
            parked = asyncio.create_task(client.prompt("probe-parked hello"))
            await asyncio.sleep(0.3)

        def tag(name: str) -> None:
            child.stdin.write(f"{name}\n".encode())  # type: ignore[union-attr]
            child.stdin.flush()  # type: ignore[union-attr]

        models = ["bench-b", "bench-a"]
        for index in range(args.probes):
            for kind in ("set_model", "set_effort", "slash_model"):
                name = f"{kind}-{index}"
                tag(name)
                await asyncio.sleep(0.02)
                t0 = time.perf_counter()
                try:
                    if kind == "set_model":
                        detail = await client._request(
                            "set_model", provider="test", model_id=models[index % 2], tag=name
                        )
                    elif kind == "set_effort":
                        detail = await client._request(
                            "set_effort", effort=_LADDER[index % 3], tag=name
                        )
                    else:
                        detail = await client._request_payload(
                            "slash_result",
                            command="model",
                            args=f"test/{models[(index + 1) % 2]}",
                            images=[],
                            tag=name,
                        )
                    error = ""
                except Exception as exc:  # noqa: BLE001 — recorded, not fatal
                    detail, error = "", f"{type(exc).__name__}: {exc}"
                rows.append(
                    {
                        "tag": name,
                        "kind": kind,
                        "t0": t0,
                        "ack_ms": (time.perf_counter() - t0) * 1000,
                        "detail": str(detail)[:80],
                        "error": error[:160],
                    }
                )
                await asyncio.sleep(args.gap)
        if parked is not None:
            parked.cancel()
        await viewer.dispose()
    finally:
        child.send_signal(signal.SIGTERM)
        try:
            await asyncio.to_thread(child.wait, 60)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    child_report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for row in rows:
        for kind, name, ts in child_report.get("marks", []):
            if name == row["tag"] and ts >= row["t0"]:
                key = f"{kind}_ms"
                row.setdefault(key, (ts - row["t0"]) * 1000)
    summary = {
        kind: {
            key: base._pcts([r[key] for r in rows if r["kind"] == kind and key in r])
            for key in ("rpc_ms", "body_ms", "build_ms", "access_ms", "accessed_ms", "ack_ms")
        }
        for kind in ("set_model", "set_effort", "slash_model")
    }
    return {
        "imported": base._tree_identity(),
        "condition": args.condition,
        "lanes": args.lanes if busy else 0,
        "roster": args.roster if busy else 0,
        "transcript_mb": args.transcript_mb if busy else 0,
        "summary": summary,
        "child_cpu_ms": child_report.get("cpu_ms"),
        "loop_lag_ms": child_report.get("loop_lag_ms"),
        "stall_offenders_ms": child_report.get("stall_offenders_ms"),
        "errors": [r for r in rows if r["error"]],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("idle", "busy", "parked"), default="busy")
    parser.add_argument("--lanes", type=int, default=12)
    parser.add_argument("--roster", type=int, default=240)
    parser.add_argument("--transcript-mb", type=int, default=40)
    parser.add_argument("--lane-turns", type=int, default=400)
    parser.add_argument("--probes", type=int, default=5)
    parser.add_argument("--gap", type=float, default=0.2)
    parser.add_argument("--settle", type=float, default=3.0)
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
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
