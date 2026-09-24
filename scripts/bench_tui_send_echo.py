"""Enter -> the user's bubble on screen, in the REAL TUI attached to a LOADED runtime.

WHY THIS EXISTS
===============
The TUI paints the user's bubble before it calls ``prompt()`` (``_submit_prompt``
mounts the ``UserBlock`` synchronously), so the runtime cannot delay the echo.
Yet against a parent carrying 12 stepping child lanes and 240 settled children,
Enter -> bubble measured 2-4 s p50 and 11-32 s p95: the TUI's OWN event loop was
saturated re-projecting the child roster the runtime streamed at it (status band
52% of loop samples, roster snapshot apply 36%, subagent panel 33%, 40-47 ms CPU
per call). This rig is the acceptance test for that: it measures the hop the
user sees, on the production objects, under the load that broke it.

WHAT IT DRIVES
==============
* The RUNTIME in a child process of this script (``--child``): a real
  ``Session`` over a real transcript, ``ServingSessionHandle`` and
  ``RuntimeServer`` (serving plane on its own thread), N in-process child lanes
  that keep stepping (text, one tool call, text, ...) and a restored roster of
  M settled children with ~13 KB launch prompts -- the measured shape of the
  fleet's largest ``subagent-roster.v1.json`` sidecars. The only double is the
  provider: a paced scripted stream, so every number is local-operator's own.
* The VIEWER in this process: the production ``AttachedSession`` dialled over
  the real socket, handed to a real ``OperatorApp`` (stylesheet loaded, driven
  by Textual's ``run_test`` pilot). Enter is delivered as a real key event
  through the driver, exactly as a terminal would.

Per probe, milliseconds after Enter:

  submit   ``OperatorApp._submit_prompt`` entered (the key was handled)
  mount    the probe's ``UserBlock`` mounted (the echo, local)
  frame    the first compositor frame after the mount (the echo, ON SCREEN)

Beside them: the TUI loop's lag (a 10 ms heartbeat's overshoot -- wall, so read
it against the host load printed with it) and the per-call CPU
(``time.thread_time``) of the three paths the stack sampler blamed.

WHICH SOURCE TREE
=================
The rig imports ``local_operator`` from the tree it LIVES in (``--repo``
overrides), put first on ``sys.path`` for this process and via ``PYTHONPATH``
for the runtime child, and it PRINTS the ``__file__`` both processes actually
imported. A bench that silently measures a different checkout than the one
named is the failure review found on #1528; the printed paths are what make a
before/after pair trustworthy. For an A/B, point ``--repo`` at a checkout of
the base and one of the change, with the SAME script:

    ISO=$(mktemp -d)
    env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" PATH="$PATH" \\
      TERM=xterm-256color .venv/bin/python scripts/bench_tui_send_echo.py \\
      --repo <checkout> --lanes 12 --roster 240 --probes 8 --json out.json

ISOLATION
=========
Refuses to run unless ``LOCAL_OPERATOR_CONFIG_DIR`` is set AND lies under
``HOME``, so it cannot touch an operator's real store. ``CMUX_*``/``LOP_*`` and
``XPC_FLAGS`` are stripped from this process before any app import (a headless
TUI that inherits ``CMUX_WORKSPACE_ID`` renames the operator's real cmux
workspaces) and from the child. Every process this script starts is ended by
exact pid on exit. Wall figures are observations on a shared host: compare arms
interleaved on the same host, never against a fixed ceiling (AGENTS.md
"Timing, flakes").
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

for _key in tuple(os.environ):
    if _key.startswith(("CMUX_", "LOP_")) or _key == "XPC_FLAGS":
        os.environ.pop(_key)
os.environ.pop("NO_COLOR", None)
os.environ["TERM"] = "xterm-256color"
os.environ["LOCAL_OPERATOR_NO_SHIMMER"] = "1"
os.environ["LOCAL_OPERATOR_NO_NOTIFICATIONS"] = "1"
os.environ["LOCAL_OPERATOR_NO_DESKTOP_LAUNCH"] = "1"


def _repo_from_argv() -> Path:
    """``--repo`` if given, else the tree this script lives in -- before any import."""
    for index, arg in enumerate(sys.argv):
        if arg == "--repo" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).resolve()
        if arg.startswith("--repo="):
            return Path(arg.split("=", 1)[1]).resolve()
    return Path(__file__).resolve().parents[1]


REPO = _repo_from_argv()
sys.path.insert(0, str(REPO))

_config = os.environ.get("LOCAL_OPERATOR_CONFIG_DIR", "")
_home = os.environ.get("HOME", "")
if not _config or not _home or not Path(_config).resolve().is_relative_to(Path(_home).resolve()):
    raise SystemExit(
        "refusing to run un-isolated: set HOME to a scratch dir and "
        "LOCAL_OPERATOR_CONFIG_DIR beneath it"
    )
CONFIG_DIR = Path(_config)
SESSION_ID = "benchecho0001"
#: A launch prompt the size the fleet's roster records carry (effective_prompt
#: p50 across 54 live sidecars is dominated by a ~12.9 KB team/role preamble).
_PREAMBLE = ("You are a coder on this team. " * 440)[:12_900]


def _pcts(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)

    return {"p50": at(0.5), "p95": at(0.95), "max": round(ordered[-1], 1), "n": len(ordered)}


# ---------------------------------------------------------------------------
# child: the loaded runtime
# ---------------------------------------------------------------------------


class _PacedStream:
    """Text deltas at a fixed pace; a lane calls one tool per turn for ``lane_turns``."""

    def __init__(self, lane_turns: int) -> None:
        self.lane_turns = lane_turns

    def fork(self, _conversation: str) -> "_PacedStream":
        return self

    async def close(self) -> None:  # pragma: no cover - fork protocol
        return None

    def __call__(self, request: Any, signal: Any = None) -> Any:
        from local_operator.harness.types import Message

        users = [m for m in request.messages if isinstance(m, Message) and m.role == "user"]
        text = users[-1].text if users else ""
        done = sum(1 for m in request.messages if isinstance(m, Message) and m.role == "assistant")
        return self._gen(text, done)

    async def _gen(self, text: str, done: int) -> Any:
        from local_operator.harness.types import (
            StreamEndEvent,
            StreamTextDelta,
            StreamToolCallDelta,
            Usage,
        )

        lane = text.startswith(_PREAMBLE[:40])
        for i in range(40):
            yield StreamTextDelta(delta="word ")
            await asyncio.sleep(0.004 if i % 5 == 4 else 0)
        usage = Usage(input_tokens=20_000 + done * 1500, output_tokens=40)
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


async def _child_main(args: argparse.Namespace) -> None:
    import local_operator
    from local_operator.harness.types import ModelSpec, ToolContext
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.tools.registry import create_tools

    directory = CONFIG_DIR / "sessions" / SESSION_ID
    if args.roster:
        _seed_roster(directory, args.roster)
    session = Session(
        model=ModelSpec(provider="test", model_id="bench", context_window=200_000),
        stream_fn=_PacedStream(args.lane_turns),
        tools=create_tools(ToolContext(cwd=str(directory)), enabled=["todo"]),
        transcript=Transcript(directory),
        system_blocks_provider=lambda *a, **k: ["bench"],
        cwd=str(directory),
        yolo=True,
    )
    await session.async_init()
    session.jobs.set_max_running(max(1, args.lanes))
    loop = asyncio.get_running_loop()
    server = RuntimeServer(ServingSessionHandle(session, loop, cwd=str(directory)), kind="daemon")
    server.start()
    await server.wait_until_published()
    (directory / ".session.pid").write_text(str(os.getpid()))
    for i in range(args.lanes):
        session._launch_subagent(label=f"lane-{i}", prompt=_PREAMBLE + f" lane {i} work")
    print(f"READY {local_operator.__file__}", flush=True)
    # The parent closes our stdin when it is done; it owns our lifetime.
    await loop.run_in_executor(None, sys.stdin.read)
    os._exit(0)  # the lanes are still stepping


# ---------------------------------------------------------------------------
# parent: the real TUI, attached
# ---------------------------------------------------------------------------


def _child_env() -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LOP_", "CMUX_")) and k != "XPC_FLAGS"
    }
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


async def _wait_record(timeout: float = 90.0) -> Any:
    from local_operator.session.runtime import registry

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for record, _state in registry.scan(CONFIG_DIR):
            if getattr(record, "session_id", "") == SESSION_ID:
                return record
        await asyncio.sleep(0.05)
    raise RuntimeError("runtime never published")


async def _parent_main(args: argparse.Namespace) -> dict[str, Any]:
    from textual import events

    import local_operator
    from local_operator.session.attached import AttachedSession
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.editor import Editor
    from local_operator.tui.widgets.transcript import UserBlock

    marks: dict[str, float] = {}
    original_mount = getattr(UserBlock, "_on_mount", None)

    def on_mount(self: Any, *a: Any, **kw: Any) -> Any:
        marks.setdefault("mount", time.perf_counter())
        if original_mount is not None:
            return original_mount(self, *a, **kw)
        return None

    UserBlock._on_mount = on_mount  # type: ignore[attr-defined]
    original_display = OperatorApp._display

    def display(self: Any, screen: Any, renderable: Any) -> Any:
        if "mount" in marks and "frame" not in marks:
            marks["frame"] = time.perf_counter()
        return original_display(self, screen, renderable)

    OperatorApp._display = display  # type: ignore[method-assign]
    original_submit = OperatorApp._submit_prompt

    def submit(self: Any, *a: Any, **kw: Any) -> Any:
        marks.setdefault("submit", time.perf_counter())
        return original_submit(self, *a, **kw)

    OperatorApp._submit_prompt = submit  # type: ignore[method-assign]
    calls: dict[str, list[float]] = {}

    def time_cpu(name: str) -> None:
        original = getattr(OperatorApp, name, None)
        if original is None:
            return

        def timed(self: Any, *a: Any, **kw: Any) -> Any:
            started = time.thread_time()
            try:
                return original(self, *a, **kw)
            finally:
                calls.setdefault(name, []).append((time.thread_time() - started) * 1000)

        setattr(OperatorApp, name, timed)

    for name in ("_apply_pending_frontend_state", "_poll_subagents", "_refresh_band"):
        time_cpu(name)

    child = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--repo",
            str(REPO),
            "--lanes",
            str(args.lanes),
            "--roster",
            str(args.roster),
            "--lane-turns",
            str(args.lane_turns),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=_child_env(),
    )
    rows: list[dict[str, int]] = []
    lags: list[float] = []
    try:
        assert child.stdout is not None
        line = (await asyncio.to_thread(child.stdout.readline)).decode().strip()
        if not line.startswith("READY"):
            raise RuntimeError(f"runtime child did not start: {line!r}")
        child_tree = line.split(" ", 1)[1]
        record = await _wait_record()

        async def never() -> Any:
            raise RuntimeError("the bench never takes over")

        viewer = await AttachedSession.connect(
            record, SESSION_ID, config_dir=CONFIG_DIR, takeover_factory=never
        )

        async def factory() -> Any:
            return viewer

        app = OperatorApp(factory)
        async with app.run_test(size=(120, 40)):
            await asyncio.sleep(args.settle)
            calls.clear()
            loop = asyncio.get_running_loop()

            async def heartbeat() -> None:
                while True:
                    due = loop.time() + 0.01
                    await asyncio.sleep(0.01)
                    lags.append(max(0.0, (loop.time() - due) * 1000))

            beat = asyncio.ensure_future(heartbeat())
            for n in range(args.probes):
                # Every probe is sent to an IDLE owner, so a probe measures the
                # echo rather than the prompt-vs-steer path of a busy one.
                while viewer.is_streaming:
                    await asyncio.sleep(0.05)
                await asyncio.sleep(args.gap)
                tag = f"probe-{n}"
                app.query_one(Editor).load_text(f"{tag} hello")
                await asyncio.sleep(0.2)
                marks.clear()
                started = time.perf_counter()
                key = events.Key("enter", "\r")
                key.set_sender(app)
                assert app._driver is not None
                app._driver.send_message(key)
                deadline = started + args.probe_timeout
                while "frame" not in marks and time.perf_counter() < deadline:
                    await asyncio.sleep(0.002)
                rows.append({k: round((v - started) * 1000) for k, v in marks.items()})
            beat.cancel()
            if child.stdin:
                child.stdin.close()
            await viewer.dispose()
    finally:
        if child.stdin and not child.stdin.closed:
            child.stdin.close()
        try:
            child.wait(30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    return {
        "tree": {"viewer": local_operator.__file__, "runtime": child_tree},
        "lanes": args.lanes,
        "roster": args.roster,
        "load": round(os.getloadavg()[0], 1),
        "echo_ms": {
            key: _pcts([row[key] for row in rows if key in row])
            for key in ("submit", "mount", "frame")
        },
        "missed": sum(1 for row in rows if "frame" not in row),
        "tui_loop_lag_ms": _pcts(lags),
        "call_cpu_ms": {
            name: {**_pcts(values), "total": round(sum(values), 1)}
            for name, values in calls.items()
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=str(REPO), help="checkout whose code to measure")
    parser.add_argument("--lanes", type=int, default=12)
    parser.add_argument("--roster", type=int, default=240)
    parser.add_argument("--lane-turns", type=int, default=400)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--gap", type=float, default=0.4, help="idle gap before each probe")
    parser.add_argument("--settle", type=float, default=4.0)
    parser.add_argument("--probe-timeout", type=float, default=60.0)
    parser.add_argument("--json", default="")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    import logging

    logging.disable(logging.WARNING)
    if args.child:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        asyncio.run(_child_main(args))
        return 0
    out = asyncio.run(_parent_main(args))
    brief = {key: value for key, value in out.items() if key != "rows"}
    print(json.dumps(brief, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
