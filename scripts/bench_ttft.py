"""Measure TIME TO FIRST TOKEN on the three paths a user actually lives on.

WHY THIS EXISTS
===============
The complaint is not throughput or total turn time — it is the pause between
hitting Enter and seeing the first streamed character. That pause has three
different shapes depending on which front end you are in, and they do not share
a fix:

* ``tui``      — in-process ``Session``: ``session.prompt()`` to the first
  ``text_delta`` reaching the stream callback. This is the floor: no HTTP, no
  runtime process, no bridge.
* ``desktop-cold`` — the Electron UI's first message in a session that has no
  runtime yet. The POST carries the cold engage (spawn a child, import the
  composition root, construct the session, publish a record, bind) before the
  turn can start. One honest qualification: the harness opens the SSE
  subscription and posts ``/watch`` before the measured message, and a visible
  watch lease arms a speculative warm of its own
  (``server/utils/desktop_sessions.py``), so this could race a spawn the
  harness started rather than always performing the engage inline. Both arms
  race the same way, so the comparison holds; the absolute number is "first
  message on a session somebody is already looking at", which is the real
  shape in the app.
* ``desktop-warm`` — the same UI on a session whose runtime is already up. The
  difference between this and ``desktop-cold`` is exactly what a speculative
  warm buys; the difference between this and ``tui`` is the cost the daemon
  plane adds.

WHAT IS FAKED, AND WHY THAT IS THE POINT
========================================
The PROVIDER is the built-in ``test`` mock: it answers with canned deltas and
no network at all. Everything measured here is therefore local-operator's own
overhead — imports, tokenizer, prompt construction, IPC, JSON — which is the
only part this repository can move. A live provider would bury the signal under
its own time-to-first-byte and make before/after incomparable. The mock is
production code (``providers/clients.py``), not a test double.

Everything else is real: real uvicorn daemon, real HTTP + SSE, real spawned
``python -m local_operator.session.runtime.process`` child, real transcript.

ISOLATION
=========
Every run gets a fresh ``HOME``, ``LOCAL_OPERATOR_CONFIG_DIR`` and ``TMPDIR``,
seeded with ``hosting: test``. The operator's live sessions are never touched
and a benchmark child can never attach to a real conversation.

USAGE
=====
    .venv/bin/python scripts/bench_ttft.py --runs 7
    .venv/bin/python scripts/bench_ttft.py --runs 7 --json out.json
    .venv/bin/python scripts/bench_ttft.py --scenario tui --runs 20

Report the MEDIAN. The first run in a process pays for cold page cache on the
interpreter and site-packages, and the distribution has a long right tail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, cast

# A benchmark under scripts/ must read the tree it lives in, not whatever tree
# the venv was installed from (AGENTS.md, "Every feature worktree owns its own
# venv"). Without this a benchmark run in a worktree silently measures main.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rig_safety import NO_NOTIFY_ENV, disable_notifications  # noqa: E402

#: The scenario names this benchmark reports. Kept in one place so the CLI
#: choices and the summary table cannot drift apart.
SCENARIOS = ("tui", "desktop-cold", "desktop-warm")

#: How long a single first-token wait may take before the run is called a
#: failure. Generous: a cold engage on a loaded machine plus a turn is well
#: under this, and a genuine hang deserves to be reported as one rather than
#: silently lengthening the median.
FRAME_TIMEOUT_S = 60.0

#: The bytecode-cache prefix every measured process runs under, set once per
#: invocation and shared by every run — the shape the desktop app creates, where
#: one cache under userData is read by the daemon and by every runtime child.
#: Set explicitly rather than inherited: the operator's shell may already carry
#: one pointing at the real app cache, and a benchmark that silently measured
#: that (warm, populated by unrelated runs) would report the wrong number.
_PYCACHE_PREFIX: Path | None = None

#: tiktoken downloads ``cl100k_base.tiktoken`` unless it finds the file, and it
#: looks in ``$TIKTOKEN_CACHE_DIR`` or else ``<TMPDIR>/data-gym-cache``. This
#: benchmark gives every run a fresh ``TMPDIR`` for isolation, which silently
#: turned the first use of the tokenizer into a TLS round trip — 413 ms of
#: ``SSLSocket.read``, 326 ms inside ``load_tiktoken_bpe`` and 115 ms in
#: ``getaddrinfo``, measured in a profile of the child. That is a property of
#: the harness, not of local-operator, so the DATA is pinned to one directory
#: per invocation the way an operator's machine pins it.
_TIKTOKEN_CACHE_DIR: Path | None = None


def _prime_bytecode_cache() -> None:
    """Populate the run's bytecode cache through the production entry point.

    Exists so "before" and "after" can be compared under the desktop app's real
    arrangement: an interpreter that REFUSES bytecode writes, and a cache that
    something long-lived populates once for the children that follow. Run as a
    subprocess because that is how the daemon does it. Absent on a tree without
    this module — reported, not fatal, so the same script can measure both arms.
    """
    env = dict(os.environ)
    env["LOP_TTFT_REPO"] = str(Path(__file__).resolve().parents[1])
    code = (
        "import os, sys\n"
        "sys.path.insert(0, os.environ['LOP_TTFT_REPO'])\n"
        "try:\n"
        "    from local_operator.bytecode import warm_bytecode_cache_in_background\n"
        "except ImportError:\n"
        "    print('NO_MODULE')\n"
        "    raise SystemExit(0)\n"
        "thread = warm_bytecode_cache_in_background()\n"
        "if thread is None:\n"
        "    print('ALREADY_WARM')\n"
        "else:\n"
        "    thread.join(600)\n"
        "    print('PRIMED')\n"
    )
    completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, "-c", code],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=900,
        check=False,
    )
    print(f"  bytecode prime: {completed.stdout.decode().strip() or 'FAILED'}", flush=True)


def _seed_config(config_dir: Path) -> None:
    """Write the minimum config a runtime needs to construct.

    Through ``ConfigManager`` rather than hand-rolled YAML: the metadata block
    carries fields the loader requires, and a hand-written file raises a
    ``KeyError`` from inside the child that reads like a startup regression.
    """
    from local_operator.config import ConfigManager

    config_dir.mkdir(parents=True, exist_ok=True)
    manager = ConfigManager(config_dir=config_dir)
    manager.update_config({"hosting": "test", "model_name": "mock"})


def _kill_children(config_dir: Path) -> None:
    """Terminate any runtime this run spawned.

    A benchmark that leaves runtimes resident measures its later runs against a
    machine it degraded itself — and worse, a leftover child keeps the session's
    lease, so the next run's engage would find a live owner and report a
    suspiciously fast cold start.
    """
    from local_operator.session.runtime import registry

    for record, _state in registry.scan(config_dir):
        pid = getattr(record, "pid", None)
        if isinstance(pid, int) and pid > 0 and pid != os.getpid():
            try:
                os.kill(pid, 15)
            except (ProcessLookupError, PermissionError, OSError):
                pass


# ---------------------------------------------------------------------------
# Scenario 1: in-process session (the TUI)
#
# Run in a CHILD PROCESS, one process per measurement, because the cost this
# scenario exists to expose is per-process: a TUI is a single long-lived
# process, so its FIRST turn ever is the only one that pays the cold caches,
# and an in-process loop would measure the second turn six times out of seven
# and report a settled number for a cost the user meets once. Each run also
# takes a SECOND turn, so the report carries both the cold-process number and
# the steady-state number the same process reaches once warm.
# ---------------------------------------------------------------------------


async def _tui_child(config_dir: Path, cwd: Path) -> dict[str, float]:
    """Run two turns in THIS process and report each one's first-token time."""
    import argparse as _argparse

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.harness.types import StreamTextDelta
    from local_operator.session_factory import create_session, warm_session_imports

    # EXACTLY WHAT THE TUI DOES FIRST, and the reason it is here: the real app
    # runs this off-loop at boot (``tui/app.py``), and it is the seam that now
    # carries the tokenizer warm. A benchmark that skipped it would charge the
    # tokenizer to the first turn — production does not — and would then report
    # the TUI as unchanged by a change that specifically moves that cost to
    # boot.
    await asyncio.to_thread(warm_session_imports)

    args = _argparse.Namespace()
    for key, value in {
        "hosting": "test",
        "model": "mock",
        "agent_name": None,
        "agent_id": None,
        "yolo": True,
        "train": False,
        "resume": None,
        "agent": None,
    }.items():
        setattr(args, key, value)

    session = await create_session(
        args,
        ConfigManager(config_dir),
        CredentialManager(config_dir),
        AgentRegistry(config_dir),
        cwd=str(cwd),
    )
    marks: dict[str, float] = {}
    try:
        inner = cast(Any, session)._stream_fn

        class TimedStream:
            """Wrap the session's stream fn to timestamp the first delta."""

            def __init__(self, wrapped: Any, base: float, label: str) -> None:
                self._wrapped = wrapped
                self._base = base
                self._label = label

            def __getattr__(self, name: str) -> Any:
                return getattr(self._wrapped, name)

            def __call__(self, request: Any, signal: Any = None) -> Any:
                marks[self._label + "request_built_ms"] = (time.perf_counter() - self._base) * 1000
                source = self._wrapped(request, signal)

                async def relay() -> Any:
                    first = self._label + "first_delta_ms"
                    async for event in source:
                        if isinstance(event, StreamTextDelta) and first not in marks:
                            marks[first] = (time.perf_counter() - self._base) * 1000
                        yield event

                return relay()

        for label in ("cold_", "warm_"):
            base = time.perf_counter()
            cast(Any, session)._stream_fn = TimedStream(inner, base, label)
            await session.prompt("Reply with one short sentence.")
            marks[label + "turn_ms"] = (time.perf_counter() - base) * 1000
        return marks
    finally:
        await session.dispose()


async def _run_tui_child(config_dir: Path, cwd: Path) -> dict[str, float]:
    """Run :func:`_tui_child` in a fresh interpreter and parse its report."""
    env = dict(os.environ)
    env["HOME"] = str(config_dir.parent)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    env["TMPDIR"] = str(config_dir.parent)
    env["LOP_TTFT_CHILD"] = "1"
    env["LOP_TTFT_CWD"] = str(cwd)
    # The child boots the real TUI (``--child-tui``), so it is a notification
    # surface: gated explicitly, not inherited.
    env.update(NO_NOTIFY_ENV)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-tui",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    out, err = await proc.communicate()
    for line in out.decode().splitlines():
        if line.startswith("TTFT_JSON "):
            return json.loads(line[len("TTFT_JSON ") :])
    raise AssertionError(f"child reported no timing: {err.decode()[-2000:]}")


# ---------------------------------------------------------------------------
# Scenario 2/3: the desktop plane, over real HTTP + SSE
# ---------------------------------------------------------------------------


def _bind_listener() -> socket.socket:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    return listener


async def _serve(address: socket.socket) -> Any:
    """Start the real daemon app on ``address``; await readiness."""
    import uvicorn

    from local_operator.server.app import app

    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[address]))
    for _ in range(100000):
        if server.started:
            break
        if task.done():
            await task
        await asyncio.sleep(0)
    assert server.started, "daemon did not start"
    return server, task


async def _next_frame(
    lines: Any, predicate: Any, timeout: float = FRAME_TIMEOUT_S
) -> dict[str, Any]:
    """Read SSE frames until one satisfies ``predicate``."""

    async def read() -> dict[str, Any]:
        async for line in lines:
            if line.startswith("data: "):
                frame = json.loads(line[6:])
                if predicate(frame):
                    return frame
        raise AssertionError("stream ended before the expected frame")

    return await asyncio.wait_for(read(), timeout)


def _is_text(frame: dict[str, Any]) -> bool:
    """The first frame that carries streamed assistant text.

    The desktop wire projects streamed text as ``message_update`` frames with a
    non-empty ``delta`` (``AgentEventBridge``), NOT as the harness's
    ``text_delta`` stream event — the renderer appends ``delta`` as it arrives,
    so this is the frame the user is waiting for.
    """
    payload = frame.get("payload", {})
    return (
        frame.get("type") == "event"
        and payload.get("type") == "message_update"
        and bool(payload.get("delta"))
    )


async def _run_desktop(config_dir: Path, cwd: Path, *, warm: bool) -> dict[str, float]:
    """One message through the desktop HTTP API, to the first streamed token."""
    import httpx

    address = _bind_listener()
    server, task = await _serve(address)
    base_url = f"http://127.0.0.1:{address.getsockname()[1]}"
    headers = {"Authorization": "Bearer " + os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"]}
    marks: dict[str, float] = {}
    try:
        async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=120) as client:
            created = await client.post(
                "/v1/desktop/sessions",
                json={"request_id": str(uuid.uuid4()), "cwd": str(cwd)},
            )
            created.raise_for_status()
            session_id = created.json()["result"]["session_id"]
            target = f"/v1/desktop/sessions/{session_id}"

            async with client.stream("GET", target + "/events") as response:
                lines = response.aiter_lines()
                opened = await _next_frame(lines, lambda f: f["type"] == "open")
                await _next_frame(lines, lambda f: f["type"] == "snapshot")
                await client.post(
                    target + "/watch",
                    json={
                        "subscription_id": opened["payload"]["subscription_id"],
                        "visible": True,
                        "can_notify": False,
                    },
                )

                if warm:
                    # Produce one complete turn first, so the measured turn runs
                    # on exactly the state a user's SECOND message finds: a live
                    # runtime, a bound bridge, a populated transcript.
                    priming = time.perf_counter()
                    await client.post(
                        target + "/messages",
                        json={"request_id": str(uuid.uuid4()), "text": "Priming turn."},
                    )
                    await _next_frame(
                        lines,
                        lambda f: f["type"] == "event"
                        and f.get("payload", {}).get("type") == "agent_end",
                    )
                    marks["prime_ms"] = (time.perf_counter() - priming) * 1000
                start = time.perf_counter()
                admitted = await client.post(
                    target + "/messages",
                    json={
                        "request_id": str(uuid.uuid4()),
                        "text": "Reply with one short sentence.",
                    },
                )
                admitted.raise_for_status()
                marks["admitted_ms"] = (time.perf_counter() - start) * 1000
                await _next_frame(lines, _is_text)
                marks["first_token_ms"] = (time.perf_counter() - start) * 1000
        return marks
    finally:
        server.should_exit = True
        with_stop = asyncio.wait_for(task, 30)
        try:
            await with_stop
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 — teardown is best effort
            pass
        address.close()
        _kill_children(config_dir)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _isolated_root() -> tuple[Path, Path, Path]:
    root = Path(tempfile.mkdtemp(prefix="lop-ttft-"))
    config_dir = root / ".local-operator"
    cwd = root / "workspace"
    cwd.mkdir(parents=True, exist_ok=True)
    return root, config_dir, cwd


async def _one(scenario: str, index: int) -> dict[str, float]:
    root, config_dir, cwd = _isolated_root()
    _seed_config(config_dir)
    saved = {
        key: os.environ.get(key)
        for key in (
            "HOME",
            "LOCAL_OPERATOR_CONFIG_DIR",
            "TMPDIR",
            "LOCAL_OPERATOR_DESKTOP_TOKEN",
        )
    }
    os.environ["HOME"] = str(root)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    os.environ["TMPDIR"] = str(root)
    os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"] = secrets.token_hex(32)
    # The desktop scenarios drive a real ``lop serve`` whose machine-wide feed
    # raises banners, and the children inherit THIS environment — so the gate is
    # set here, once per run, rather than only in the child mappings below.
    disable_notifications()
    if _TIKTOKEN_CACHE_DIR is not None:
        os.environ["TIKTOKEN_CACHE_DIR"] = str(_TIKTOKEN_CACHE_DIR)
    try:
        if scenario == "tui":
            marks = await _run_tui_child(config_dir, cwd)
        else:
            marks = await _run_desktop(config_dir, cwd, warm=scenario == "desktop-warm")
        marks["run"] = float(index)
        return marks
    finally:
        _kill_children(config_dir)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(root, ignore_errors=True)


async def _run_scenario(scenario: str, runs: int) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for index in range(runs):
        result = await _one(scenario, index)
        results.append(result)
        shown = " ".join(
            f"{key}={value:.0f}ms" for key, value in result.items() if key.endswith("_ms")
        )
        print(f"  {scenario} run {index + 1}/{runs}: {shown}", flush=True)
    return results


def _summarize(results: list[dict[str, float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"runs": len(results)}
    keys = sorted({key for result in results for key in result if key.endswith("_ms")})
    for key in keys:
        values = [result[key] for result in results if key in result]
        if not values:
            continue
        summary[key] = {
            "median": round(statistics.median(values), 1),
            "min": round(min(values), 1),
            "max": round(max(values), 1),
        }
    return summary


async def _amain(args: argparse.Namespace) -> int:
    scenarios = [args.scenario] if args.scenario else list(SCENARIOS)
    report: dict[str, Any] = {}
    if args.prime_bytecode:
        _prime_bytecode_cache()
    for scenario in scenarios:
        print(f"\n=== {scenario} ({args.runs} runs) ===", flush=True)
        results = await _run_scenario(scenario, args.runs)
        summary = _summarize(results)
        report[scenario] = {"summary": summary, "results": results}
        print(f"--- {scenario} median ---")
        for key, stat in summary.items():
            if not isinstance(stat, dict):
                continue
            print(
                f"  {key:<18} median={stat['median']:>8}  "
                f"min={stat['min']:>8}  max={stat['max']:>8}"
            )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=7, help="runs per scenario")
    parser.add_argument("--scenario", choices=SCENARIOS, default="", help="one scenario only")
    parser.add_argument("--json", type=str, default="", help="write raw results here")
    parser.add_argument(
        "--pycache-prefix",
        type=str,
        default="",
        help="bytecode cache every process runs under (default: one temp dir per invocation)",
    )
    parser.add_argument(
        "--prime-bytecode",
        action="store_true",
        help="populate that cache once before measuring, as the daemon does",
    )
    parser.add_argument(
        "--child-tui",
        action="store_true",
        help="(internal) run one TUI measurement in this process and report JSON",
    )
    args = parser.parse_args()
    global _PYCACHE_PREFIX
    _PYCACHE_PREFIX = Path(args.pycache_prefix or tempfile.mkdtemp(prefix="lop-ttft-pycache-"))
    _PYCACHE_PREFIX.mkdir(parents=True, exist_ok=True)
    global _TIKTOKEN_CACHE_DIR
    _TIKTOKEN_CACHE_DIR = Path(tempfile.mkdtemp(prefix="lop-ttft-tiktoken-"))
    # FORCED for the whole invocation, not just the measured runs — and set HERE
    # rather than in ``_one`` because the priming pass below builds its
    # environment from ``os.environ``: a prime without the tokenizer's data
    # directory warms tiktoken into the ambient cache while the measured child
    # fetches it afresh, which measures the harness's network rather than
    # local-operator.
    os.environ["TIKTOKEN_CACHE_DIR"] = str(_TIKTOKEN_CACHE_DIR)
    os.environ["PYTHONPYCACHEPREFIX"] = str(_PYCACHE_PREFIX)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    print(
        f"bytecode prefix: {_PYCACHE_PREFIX}\n"
        f"tokenizer cache: {_TIKTOKEN_CACHE_DIR}\n"
        f"sys.dont_write_bytecode in this process: {sys.dont_write_bytecode} "
        f"(the arms' CHILDREN always get 1, set in _one)",
        flush=True,
    )
    if args.child_tui:
        config_dir = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
        cwd = Path(os.environ["LOP_TTFT_CWD"])
        marks = asyncio.run(_tui_child(config_dir, cwd))
        print("TTFT_JSON " + json.dumps(marks))
        return 0
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
