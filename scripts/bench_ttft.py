"""Measure TIME TO FIRST TOKEN on the four paths a user actually lives on.

WHY THIS EXISTS
===============
The complaint is not throughput or total turn time — it is the pause between
hitting Enter and seeing the first streamed character. That pause has four
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
* ``sse-jobs`` — ``POST /v1/chat/async`` and its ``/v1/sse/jobs/{id}`` stream:
  the route answers a job id at once and the turn runs in a SPAWNED CHILD, so
  the measured wait is the client's, from submit to the first frame the job
  stream carries about that job.

WHAT IS ASSERTED
================
Every run is also a GATE (see ``GATES``): the acknowledgement and the warm
paths must reach the client inside 300 ms of the submit, a cold submit must
produce exactly ONE acknowledgement, and a duplicate submit under the same id
must replay its receipt rather than ack a second time. The before-arm of such a
change must FAIL these gates, which is why a mark that never appears is a
failure rather than a skip.

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

Report the MEDIAN, and read the p95/p99 beside it. The first run in a process
pays for cold page cache on the interpreter and site-packages, and the
distribution has a long right tail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
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

from local_operator.agent_shell import harness_child_env  # noqa: E402

#: The scenario names this benchmark reports. Kept in one place so the CLI
#: choices and the summary table cannot drift apart.
SCENARIOS = ("tui", "desktop-cold", "desktop-warm", "sse-jobs")

#: How long a single first-token wait may take before the run is called a
#: failure. Generous: a cold engage on a loaded machine plus a turn is well
#: under this, and a genuine hang deserves to be reported as one rather than
#: silently lengthening the median.
FRAME_TIMEOUT_S = 60.0

#: The submission ACKNOWLEDGEMENTS these gates are about, as a hard ceiling in ms
#: from the submit: the operator's acceptance bar for "the client is told the
#: message was accepted", and the reference point the redesign was measured
#: against. Kept as ONE number so the assertion and the sentence cannot drift.
#:
#: IT IS A CONTRACT, NOT A CALIBRATED BOUND, and the distinction matters here.
#: The work these gates cover is the host's own acknowledgement — receipt claim
#: plus one bridge publish, tens of milliseconds, measured — while the thing it
#: must not wait for (the cold engage) is 2.6-9.5 s on this box. Any ceiling in
#: between discriminates the defect, so a number fitted to an observation would
#: add nothing but the risk AGENTS.md warns about ("Calibrate ceilings from CI,
#: never from your laptop").
SUBMIT_TO_ACK_CEILING_MS = 300.0

#: The ceiling that actually holds on a busy host, as the p95 of the same mark.
#:
#: WHY THERE ARE TWO, and why the second is not a loosened first: the operator's
#: requirement is 300 ms and it is met where it is measured — the MEDIAN, which
#: sits at 60-75 ms on this box even under load. The TAIL does not stay inside
#: 300 ms and pretending it does would be the harness asserting a bound the
#: measurement contradicts: across the passes recorded on the PR the ack's p95
#: was 148 ms, 219 ms and 326 ms at load 100-220, and QA's independent matrix saw
#: one cold submit of 333 ms in seven at load 140-190. Half a second is the bound
#: those numbers support, stated with the load they were taken at, and the median
#: gate above remains the contract's own number.
#:
#: A run that fails THIS gate is a real reading about the host, not a flake to
#: re-run away: report the load beside it, as the PR's table does.
SUBMIT_TO_ACK_P95_CEILING_MS = 500.0

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
    # The child boots the real TUI (``--child-tui``), so it is a harness child:
    # ``harness_child_env`` declares the script a harness and carries the
    # notification gate with it (`agent_shell.harness_child_env`), which is what
    # keeps this child's completions out of the operator's Notification Centre.
    env = harness_child_env(env)
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


#: The session-stream frame that acknowledges a submit BEFORE the runtime is
#: engaged (``routes/desktop_sessions.py::ADMISSION_ACCEPTED_FRAME``). This is
#: the frame ``submit_to_ack_ms`` is measured to, and the one the ``desktop-cold``
#: gate is about: it is the first thing the runtime says about the request, and it
#: is emitted by the host while the engage is still starting.
ADMISSION_ACCEPTED = "admission.accepted"


async def _read_until(
    lines: Any,
    start: float,
    stop: Any,
    marks: dict[str, float],
    *,
    stop_name: str,
    timeout: float = FRAME_TIMEOUT_S,
) -> None:
    """Read frames CONCURRENTLY with the POST, timing the ones that matter.

    Reading this way rather than awaiting the response first is the whole point:
    a frame's arrival time is only observable while the response is still in
    flight, and awaiting the POST first stamps every frame with the response's
    own timestamp — which is the wait being measured, so the acknowledgement
    would be reported as arriving no earlier than the engage it exists to
    precede.

    ``submit_to_ack_ms`` is the frame the host emits on acceptance (absent on a
    tree that never emits it, which is the defect this benchmark gates on),
    ``submit_to_first_frame_ms`` the first non-heartbeat frame of ANY kind,
    ``submit_to_token_ms`` the first streamed assistant text, and ``stop_name``
    the frame that ends the read. Heartbeats are skipped on purpose: they say the
    transport is alive, never that work was accepted, so counting one as the
    first frame would let a stalled engage read as a healthy one. ``acks`` counts
    the acknowledgements over the whole turn, because exactly one submit must
    produce exactly one.
    """

    async def read() -> None:
        async for line in lines:
            if not line.startswith("data: "):
                continue
            frame = json.loads(line[6:])
            if frame.get("type") == "heartbeat":
                continue
            if "submit_to_first_frame_ms" not in marks:
                marks["submit_to_first_frame_ms"] = (time.perf_counter() - start) * 1000
            if frame.get("type") == ADMISSION_ACCEPTED:
                marks["acks"] = marks.get("acks", 0.0) + 1
                if "submit_to_ack_ms" not in marks:
                    marks["submit_to_ack_ms"] = (time.perf_counter() - start) * 1000
            if _is_text(frame) and "submit_to_token_ms" not in marks:
                marks["submit_to_token_ms"] = (time.perf_counter() - start) * 1000
            if stop(frame):
                marks[stop_name] = (time.perf_counter() - start) * 1000
                return
        raise AssertionError("stream ended before the expected frame")

    await asyncio.wait_for(read(), timeout)


#: Frames that end a measured turn: the whole run is over when ``agent_end``
#: lands, which is what lets the reader count every acknowledgement the submit
#: produced rather than stopping at the first one.
def _is_agent_end(frame: dict[str, Any]) -> bool:
    return frame.get("type") == "event" and frame.get("payload", {}).get("type") == "agent_end"


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
                request_id = str(uuid.uuid4())
                body = {"request_id": request_id, "text": "Reply with one short sentence."}
                start = time.perf_counter()
                # CONCURRENT with the read, deliberately: see _read_until. The read
                # runs to agent_end so the ack count covers the whole turn — a
                # duplicate submit that re-emitted an ack would show up as 2.
                reader = asyncio.create_task(
                    _read_until(lines, start, _is_agent_end, marks, stop_name="turn_end_ms")
                )
                admitted = await client.post(target + "/messages", json=body)
                admitted.raise_for_status()
                marks["submit_to_response_ms"] = (time.perf_counter() - start) * 1000

                # A SECOND submit under the SAME id must replay the receipt: no
                # second ack, no second turn. Measured on this plane rather than
                # only asserted in a unit test because this is where the duplicate
                # travels, and declared BEFORE the reader finishes so the count
                # below covers the frames the replay itself could produce.
                replay = await client.post(target + "/messages", json=body)
                replay.raise_for_status()
                marks["duplicate_replayed"] = 1.0 if replay.json()["result"]["replayed"] else 0.0
                await reader
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
# Scenario 4: the async job stream (`POST /v1/chat/async` -> `/v1/sse/jobs/{id}`)
#
# The second submission surface, and the one the diagnosis could not measure.
# The async route answers a job id immediately and hands the work to a SPAWNED
# CHILD process, so the interesting wait is not the POST's own latency — that is
# milliseconds — but how long a client attached to the job stream waits before
# the stream says anything about THIS job. Nothing is published on the job
# channel at accept time, so the first job-scoped frame is whatever the child
# emits once its interpreter has booted and imported the composition root: the
# same invisible wait the desktop edge has, one process boundary further out.
# ---------------------------------------------------------------------------


async def _run_sse_jobs(config_dir: Path, cwd: Path) -> dict[str, float]:
    """One async submit, timed on the job stream two ways.

    TWO MARKS, because on this transport they are different questions and only
    one of them is the change's:

    * ``submit_to_live_job_event_ms`` — a cursor-less attach, i.e. what a client
      that simply opens ``GET /v1/sse/jobs/{id}`` after the response sees. That
      attach starts LIVE (the broker replays only when a cursor is supplied), so
      it is bound by whatever the CHILD emits once it has booted. Reported, not
      gated: it is the child's own startup on the critical path, not an
      acknowledgement gate — the 202 already told this client its job was
      accepted, in ``submit_to_response_ms``.
    * ``submit_to_job_event_ms`` — the same attach asking for the channel from
      its beginning (``after_seq=0``, the documented way to say that). The
      acknowledgement the route publishes at accept is retained by the broker,
      so this client receives it instead of waiting for the child.
    * ``response_to_job_event_ms`` — that mark minus ``submit_to_response_ms``,
      i.e. the wait that starts when the client can attach at all, and the one
      THE GATE is about. The end-to-end pair is reported beside it rather than
      gated, because this route's POST creates a job and starts a child process
      before it answers, so its own latency tracks the machine's load and not
      this change.
    """
    import httpx

    del cwd  # the async route has no working-directory concept
    address = _bind_listener()
    server, task = await _serve(address)
    base_url = f"http://127.0.0.1:{address.getsockname()[1]}"
    headers = {"Authorization": "Bearer " + os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"]}
    marks: dict[str, float] = {}
    try:
        async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=120) as client:
            start = time.perf_counter()
            submitted = await client.post(
                "/v1/chat/async",
                json={
                    "prompt": "Reply with one short sentence.",
                    "hosting": "test",
                    "model": "mock",
                },
            )
            submitted.raise_for_status()
            job_id = submitted.json()["result"]["id"]
            marks["submit_to_response_ms"] = (time.perf_counter() - start) * 1000

            # BOTH ATTACHES AT ONCE, deliberately: a cursor-less attach is bound by
            # whatever the child emits, and a sequential pair would charge the
            # replayed one for the live one's whole wait. Each is a real client
            # shape: the first simply opens the stream, the second asks for the
            # channel from its beginning (``after_seq=0``), and the difference
            # between them is exactly what the retained acknowledgement is worth.
            async def read_live() -> None:
                async with client.stream("GET", f"/v1/sse/jobs/{job_id}") as response:
                    await _read_job_frames(
                        response.aiter_lines(),
                        start,
                        marks,
                        mark="submit_to_live_job_event_ms",
                        stop_first=True,
                    )

            async def read_from_start() -> None:
                async with client.stream(
                    "GET", f"/v1/sse/jobs/{job_id}", params={"after_seq": 0}
                ) as response:
                    await _read_job_frames(
                        response.aiter_lines(), start, marks, mark="submit_to_job_event_ms"
                    )

            await asyncio.gather(read_live(), read_from_start())
            # THE DELTA THE CHANGE IS ABOUT, computed per run rather than by
            # subtracting two medians: the client cannot ask for this channel
            # before the response hands it the id, so the wait this frame removes
            # starts when that response lands. The end-to-end mark is kept and
            # reported beside it, because the POST's own latency (job creation
            # plus ``Process.start``) is a real part of what a caller waits for —
            # it is simply not a part this change moves, and a gate that
            # swallowed it would be measuring the box's load instead.
            marks["response_to_job_event_ms"] = (
                marks["submit_to_job_event_ms"] - marks["submit_to_response_ms"]
            )
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


async def _read_job_frames(
    lines: Any,
    start: float,
    marks: dict[str, float],
    *,
    mark: str,
    stop_first: bool = False,
) -> None:
    """Time the first frame about THIS job, then read out the stream.

    ``open`` and the connect comment are transport metadata: they exist whether
    or not a job does, so the metric is the first frame that says something
    about this job (``job.status``, a delta, a tool trace). Heartbeats are
    skipped for the same reason. The read then runs to the terminal frame, so a
    run also proves the job itself completed rather than only that it was
    acknowledged — and a turn the change broke cannot pass as an ack that
    arrived quickly.
    """

    async def read() -> None:
        async for line in lines:
            if not line.startswith("data: "):
                continue
            frame = json.loads(line[6:])
            kind = frame.get("type")
            if kind in ("open", "keepalive", "stream.open"):
                continue
            if mark not in marks:
                marks[mark] = (time.perf_counter() - start) * 1000
                if stop_first:
                    return
            if kind == "stream.terminal":
                marks["submit_to_terminal_ms"] = (time.perf_counter() - start) * 1000
                return
        if mark not in marks:
            raise AssertionError("job stream ended without a frame about the job")

    await asyncio.wait_for(read(), FRAME_TIMEOUT_S)


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
    from local_operator.tui.notify import suppress_notifications_for_process

    suppress_notifications_for_process("ttft benchmark driving the real CLI")
    if _TIKTOKEN_CACHE_DIR is not None:
        os.environ["TIKTOKEN_CACHE_DIR"] = str(_TIKTOKEN_CACHE_DIR)
    try:
        if scenario == "tui":
            marks = await _run_tui_child(config_dir, cwd)
        elif scenario == "sse-jobs":
            marks = await _run_sse_jobs(config_dir, cwd)
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


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile, the convention every other number here uses.

    Nearest rank rather than an interpolating estimator because the question the
    gates ask is "how slow is the slow run", and interpolation between two
    samples invents a value no run produced. With the small run counts this
    script is used at (7-20), the two agree to within a sample.
    """
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _summarize(results: list[dict[str, float]]) -> dict[str, Any]:
    """Median and the two tail percentiles for every mark the runs produced.

    EVERY mark, not only the ``_ms`` ones: the counters this harness asserts on
    (``acks`` must be exactly 1, ``duplicate_replayed`` must be 1) are marks, and
    a summary that dropped them would leave the gates reading a value the JSON
    does not carry. ``run`` is dropped because it is an index, not a measurement.
    """
    summary: dict[str, Any] = {"runs": len(results)}
    keys = sorted({key for result in results for key in result if key != "run"})
    for key in keys:
        values = [result[key] for result in results if key in result]
        if not values:
            continue
        summary[key] = {
            "median": round(statistics.median(values), 1),
            "p95": round(_percentile(values, 0.95), 1),
            "p99": round(_percentile(values, 0.99), 1),
            "min": round(min(values), 1),
            "max": round(max(values), 1),
        }
    return summary


#: The acceptance gates: ``(scenario, mark, percentile, comparison, ceiling)``.
#: WHY A GATE AND NOT A SENTENCE — a number in a PR description is a claim, and
#: the whole point of this change is a latency the harness can refute. The median
#: ceiling is the operator's own bar ("under 300 ms from submit to the first event
#: emitted by the runtime and the front end") rather than a figure fitted to an
#: observation; the p95 ceilings are the TOLERANCE that holds on a loaded host and
#: are named as such (see ``SUBMIT_TO_ACK_P95_CEILING_MS``). Every run's load
#: average is recorded in the report beside them, because a latency number without
#: its host is not evidence.
#:
#: The MAXIMUM is deliberately not gated: this box runs ~25 concurrent sessions,
#: the engage is not on any of these paths, and a ceiling tuned to the worst run
#: of one afternoon is the mis-calibration AGENTS.md warns about ("Calibrate
#: ceilings from CI, never from your laptop"). `desktop-cold`'s first TOKEN is
#: reported and never gated: its floor IS the engage.
GATES: tuple[tuple[str, str, str, str, float], ...] = (
    ("tui", "warm_first_delta_ms", "median", "<", SUBMIT_TO_ACK_CEILING_MS),
    ("desktop-warm", "submit_to_token_ms", "median", "<", SUBMIT_TO_ACK_CEILING_MS),
    ("desktop-cold", "submit_to_ack_ms", "median", "<", SUBMIT_TO_ACK_CEILING_MS),
    # The tail gate, on the same mark: the median says the acknowledgement is
    # where the contract puts it, and this says how far the slowest run in seven
    # may drift on a loaded host (see SUBMIT_TO_ACK_P95_CEILING_MS).
    ("desktop-cold", "submit_to_ack_ms", "p95", "<", SUBMIT_TO_ACK_P95_CEILING_MS),
    ("desktop-cold", "acks", "median", "==", 1.0),
    ("desktop-cold", "duplicate_replayed", "median", "==", 1.0),
    ("sse-jobs", "response_to_job_event_ms", "median", "<", SUBMIT_TO_ACK_CEILING_MS),
    ("sse-jobs", "response_to_job_event_ms", "p95", "<", SUBMIT_TO_ACK_P95_CEILING_MS),
)


def _evaluate_gates(report: dict[str, Any]) -> list[str]:
    """Check every gate against the run's own summary; return the failures.

    ONLY THE SCENARIOS THAT RAN are evaluated, so ``--scenario tui`` is a useful
    command rather than four spurious failures about marks nothing measured.
    Within a scenario that DID run, a mark that is ABSENT is a failure rather
    than a skip, and that is the whole reason the cold gate can discriminate: on
    a tree that never emits the acknowledgement there is no number to compare,
    and "no acknowledgement" is exactly the defect. A gate quietly skipped for a
    missing key would pass on the before-arm — the arm it exists to fail.
    """
    failures: list[str] = []
    # Scenarios only: the report also carries environment facts (the load the
    # run was taken at), and a gate is never asked for one.
    ran = set(report) & set(SCENARIOS)
    print("\n=== gates ===")
    for scenario, mark, percentile, comparison, ceiling in GATES:
        if scenario not in ran:
            continue
        section = report.get(scenario, {})
        summary = section.get("summary", {})
        stat = summary.get(mark)
        value = stat.get(percentile) if isinstance(stat, dict) else None
        label = f"{scenario}.{mark}[{percentile}]"
        if value is None:
            failures.append(f"{label}: NEVER OBSERVED (gate {comparison} {ceiling:g})")
            print(f"  FAIL  {label}: never observed (gate {comparison} {ceiling:g})")
            continue
        numeric = float(cast(float, value))
        ok = numeric < ceiling if comparison == "<" else numeric == ceiling
        line = f"  {'PASS' if ok else 'FAIL'}  {label}: {numeric:g} {comparison} {ceiling:g}"
        print(line)
        if not ok:
            failures.append(f"{label}: {numeric:g} {comparison} {ceiling:g}")
    return failures


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
                f"  {key:<26} median={stat['median']:>9}  p95={stat['p95']:>9}  "
                f"p99={stat['p99']:>9}  min={stat['min']:>9}  max={stat['max']:>9}"
            )
    load = os.getloadavg()
    report["load_average"] = {
        "1m": round(load[0], 2),
        "5m": round(load[1], 2),
        "15m": round(load[2], 2),
    }
    print(f"\nload average: {load[0]:.1f} {load[1]:.1f} {load[2]:.1f}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    failures = _evaluate_gates(report)
    if failures:
        print("\nGATE FAILED: " + "; ".join(failures))
        return 1
    print("\nall gates passed")
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
