"""End to end: does a busy multi-subagent roster keep its loop answerable?

The operator's symptom is a LIVE session that every surface reports as wedged: the
TUI refuses to resume it and the desktop cannot send to it. The mechanism is that
the record's heartbeat is written by an asyncio task *on the same loop* the roster
pump is saturating, so loop work that scales with the retained window turns into
minutes of silence while the process is perfectly alive.

This harness stands the real parts up in one process:

* a real record published through ``registry.publish`` into an ISOLATED config
  dir, heartbeated by a task shaped exactly like ``RuntimeServer._heartbeat_loop``
  (``await sleep(HEARTBEAT_INTERVAL_S)`` then write);
* a real ``asyncio.start_server`` control port that accepts an auth frame and
  answers a welcome, both halves loop callbacks -- so a starved loop delays the
  WELCOME with the kernel handshake already complete;
* the repaint pump, faithfully: a 50 ms ``loop.call_later`` tick calling the real
  ``FrontendStateStore.refresh_jobs`` over 5 subagent rows at the 500-row cap,
  on the real code path;
* a sibling thread that reads the record every second the way ``registry.scan``
  does, and one that dials the port the way the attach path does.

Run the same file against both trees (this is the A/B; the first line names the
tree that actually ran):

  cd /tmp && env -u CMUX_WORKSPACE_ID -u CMUX_SESSION_ID -u CMUX_SURFACE_ID \
    -u LOP_MOBILE_CHILD_PROVIDER -u LOP_RUNTIME_ADOPT_SESSION \
    PYTHONPATH=~/local-operator-worktrees/<tree> \
    ~/local-operator-worktrees/frame-cost-loop-starvation/.venv/bin/python \
    docs/evidence/frame-cost-loop-starvation/loop_responsiveness_demo.py

Nothing of the operator's is touched: an isolated ``LOCAL_OPERATOR_CONFIG_DIR``,
a synthetic session id, no TUI, no fork, no signal to any live process.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

CFG = Path(os.environ.get("DEMO_CONFIG_DIR", "/tmp/frame-cost-demo-cfg"))
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(CFG)

from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY as SEQ  # noqa: E402
from local_operator.session import frontend_state as fs  # noqa: E402
from local_operator.session.runtime import registry as reg  # noqa: E402
from local_operator.session.runtime.types import (  # noqa: E402
    HEARTBEAT_INTERVAL_S,
    HEARTBEAT_TIMEOUT_S,
    SessionRecord,
)

T0 = time.time()
SESSION_ID = "framecostdemo01"
JOBS = 5
ROWS = 500
ROW_TEXT_BYTES = 8192
TICK_S = 0.05
DIAL_WELCOME_BUDGET_S = 12.0
RUN_S = float(os.environ.get("DEMO_RUN_S", "60"))
STREAM = os.environ.get("DEMO_STREAM", "1") == "1"


def make_row(i: int) -> dict[str, Any]:
    return {
        "type": "tool_execution_end",
        SEQ: i,
        "tool_call_id": f"call_{i:08d}",
        "tool_name": "bash",
        "intent": "run a command and read what it printed",
        "result": {
            "content": [{"type": "text", "text": "x" * ROW_TEXT_BYTES}],
            "details": {
                "exit_code": 0,
                "added": 12,
                "removed": 3,
                "diff": "\n".join(f"+added line {k}" for k in range(200)),
            },
        },
    }


class FakeJob:
    def __init__(self, job_id: str, rows: list[dict[str, Any]]) -> None:
        self.id = job_id
        self.type = "subagent"
        self.status = "running"
        self.label = f"child {job_id}"
        self.agent = "coder"
        self.intent = "pin the event-loop hotspot"
        self.trajectory = rows
        self.latest_details = {"progress": "thinking"}
        self.prompt = "the child's launch prompt"
        self.usage = None
        self.descendant_usage = []
        self.model_label = "deepseek/deepseek-flash"
        self.start_time = 1_700_000_000.0


class FakeJobsManager:
    def __init__(self, jobs: list[FakeJob]) -> None:
        self._jobs = jobs

    def list(self) -> list[FakeJob]:
        return list(self._jobs)

    def accounting_components(self) -> list[Any]:
        return []


class FakeSession:
    def __init__(self, jobs: list[FakeJob]) -> None:
        self.jobs = FakeJobsManager(jobs)
        self.model = None


def build() -> tuple[Any, FakeSession]:
    jobs = [FakeJob(f"job-{k}", [make_row(i) for i in range(ROWS)]) for k in range(JOBS)]
    session = FakeSession(jobs)
    state = fs.FrontendSessionState.model_validate(
        {
            "session_id": SESSION_ID,
            "epoch": "e1",
            "sequence": 0,
            "jobs": [fs.JobState.from_job(j) for j in jobs],
        }
    )
    return fs.FrontendStateStore(state), session


class TickMeter:
    def __init__(self) -> None:
        self.times: list[float] = []

    def add(self, dt: float) -> None:
        self.times.append(dt)
        if len(self.times) > 400:
            del self.times[:200]

    def summary(self) -> str:
        if not self.times:
            return "ticks=0"
        ts = sorted(self.times)
        n = len(ts)
        return (
            f"ticks={n} mean={sum(ts) / n * 1000:.1f} ms "
            f"p50={ts[n // 2] * 1000:.1f} ms p95={ts[int(n * 0.95)] * 1000:.1f} ms "
            f"max={ts[-1] * 1000:.1f} ms"
        )


def watcher(record_path: Path, stop: threading.Event, ticks: TickMeter, out: list[str]) -> None:
    while not stop.is_set():
        try:
            raw = json.loads(record_path.read_text())
            age = time.time() - raw["heartbeat_at"]
        except Exception as exc:  # noqa: BLE001
            age, raw = float("nan"), {"error": repr(exc)}
        scanned = reg.scan(root=CFG)
        verdict = scanned[0][1] if scanned else "no-record"
        line = (
            f"t={time.time() - T0:5.1f}s  heartbeat_age={age:5.1f}s  "
            f"scan()={verdict:<7} subagents_running={raw.get('subagents_running')}  "
            f"{ticks.summary()}"
        )
        out.append(line)
        print(line, flush=True)
        time.sleep(5.0)


def dialer(port: int, key: str, stop: threading.Event, out: list[str]) -> None:
    n = 0
    while not stop.is_set():
        time.sleep(8.0 if not n else 20.0)
        n += 1
        t0 = time.monotonic()
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        except OSError as exc:
            line = f"   dial: CONNECT FAILED after {time.monotonic() - t0:.2f}s: {exc!r}"
            print(line, flush=True)
            out.append(line)
            continue
        connect_s = time.monotonic() - t0
        sock.settimeout(DIAL_WELCOME_BUDGET_S)
        try:
            sock.sendall(
                json.dumps({"key": key, "client": "attach", "locality": "local"}).encode() + b"\n"
            )
            t1 = time.monotonic()
            data = sock.recv(4096)
            welcome_s = time.monotonic() - t1
            line = (
                f"   dial: connect {connect_s * 1000:.0f} ms, "
                f"welcome {welcome_s * 1000:.0f} ms after auth "
                f"({len(data)} B: {data[:32]!r})"
            )
        except (TimeoutError, socket.timeout):
            line = (
                f"   dial: connect {connect_s * 1000:.0f} ms, NO WELCOME within "
                f"{DIAL_WELCOME_BUDGET_S:.0f}s (the runtime never got back to the socket)"
            )
        except OSError as exc:
            line = f"   dial: connect {connect_s * 1000:.0f} ms, then {exc!r}"
        finally:
            sock.close()
        print(line, flush=True)
        out.append(line)


async def main() -> None:
    CFG.mkdir(parents=True, exist_ok=True)
    for stale in (CFG / "run" / "mobile").glob("*.json"):
        stale.unlink()

    store, session = build()
    record = SessionRecord(
        pid=os.getpid(),
        kind="daemon",
        session_id=SESSION_ID,
        conversation_name="frame cost demo",
        cwd=str(CFG),
        model_label="deepseek/deepseek-flash",
        control_port=0,
        control_key="demo-key",
        busy=True,
        started=True,
        subagents_running=JOBS,
        subagents_queued=0,
    )

    clients: list[asyncio.StreamWriter] = []

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # The runtime's accept path: auth frame in, welcome out. Both halves are
        # loop callbacks, so a saturated loop delays the WELCOME even though the
        # kernel already completed the handshake.
        try:
            await reader.readline()
            writer.write(b'{"op": "welcome", "protocol": 5}\n')
            await writer.drain()
            clients.append(writer)
        except Exception:  # noqa: BLE001
            pass

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    record.control_port = port
    reg.publish(record, root=CFG)
    record_path = reg.record_path(os.getpid(), root=CFG)

    ticks = TickMeter()
    stop = threading.Event()
    out: list[str] = []
    threading.Thread(target=watcher, args=(record_path, stop, ticks, out), daemon=True).start()
    threading.Thread(target=dialer, args=(port, record.control_key, stop, out), daemon=True).start()

    loop = asyncio.get_running_loop()
    counters = {job.id: ROWS for job in session.jobs.list()}

    def tick() -> None:
        """``Session._schedule_frontend_jobs``'s coalesced tick, faithfully.

        Streaming by default: the tick that exists is the one a roster change
        scheduled, so one new row per child (and the front trim it causes at the
        cap) is the shape under load. ``DEMO_STREAM=0`` runs the idle shape.
        """
        nonlocal tick_handle
        if STREAM:
            for job in session.jobs.list():
                job.trajectory.append(make_row(counters[job.id]))
                counters[job.id] += 1
                if len(job.trajectory) > ROWS:
                    del job.trajectory[: len(job.trajectory) - ROWS]
        t0 = time.perf_counter()
        store.refresh_jobs(session)
        ticks.add(time.perf_counter() - t0)
        tick_handle = loop.call_later(TICK_S, tick)

    tick_handle: asyncio.TimerHandle | None = loop.call_later(TICK_S, tick)

    async def heartbeat_loop() -> None:
        """Shape copied from ``RuntimeServer._heartbeat_loop``: an asyncio task."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            reg.publish(record, root=CFG)

    print(
        f"pid={os.getpid()} port={port} cfg={CFG}\n"
        f"tree: {fs.__file__}\n"
        f"pump: {JOBS} jobs x {ROWS} rows x {ROW_TEXT_BYTES} B, tick every "
        f"{TICK_S * 1000:.0f} ms, stream={'on' if STREAM else 'off'}\n"
        f"HEARTBEAT_INTERVAL_S={HEARTBEAT_INTERVAL_S} "
        f"HEARTBEAT_TIMEOUT_S={HEARTBEAT_TIMEOUT_S}  run={RUN_S:.0f}s\n",
        flush=True,
    )
    beat = asyncio.create_task(heartbeat_loop())
    await asyncio.sleep(RUN_S)
    stop.set()
    beat.cancel()
    tick_handle.cancel()
    server.close()
    for w in clients:
        w.close()
    reg.unpublish(os.getpid(), root=CFG)

    ages = [
        float(line.split("heartbeat_age=")[1].split("s")[0])
        for line in out
        if "heartbeat_age=" in line
    ]
    line = [entry for entry in out if entry.startswith("   dial")]
    print("\n--- verdict ---")
    if ages:
        print(
            f"heartbeat age: max {max(ages):.1f}s over {len(ages)} samples "
            f"(timeout {HEARTBEAT_TIMEOUT_S:.0f}s); {ticks.summary()}"
        )
    for entry in line:
        print(entry)
    scans = {entry.split("scan()=")[1].split()[0] for entry in out if "scan()=" in entry}
    print(f"registry.scan verdicts seen: {sorted(scans)}")


if __name__ == "__main__":
    asyncio.run(main())
