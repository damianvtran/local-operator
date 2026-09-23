"""Attach latency for one runtime, on production objects, at several owner states.

WHY IT IS IN THE TREE. The numbers this produces are the evidence for the
off-loop attach path (``_ONLOOP_BIND_GRACE_S`` in
``local_operator/session/runtime/server.py``), and a rig that lives only in a
session's scratchpad is unversioned: a future reader can see the table but not
reproduce or refute it. It is the ``scripts/bench_*.py`` convention, and the
design-note evidence is regenerable with the two commands below.

WHAT IT DRIVES. Real ``Session`` over the fixture's transcript,
``ServingSessionHandle``, ``RuntimeServer`` serving on ITS OWN THREAD
(``start()``, the production daemon/exec shape — the serving plane is decoupled
from the session's workload loop), and the production ``AttachedSession``
client. The only double is the provider stream.

Scenarios
  idle        owner idle; measure connect() total + phases
  busy_sync   owner's WORKLOAD loop is blocked by a synchronous step of
              --block-s seconds (a long sync tool / json encode / regex) while
              the viewer dials. Serving plane is free; frontend_sync is not,
              because subscribe_frontend is @_on_session_loop (serving.py:363,
              2413) — so this reproduces "owner-silent".
  busy_read   same, but through the READ envelope the desktop uses
              (attach_existing(budget=READ_ATTACH_BUDGET_S), attached.py:2401).
  busy_cpu    same, but the loop is blocked by a TIGHT PURE-PYTHON loop rather
              than time.sleep. Sleep releases the GIL; a CPU-bound step does not,
              so this is the case that shows the fallback's cost under GIL
              contention (it advances at the 5 ms switch interval).

Phases recorded for ``connect()`` (attached.py:1390):
  dial_ms          _dial -> welcome read  (AttachClient.connect, attach_client.py:1033)
  sync_ms          _await_frontend (runtime's subscribe_frontend on workload loop)
  history_ms       _load_frontend_history (viewer re-reads the durable cut)
  total_ms

The grace and the fallback are separated by ``--grace-ms 0``: with the grace off,
``sync_ms`` IS the off-loop leg (dial + snapshot + the durable read), which is the
headroom against the 300 ms bar the fixed 100 ms grace would otherwise hide.

Usage (from the repo root; build the fixtures once, then measure one row):

  ISO=$(mktemp -d)                      # isolated HOME + config dir
  .venv/bin/python scripts/bench_attach_fixtures.py "$ISO/.local-operator" p50 s5000
  env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" PATH="$PATH" \
    TERM=xterm-256color .venv/bin/python scripts/bench_attach_latency.py \
    --root "$ISO/.local-operator" --runs 5 --scenario busy_sync [--grace-ms 0]

Every run must be isolated (its own HOME/config dir): the rig writes
``.session.pid`` liveness markers into the fixture sessions it is given.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(os.environ.get("PYTHONPATH", ".")).resolve()))


def pct(v, q):
    s = sorted(v)
    return s[max(0, min(len(s) - 1, round(q * (len(s) - 1))))]


async def _wait_record(config_dir: Path, sid: str, timeout: float = 30.0):
    from local_operator.session.runtime import registry

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for record, _state in registry.scan(config_dir):
            if getattr(record, "session_id", "") == sid:
                return record
        await asyncio.sleep(0.02)
    raise RuntimeError("no record")


async def one(root: Path, sid: str, scenario: str, block_s: float) -> dict[str, Any]:
    from local_operator.session.attached import READ_ATTACH_BUDGET_S, AttachedSession
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.e2e.harness import ScriptedStream, build_session, text_turn

    directory = root / "sessions" / sid
    marks: dict[str, float] = {}
    t = time.perf_counter()
    session = build_session(directory, ScriptedStream([text_turn("ok")]))
    marks["runtime_build_session_ms"] = (time.perf_counter() - t) * 1000
    # Resume semantics: the runtime replays the transcript into its context the
    # way session_factory does (Transcript() parse happened inside build_session).
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    t = time.perf_counter()
    server.start()  # own thread, the production serving-plane shape
    record = await _wait_record(root, session.session_id)
    marks["runtime_publish_ms"] = (time.perf_counter() - t) * 1000

    async def never():
        raise RuntimeError("never take over")

    # Phase instrumentation: wrap the facade's private seams for this run only.
    orig_dial = AttachedSession._dial
    orig_await = AttachedSession._await_frontend
    orig_hist = AttachedSession._load_frontend_history

    async def dial(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return await orig_dial(self, *a, **kw)
        finally:
            marks["dial_ms"] = (time.perf_counter() - t0) * 1000

    async def await_frontend(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return await orig_await(self, *a, **kw)
        finally:
            marks["sync_ms"] = (time.perf_counter() - t0) * 1000

    async def hist(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return await orig_hist(self, *a, **kw)
        finally:
            marks["history_ms"] = (time.perf_counter() - t0) * 1000

    AttachedSession._dial = dial
    AttachedSession._await_frontend = await_frontend
    AttachedSession._load_frontend_history = hist
    try:
        # The in-process runtime never takes the transcript lease, so write the
        # liveness marker the discovery path reads (resume.live_runtime_pid,
        # resume.py:1155) — a marker with no claim beside it keeps plain
        # pid-liveness semantics (documented there). Removed in finally.
        (directory / ".session.pid").write_text(str(os.getpid()))

        def run_viewer():
            async def go():
                t0 = time.perf_counter()
                c0 = time.thread_time()
                if scenario == "busy_read":
                    v = await AttachedSession.cold(
                        sid,
                        config_dir=root,
                        cwd=str(directory),
                        takeover_factory=never,
                        surface="desktop",
                    )
                    marks["cold_facade_ms"] = (time.perf_counter() - t0) * 1000
                    t1 = time.perf_counter()
                    ok = await v.attach_existing(budget=READ_ATTACH_BUDGET_S)
                    marks["read_attach_ms"] = (time.perf_counter() - t1) * 1000
                    marks["read_attached"] = float(ok)
                    marks["cold_reason"] = v.cold_reason  # type: ignore[assignment]
                else:
                    v = await AttachedSession.connect(
                        record, session.session_id, config_dir=root, takeover_factory=never
                    )
                marks["total_ms"] = (time.perf_counter() - t0) * 1000
                marks["viewer_cpu_ms"] = (time.thread_time() - c0) * 1000
                marks["history_messages"] = len(getattr(v, "_history", []) or [])
                await v.dispose()

            try:
                asyncio.run(go())
            except BaseException as e:  # noqa: BLE001
                marks["error"] = f"{type(e).__name__}: {e}"[:200]  # type: ignore[assignment]

        th = threading.Thread(target=run_viewer)
        th.start()
        if scenario.startswith("busy"):
            # Block the SESSION's (workload) loop synchronously, the way a long
            # synchronous step inside a turn would, WHILE the viewer (on its own
            # thread/loop) dials. The serving plane has its own thread.
            time.sleep(0.005)
            marks["block_started_s"] = time.perf_counter()
            if scenario == "busy_cpu":
                # A TIGHT PURE-PYTHON BLOCKER, never ``time.sleep``. Sleep RELEASES
                # the GIL, so the serving thread would run at full speed and hide
                # exactly the cost this scenario exists to show: the off-loop
                # fallback serialises on the serving thread while the workload
                # loop holds the GIL in bytecode, so it progresses at the
                # interpreter's switch interval (5 ms) rather than continuously.
                # A real synchronous step inside a turn (a long JSON encode, a
                # regex, a big model_dump) holds the GIL the same way.
                #
                # ``perf_counter`` is a plain C call that does NOT drop the GIL,
                # so the inner range is the only place the interpreter can switch.
                deadline = time.perf_counter() + block_s
                acc = 0
                while time.perf_counter() < deadline:
                    for i in range(50_000):
                        acc += i
                marks["cpu_checksum"] = float(acc)
            else:
                time.sleep(block_s)
        while th.is_alive():
            await asyncio.sleep(0.01)
    finally:
        AttachedSession._dial = orig_dial
        AttachedSession._await_frontend = orig_await
        AttachedSession._load_frontend_history = orig_hist
        server.close()
        await session.dispose()
        (directory / ".session.pid").unlink(missing_ok=True)
    marks.pop("block_started_s", None)
    return marks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--only", nargs="*")
    ap.add_argument(
        "--scenario", default="idle", choices=("idle", "busy_sync", "busy_read", "busy_cpu")
    )
    ap.add_argument("--block-s", type=float, default=20.0)
    ap.add_argument(
        "--grace-ms",
        type=float,
        default=None,
        help=(
            "override the runtime's on-loop bind grace for this run, in ms. 0 "
            "measures the off-loop leg on its own — the cell that separates the "
            "grace constant from the fallback's own cost"
        ),
    )
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if a.grace_ms is not None:
        # Patched as a MODULE attribute rather than threaded through the runtime:
        # the serving path reads it at call time, so this is the constant under
        # test and nothing else. It is how the "--grace-ms 0" row below is taken.
        from local_operator.session.runtime import server as server_module

        server_module._ONLOOP_BIND_GRACE_S = a.grace_ms / 1000.0
    import logging

    logging.disable(logging.WARNING)
    out = {}
    for d in sorted((a.root / "sessions").iterdir()):
        if a.only and d.name not in a.only:
            continue
        rows = []
        for _ in range(a.runs):
            load = os.getloadavg()[0]
            m = asyncio.run(one(a.root, d.name, a.scenario, a.block_s))
            m["load"] = load
            rows.append(m)
        summ = {
            "bytes": (d / "transcript.jsonl").stat().st_size,
            "load_p50": round(statistics.median(r["load"] for r in rows), 1),
        }
        for k in sorted({k for r in rows for k, v in r.items() if isinstance(v, float)} - {"load"}):
            v = [r[k] for r in rows if isinstance(r.get(k), float)]
            summ[k] = {"p50": round(pct(v, 0.5), 1), "p95": round(pct(v, 0.95), 1)}
        for k in ("error", "cold_reason"):
            vals = {r.get(k) for r in rows if r.get(k) is not None}
            if vals:
                summ[k] = sorted(map(str, vals))
        out[d.name] = summ
        print(json.dumps({d.name: summ}), flush=True)
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
