#!/usr/bin/env python3
"""Reproduce "a read receipt is refused with store-busy under fleet contention".

WHAT THIS REPRODUCES, AND WHY IT NEEDS REAL CONCURRENCY.
On 2026-09-29 the desktop app answered a session-open with the toast

    "Read state is busy right now, so nothing was written. Try again in a
     moment. The unread marks were not cleared."

and the store's own log on this machine carries the underlying verdict many
times over ("attention: ... stayed busy through 2 attempts: database is
locked", reads and publishes, 2026-09-24..28). The read state is one SQLite
database per config root (``AttentionStore``), reached by ~25 concurrent
sessions, the mobile daemon, the tunnel connector and the browser bridge, and
the acknowledgement that clears an unread mark was the one write path with no
retry at all: a single contended acquisition refused it, and the refusal
reached the operator as the toast above.

WHAT THE SCRIPT DOES. It builds the same shape the machine lives in:

* boots the REAL FastAPI app over uvicorn (the desktop control plane, not a
  test host), on an isolated ``HOME``/config root, and creates one session
  through ``POST /v1/desktop/sessions``;
* publishes one completion for that session (the turn outcome the ack is for)
  and then attempts the REAL user gesture -- ``POST
  /v1/desktop/sessions/{id}/seen`` -- in a loop;
* while that loop runs, N WRITER subprocesses call ``AttentionStore.publish``
  in tight loops and M READER subprocesses call ``state_many`` / ``revision``
  / ``published_since``, exactly the operations the fleet's observers run;
* records every ack outcome and latency, and every worker's ok/deferred
  counts, as a single ``DIGEST`` JSON line;
* runs a second, sharper scenario: an EXTERNAL process holds SQLite's write
  lock on the store for ``--hold-ms`` (proven to be held by a canary attempt
  that must be refused), then ONE ack is fired. The ack must ride the hold out.

WHAT EACH MODE PROVES, and why the script cannot pass vacuously:

* ``--mode refused`` (the pre-fix tree): the run is VALID only when at least
  one ack came back 503 ``store_busy``, and the lone-ack scenario is refused
  too. Exit non-zero otherwise, saying which side it landed on.
* ``--mode clean`` (the fixed tree): the run is VALID only when ZERO acks
  were refused AND the lone ack rode the external hold out (status 200 after
  a wait of at least half the hold). Exit non-zero otherwise.

Both captures use the same load parameters, so the two digests compare.

Run it under a worktree venv, from the repository root::

    .venv/bin/python scripts/repro-attention-contention.py --mode refused
    .venv/bin/python scripts/repro-attention-contention.py --mode clean

Everything it writes lives under ``--root`` (default: a fresh directory under
``$LOCAL_OPERATOR_SCRATCHPAD``); it never touches the operator's own config
root, and it strips ``CMUX_*``/``LOP_*`` from the child environment for the
reason ``scripts/repro-enospc-send.py`` spells out (an inherited
``CMUX_WORKSPACE_ID`` has renamed the operator's real cmux workspaces). The
worker subprocesses are re-execs of this same script (``--worker``), so the
tree under test is the tree the script lives in.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

#: The repository this script lives in, put on the path FIRST: a script's own
#: directory is what Python puts on ``sys.path[0]``, and the run must exercise
#: the TREE THE SCRIPT LIVES IN rather than whichever tree the venv's editable
#: install happens to resolve. Same convention as ``scripts/repro-enospc-send.py``.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Worker modes (subprocesses of this same script)
# ---------------------------------------------------------------------------


def _worker_writer(db: str, prefix: str, seconds: float, stats: str, stop: str) -> int:
    """A fleet session publishing completions, as tightly as the store allows.

    Counts are FLUSHED EVERY ITERATION rather than at the end: the parent may
    end the run while this loop is inside a store call, and a repro's evidence
    must survive its own teardown. The stop-file is the cooperative half of the
    same rule -- the parent writes it when its window closes, so the loop ends
    between operations instead of being SIGTERMed inside one.
    """
    from local_operator.session.attention import AttentionStore, AttentionWriteDeferred

    store = AttentionStore(Path(db))
    ok = deferred = other = 0
    first_error = ""
    deadline = time.monotonic() + seconds
    i = 0
    while time.monotonic() < deadline and not Path(stop).exists():
        i += 1
        try:
            store.publish(f"session/{prefix}-{i}", str(uuid.uuid4()), f"anchor-{i}", "complete")
            ok += 1
        except AttentionWriteDeferred as busy:
            deferred += 1
            first_error = first_error or str(busy)
        except Exception as error:  # noqa: BLE001 - a repro counts, it does not hide
            other += 1
            first_error = first_error or f"{type(error).__name__}: {error}"
        Path(stats).write_text(
            json.dumps(
                {"ok": ok, "deferred": deferred, "other": other, "first_error": first_error}
            ),
            encoding="utf-8",
        )
    return 0


def _worker_reader(db: str, seconds: float, stats: str, scan: int, stop: str) -> int:
    """A fleet observer: the list, revision and delta reads on their own loops."""
    from local_operator.session.attention import AttentionReadDeferred, AttentionStore

    store = AttentionStore(Path(db))
    # Readers scan conversations that EXIST (the seeded population), because a
    # scan over empty names costs an index miss each and the daemon's own scan
    # reads real rows.
    names = [f"session/seed-{i}" for i in range(scan)]
    ok = deferred = other = 0
    first_error = ""
    deadline = time.monotonic() + seconds
    i = 0
    while time.monotonic() < deadline and not Path(stop).exists():
        i += 1
        try:
            op = i % 4
            if op == 0:
                store.state_many(names)
            elif op == 1:
                store.revision()
            elif op == 2:
                store.published_since(0)
            else:
                store.acknowledgement_map()
            ok += 1
        except AttentionReadDeferred as busy:
            deferred += 1
            first_error = first_error or str(busy)
        except Exception as error:  # noqa: BLE001
            other += 1
            first_error = first_error or f"{type(error).__name__}: {error}"
        Path(stats).write_text(
            json.dumps(
                {"ok": ok, "deferred": deferred, "other": other, "first_error": first_error}
            ),
            encoding="utf-8",
        )
    return 0


def _worker_hold(db: str, hold_ms: int, ready: str, stats: str) -> int:
    """An external writer: holds SQLite's write lock for ``hold_ms``, then releases.

    A raw ``BEGIN IMMEDIATE`` on the same database is the exact shape a slow
    sibling writer presents: it is what ``tests/unit/session/
    test_attention_lock_contention.py::_HeldWriteLock`` models, and it blocks
    both a competing writer and, in the store's historical rollback journal,
    the schema transaction's commit.

    ACQUISITION IS RETRIED, because a connection left behind by a killed
    worker can hold the lock for a beat; the ready-file is written only once
    the lock is genuinely held, which is what the parent's canary then proves.
    """
    deadline = time.monotonic() + 8.0
    conn: sqlite3.Connection | None = None
    while time.monotonic() < deadline:
        attempt = sqlite3.connect(db, timeout=0.5)
        try:
            attempt.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            attempt.close()
            time.sleep(0.05)
            continue
        conn = attempt
        break
    if conn is None:
        Path(stats).write_text(
            json.dumps({"held_ms": 0, "error": "never acquired the write lock"}),
            encoding="utf-8",
        )
        return 1
    Path(ready).write_text("HELD", encoding="utf-8")
    time.sleep(hold_ms / 1000.0)
    conn.rollback()
    conn.close()
    Path(stats).write_text(json.dumps({"held_ms": hold_ms}), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def prepare_env(root: Path, token: str) -> None:
    """Isolate the run COMPLETELY, then import the app.

    ``HOME`` as well as ``LOCAL_OPERATOR_CONFIG_DIR`` (the cache root derives
    from the home directory independently), and both ``CMUX_*`` and ``LOP_*``
    stripped -- the child product itself reads both prefixes, which is how a
    headless run renames the operator's cmux workspaces or inherits another
    session's provider. The full account is in ``scripts/repro-enospc-send.py``
    and AGENTS.md "Isolating a run".
    """
    os.environ["HOME"] = str(root / "home")
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(root / "cfg")
    os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"] = token
    os.environ.pop("LOCAL_OPERATOR_DESKTOP_ORIGINS", None)
    os.environ["LOCAL_OPERATOR_NO_SHIMMER"] = "1"
    os.environ["LOCAL_OPERATOR_NO_NOTIFICATIONS"] = "1"
    os.environ["LOCAL_OPERATOR_NO_TERMINAL_TITLE"] = "1"
    for name in [key for key in os.environ if key.startswith(("CMUX_", "LOP_"))]:
        del os.environ[name]


# ---------------------------------------------------------------------------
# Parent orchestration
# ---------------------------------------------------------------------------


def _spawn(worker_args: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker", *worker_args],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _stop(processes: list[subprocess.Popen[str]]) -> list[str]:
    """End every worker and read back anything they printed.

    Workers end cooperatively on the parent's stop-file; this is the sweep for
    ones wedged inside a store call, and the place their stdout/stderr are
    collected -- a worker that dies on import would otherwise just stop
    contributing load, and an empty fleet would look like a quiet store.
    """
    left: list[str] = []
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            out, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            out, _ = process.communicate()
        text = (out or "").strip()
        if text:
            left.append(f"pid {process.pid}: {text[-300:]}")
    return left


def _percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    ordered = sorted(samples)

    def at(frac: float) -> float:
        index = min(len(ordered) - 1, int(frac * len(ordered)))
        return round(ordered[index], 1)

    return {"p50": at(0.50), "p95": at(0.95), "p99": at(0.99), "max": round(ordered[-1], 1)}


async def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    if root.exists() and any(root.iterdir()) and not args.reuse:
        print(f"refusing: {root} is not empty (pass --reuse to share it)")
        return 2
    root.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    prepare_env(root, token)
    config = root / "cfg"
    config.mkdir(parents=True, exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )
    run_dir = root / "run"
    run_dir.mkdir(exist_ok=True)
    # The create route refuses a cwd that does not exist ("Choose an existing
    # working directory"), so the repro's workspace is made up front -- the
    # same step ``scripts/repro-enospc-send.py`` takes for the same reason.
    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    db_path = config / "attention.db"

    import httpx
    import uvicorn

    from local_operator.server.app import app

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    workers: list[subprocess.Popen[str]] = []
    ack_statuses: list[int] = []
    ack_latencies: list[float] = []
    sample_refusal = ""
    sample_ok = ""
    hold: dict[str, Any] = {"status": None, "ms": None, "canary_busy": None, "released": None}
    started = time.monotonic()
    try:
        for _ in range(20_000):
            if server.started:
                break
            if serving.done():
                await serving
            await asyncio.sleep(0)
        assert server.started

        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            listing = "/v1/desktop/sessions"
            created = await client.post(
                listing,
                json={
                    "request_id": "11111111-1111-4111-8111-111111111111",
                    "cwd": str(workspace),
                },
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]
            print(f"session created: {sid}")

            # The completion the ack is for. This is what a finished turn
            # publishes; the ack below is the user gesture that clears it.
            from local_operator.session.attention import AttentionStore

            completion = str(uuid.uuid4())
            AttentionStore(db_path).publish(
                f"session/{sid}", completion, "repro-anchor", "complete"
            )
            # THE LIVE STORE IS NOT EMPTY, and that is load-bearing for the load
            # shape: the operator's own attention.db holds ~25k completions, so
            # the daemon's scans read a grown table rather than a fresh one. A
            # raw multi-row insert in ONE transaction seeds the same scale --
            # publishing 25k rows through the store would take minutes.
            if args.seed:
                seed_conn = sqlite3.connect(db_path)
                seed_conn.executemany(
                    "INSERT INTO completions(conversation,token,anchor,kind) VALUES(?,?,?,?)",
                    (
                        (f"session/seed-{index}", str(uuid.uuid4()), f"seed-{index}", "complete")
                        for index in range(args.seed)
                    ),
                )
                seed_conn.commit()
                seed_conn.close()
                print(f"seeded {args.seed} completions")

            # -- fleet contention: writers and readers churn the store
            for index in range(args.writers):
                workers.append(
                    _spawn(
                        [
                            "writer",
                            "--worker-db",
                            str(db_path),
                            "--prefix",
                            f"w{index}{secrets.token_hex(2)}",
                            "--worker-seconds",
                            str(args.seconds + 4),
                            "--worker-stop",
                            str(run_dir / "stop"),
                            "--worker-stats",
                            str(run_dir / f"writer-{index}.json"),
                        ],
                    )
                )
            for index in range(args.readers):
                workers.append(
                    _spawn(
                        [
                            "reader",
                            "--worker-db",
                            str(db_path),
                            "--worker-seconds",
                            str(args.seconds + 4),
                            "--worker-stop",
                            str(run_dir / "stop"),
                            "--worker-stats",
                            str(run_dir / f"reader-{index}.json"),
                            "--scan",
                            str(args.scan),
                        ],
                    )
                )
            await asyncio.sleep(2.0)  # let the fleet saturate before the gesture

            route = f"/v1/desktop/sessions/{sid}/seen"
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                began = time.monotonic()
                response = await client.post(route, json={"completion_token": completion})
                ack_latencies.append((time.monotonic() - began) * 1000.0)
                ack_statuses.append(response.status_code)
                if response.status_code == 200 and not sample_ok:
                    sample_ok = response.text[:200]
                if response.status_code != 200 and not sample_refusal:
                    sample_refusal = response.text[:300]
                await asyncio.sleep(args.ack_interval)

            # End the fleet cooperatively, then sweep: the stop-file lets each
            # loop finish its current store call and flush its counts, which is
            # what makes the worker numbers below the run's and not a sample.
            (run_dir / "stop").write_text("stop", encoding="utf-8")
            natural_deadline = time.monotonic() + 4.0
            while time.monotonic() < natural_deadline and any(
                process.poll() is None for process in workers
            ):
                await asyncio.sleep(0.05)
            stray = _stop(workers)
            workers = []
            if stray:
                print(f"worker output after stop: {stray}")

            # -- the lone-ack scenario: an EXTERNAL write lock, then one gesture
            ready = run_dir / "hold.ready"
            hold_stats = run_dir / "hold.worker.json"
            holder = _spawn(
                [
                    "hold",
                    "--worker-db",
                    str(db_path),
                    "--hold-ms",
                    str(args.hold_ms),
                    "--ready",
                    str(ready),
                    "--worker-stats",
                    str(hold_stats),
                ],
            )
            workers.append(holder)
            for _ in range(4_000):
                if ready.exists():
                    break
                await asyncio.sleep(0.005)
            assert ready.exists(), "the holder never took the lock"
            # The canary: the lock must actually BLOCK a competing writer,
            # "proven before it is trusted" (the same rule the store's own
            # budget comment states). A lock nobody contends is not evidence.
            probe = sqlite3.connect(db_path, timeout=0.2)
            try:
                probe.execute("BEGIN IMMEDIATE")
                hold["canary_busy"] = False
                probe.rollback()
            except sqlite3.OperationalError as error:
                hold["canary_busy"] = getattr(error, "sqlite_errorname", "SQLITE_BUSY")
            finally:
                probe.close()
            began = time.monotonic()
            response = await client.post(route, json={"completion_token": completion})
            hold["status"] = response.status_code
            hold["ms"] = round((time.monotonic() - began) * 1000.0, 1)
            if response.status_code != 200 and not sample_refusal:
                sample_refusal = response.text[:300]
            await asyncio.sleep(0.2)
            hold["released"] = holder.poll() is not None

        journal = sqlite3.connect(db_path).execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        _stop(workers)
        server.should_exit = True
        await serving

    def _totals(pattern: str) -> dict[str, int]:
        totals = {"ok": 0, "deferred": 0, "other": 0}
        for path in sorted(run_dir.glob(pattern)):
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for key in totals:
                totals[key] += int(data.get(key, 0))
        return totals

    refused = sum(1 for status in ack_statuses if status == 503)
    other = sum(1 for status in ack_statuses if status not in (200, 503))
    digest = {
        "mode": args.mode,
        "seconds": args.seconds,
        "seed": args.seed,
        "acks": len(ack_statuses),
        "ack_ok": sum(1 for status in ack_statuses if status == 200),
        "ack_refused_503": refused,
        "ack_other": other,
        "ack_ms": _percentiles(ack_latencies),
        "writers": _totals("writer-*.json"),
        "readers": _totals("reader-*.json"),
        "hold": hold,
        "journal_mode": journal,
        "load1": round(os.getloadavg()[0], 1),
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "wall_s": round(time.monotonic() - started, 1),
    }
    print("DIGEST " + json.dumps(digest, sort_keys=True))
    if sample_refusal:
        print(f"sample refusal: {sample_refusal}")
    if sample_ok:
        print(f"sample ok: {sample_ok}")

    hold_was_real = hold["canary_busy"] not in (False, None)
    if args.mode == "refused":
        if refused >= 1 and hold["status"] != 200 and hold_was_real:
            print(
                "REPRODUCED: the acknowledgement was refused store_busy under fleet contention "
                f"({refused} of {len(ack_statuses)} acks), and a lone ack could not ride out an "
                f"external write lock (HTTP {hold['status']})."
            )
            return 0
        print(
            "NOT REPRODUCED: no refusals observed"
            f" (acks={len(ack_statuses)}, refused={refused}); lone ack HTTP {hold['status']}; "
            f"canary={hold['canary_busy']!r}. Raise --writers/--readers/--seconds and re-run."
        )
        return 1
    if args.mode == "clean":
        rode_out = (
            hold["status"] == 200
            and hold["ms"] is not None
            and hold["ms"] >= args.hold_ms / 2
            and hold_was_real
        )
        if refused == 0 and other == 0 and rode_out:
            print(
                f"CLEAN: zero refused acks across {len(ack_statuses)} attempts, and the lone ack "
                f"rode out the {args.hold_ms} ms external write lock (HTTP 200 in {hold['ms']} ms)."
            )
            return 0
        print(
            f"NOT CLEAN: refused={refused} other={other} of {len(ack_statuses)}; "
            f"lone ack HTTP {hold['status']} in {hold['ms']} ms (rode_out={rode_out})."
        )
        return 1
    print(f"unknown mode {args.mode!r}")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["refused", "clean"], required=False)
    parser.add_argument(
        "--root",
        default=os.path.join(
            os.environ.get("LOCAL_OPERATOR_SCRATCHPAD", "/tmp"), f"attention-repro-{os.getpid()}"
        ),
        help="isolated run root (default: a fresh dir under $LOCAL_OPERATOR_SCRATCHPAD)",
    )
    parser.add_argument("--reuse", action="store_true", help="allow a non-empty --root")
    parser.add_argument("--writers", type=int, default=6, help="publishing subprocesses")
    parser.add_argument("--readers", type=int, default=4, help="observing subprocesses")
    parser.add_argument("--seconds", type=float, default=40.0, help="contention window")
    parser.add_argument("--ack-interval", type=float, default=0.25, help="pause between acks")
    parser.add_argument("--scan", type=int, default=400, help="conversations per state_many scan")
    parser.add_argument(
        "--seed",
        type=int,
        default=25000,
        help="completions pre-seeded so scans read a grown table (the live store's scale)",
    )
    parser.add_argument("--hold-ms", type=int, default=7000, help="external write-lock hold")
    # worker-mode arguments (this same script re-exec'd)
    parser.add_argument("--worker", choices=["writer", "reader", "hold"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-db", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seconds", type=float, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-stats", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-stop", default="", help=argparse.SUPPRESS)
    parser.add_argument("--prefix", default="", help=argparse.SUPPRESS)
    parser.add_argument("--ready", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker == "writer":
        return _worker_writer(
            args.worker_db, args.prefix, args.worker_seconds, args.worker_stats, args.worker_stop
        )
    if args.worker == "reader":
        return _worker_reader(
            args.worker_db, args.worker_seconds, args.worker_stats, args.scan, args.worker_stop
        )
    if args.worker == "hold":
        return _worker_hold(args.worker_db, args.hold_ms, args.ready, args.worker_stats)
    if args.mode is None:
        parser.error("--mode is required for a parent run")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
