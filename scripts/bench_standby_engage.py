"""New session -> runtime bound and ready for a message, with and without the standby.

WHAT IT MEASURES
================
The span the TUI's ``/new`` and the desktop's first send sit on "starting…":
``AttachedSession.cold()`` -> ``bind_runtime()`` (spawn or adopt, construct,
publish, dial, frontend sync, history) — the same driver
``scripts/bench_cold_engage.py`` uses, minus its per-phase child instrument.
Each arm is one of:

  cold     ``LOP_RUNTIME_STANDBY_DISABLED=1``: the fork+import path exactly as
           it was before ``session/runtime/standby.py``.
  standby  a standby is warmed for the root FIRST, outside the timed span, and
           the timed span starts only once it is listening — the steady state
           of a host that has engaged before (every engage re-warms behind
           itself). If the standby refuses or is absent the engage falls back
           to cold, and the row records which happened (``adopted``).

The arms are INTERLEAVED per pass (AGENTS.md "Timing, flakes": wall time under
fleet load is weather), each pass in its own fresh HOME + config root, and every
row carries the 1-minute load average and the parent's CPU so a reader can see
the weather beside the number. Reported: p50/p95/max per arm.

ISOLATION
=========
Fresh ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR`` per pass (the model catalogue
cache follows HOME, AGENTS.md "Isolating a run"); every ``LOP_*``/``CMUX_*``
stripped; the ``test`` provider; every pid this script starts — the runtime, the
standby, the replacement standby — is SIGTERMed by exact pid at the end of its
pass, identified through the root's own lock file and record, never by name.

USAGE
=====
    env -u XPC_FLAGS .venv/bin/python scripts/bench_standby_engage.py --pairs 8 \\
        --json out.json

``--measured-tree`` works as in the sibling scripts (``bench_tree``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import bench_tree  # noqa: E402

_STRIPPED_PREFIXES = ("LOP_", "CMUX_")
_ARMS = ("cold", "standby")


def _strip_inherited() -> None:
    for key in list(os.environ):
        if key.startswith(_STRIPPED_PREFIXES):
            del os.environ[key]


def _lock_holders(root: Path) -> set[int]:
    """Pids holding THIS root's standby lock: the standby, and nothing else."""
    lock = root / "run" / "standby" / "lock"
    if not lock.exists():
        return set()
    out = subprocess.run(["lsof", "-t", str(lock)], capture_output=True, text=True).stdout
    return {int(x) for x in out.split() if x.strip().isdigit()} - {os.getpid()}


def _warm_standby(root: Path, timeout_s: float) -> tuple[bool, float, int | None, float | None]:
    """Warm one standby for ``root`` and wait until it listens. (ok, seconds, pid, rss_mb)."""
    from local_operator.session.runtime import standby
    from local_operator.session.runtime.launch import _spawn_interpreter

    started = time.perf_counter()
    standby._WARMING[0] = True
    standby.ensure_warm(root, _spawn_interpreter())
    sock = standby.socket_path(root, create=False)
    while time.perf_counter() - started < timeout_s:
        if sock.exists():
            holders = _lock_holders(root)
            pid = min(holders) if holders else None
            rss = None
            if pid:
                raw = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
                ).stdout.strip()
                rss = round(int(raw) / 1024, 1) if raw.isdigit() else None
            return True, time.perf_counter() - started, pid, rss
        time.sleep(0.05)
    return False, time.perf_counter() - started, None, None


def _cpu_ms() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return (usage.ru_utime + usage.ru_stime) * 1000.0


async def _engage(root: Path, session_id: str) -> dict[str, Any]:
    from local_operator.mobile.attach_client import find_runtime_record
    from local_operator.session.attached import AttachedSession

    async def _no_takeover() -> None:
        raise RuntimeError("benchmark has no takeover")

    t0 = time.perf_counter()
    c0 = _cpu_ms()
    remote = await AttachedSession.cold(
        session_id,
        config_dir=root,
        cwd=str(root.parent),
        takeover_factory=_no_takeover,
        surface="desktop",
    )
    t1 = time.perf_counter()
    await remote.bind_runtime()
    bound_ms = (time.perf_counter() - t0) * 1000.0
    bind_ms = (time.perf_counter() - t1) * 1000.0
    parent_cpu = _cpu_ms() - c0
    record, _ = await asyncio.to_thread(find_runtime_record, root, session_id)
    try:
        await remote.dispose()
    except Exception:  # noqa: BLE001 — teardown noise is not a measurement
        pass
    return {
        "bound_ms": round(bound_ms, 1),
        "cold_facade_ms": round(bound_ms - bind_ms, 1),
        "bind_ms": round(bind_ms, 1),
        "parent_cpu_ms": round(parent_cpu, 1),
        "runtime_pid": getattr(record, "pid", None),
    }


def _warm_parent() -> None:
    import local_operator.mobile.attach_client  # noqa: F401
    import local_operator.session.attached  # noqa: F401
    import local_operator.session.runtime.launch  # noqa: F401
    from local_operator.session_factory import warm_session_imports

    warm_session_imports()


def _one(arm: str, index: int, warm_timeout_s: float) -> dict[str, Any]:
    base = Path(tempfile.mkdtemp(prefix=f"lopsa-bench-{arm}-"))
    root = base / ".local-operator"
    root.mkdir(parents=True)
    saved = {
        k: os.environ.get(k)
        for k in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR", "LOP_RUNTIME_STANDBY_DISABLED")
    }
    os.environ["HOME"] = str(base)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)
    from local_operator.config import ConfigManager
    from local_operator.session.runtime import standby

    ConfigManager(config_dir=root).update_config({"hosting": "test", "model_name": "test-model"})
    row: dict[str, Any] = {"arm": arm, "pass": index}
    spare_pid = None
    try:
        if arm == "cold":
            os.environ[standby.DISABLE_ENV] = "1"
            standby._WARMING[0] = False
        else:
            os.environ.pop(standby.DISABLE_ENV, None)
            ok, secs, spare_pid, rss = _warm_standby(root, warm_timeout_s)
            row.update(standby_ready=ok, standby_warm_s=round(secs, 1), standby_rss_mb=rss)
            # Nothing re-warms INSIDE the timed span; the replacement is its
            # own measurement (standby_warm_s), not weather on this one.
            standby._WARMING[0] = False
        row["load1"] = round(os.getloadavg()[0], 1)
        row.update(asyncio.run(_engage(root, uuid.uuid4().hex[:12])))
        row["adopted"] = bool(spare_pid and row.get("runtime_pid") == spare_pid)
    finally:
        pids = {p for p in (row.get("runtime_pid"), spare_pid) if p} | _lock_holders(root)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        time.sleep(0.5)
        shutil.rmtree(base, ignore_errors=True)
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in _ARMS:
        values = sorted(r["bound_ms"] for r in rows if r["arm"] == arm and "bound_ms" in r)
        if not values:
            continue

        def pct(q: float) -> float:
            return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]

        binds = sorted(r["bind_ms"] for r in rows if r["arm"] == arm and "bind_ms" in r)
        out[arm] = {
            "bind_p50": round(statistics.median(binds), 1) if binds else None,
            "bind_p95": (
                binds[min(len(binds) - 1, int(round(0.95 * (len(binds) - 1))))] if binds else None
            ),
            "n": len(values),
            "p50": round(statistics.median(values), 1),
            "p95": round(pct(0.95), 1),
            "max": values[-1],
            "adopted": sum(1 for r in rows if r["arm"] == arm and r.get("adopted")),
            "load1_median": statistics.median(r["load1"] for r in rows if r["arm"] == arm),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warm-timeout", type=float, default=600.0)
    parser.add_argument("--json", default="")
    parser.add_argument("--measured-tree", default="")
    args = parser.parse_args()
    tree = bench_tree.describe(args.measured_tree)
    print(bench_tree.format_banner(tree), flush=True)
    _strip_inherited()
    from local_operator.tui.notify import suppress_notifications_for_process

    suppress_notifications_for_process("standby engage benchmark")
    # THE PARENT IS WARMED FIRST, because the host this stands in for is: a TUI
    # or the desktop daemon has imported the viewer stack long before its first
    # /new. Without this, pass 0 of whichever arm runs first pays ~1 s of the
    # PARENT's own imports inside ``AttachedSession.cold()`` (measured), which is
    # a benchmark artefact, not a cost either arm has in production.
    _warm_parent()
    rows: list[dict[str, Any]] = []
    for index in range(args.pairs):
        order = _ARMS if index % 2 == 0 else tuple(reversed(_ARMS))
        for arm in order:
            row = _one(arm, index, args.warm_timeout)
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = _summary(rows)
    print(json.dumps({"summary": summary}, indent=1))
    if args.json:
        Path(args.json).write_text(
            json.dumps({"rows": rows, "summary": summary, "tree": tree}, indent=1)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
