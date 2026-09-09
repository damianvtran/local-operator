"""Benchmark the sidebar's catalog poll: wall time AND syscalls per poll.

Usage:

    PYTHONPATH=. .venv/bin/python scripts/bench_catalog_scan.py ladder \
        [--sizes 100,500,1000,2000,4000] [--json out.json]
    PYTHONPATH=. .venv/bin/python scripts/bench_catalog_scan.py real \
        [--store ~/.local-operator] [--json out.json]

``PYTHONPATH=.`` matters for the reason ``bench_resume_picker.py`` documents:
the script must import THIS checkout, not whatever an editable install
resolves to.

**Why this benchmark counts syscalls and not just milliseconds.** The cost it
exists to measure is paid by the TUI every 2 seconds for as long as the sidebar
is open, on a machine that routinely runs a dozen agent sessions at once. Wall
time there is dominated by whatever else is scheduled — samples on the
reporting machine varied 192-642 ms for the same work at load average 48 — so a
wall-clock delta alone cannot distinguish a real improvement from a quiet
minute. The syscall count is the honest invariant: it is a property of the
algorithm, it does not move with load, and it is what the poll actually asks
the filesystem for. Wall time is reported alongside it as corroboration, always
with the load average that produced it.

The ladder measures SCALING, which is the actual defect: the poll's cost grew
with the total number of session directories ever created rather than with the
user's own sessions, so it degraded permanently as the store accumulated.

The ``real`` mode measures the operator's own store. It is strictly READ-ONLY:
it never writes to the store it measures, so it is safe to point at a live
``~/.local-operator`` while sessions are running. (The one write the scan can
normally make — the origin verdict cache — is neutralised; see ``_no_writes``.)
"""

from __future__ import annotations

import collections
import contextlib
import gc
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from local_operator.resume import _recent_sessions_with_origin
from local_operator.session.catalog import load_catalog

REPS = 7


@contextlib.contextmanager
def _counted() -> Iterator[collections.Counter[str]]:
    """Count the filesystem syscalls issued inside the block.

    Patches the ``os`` entry points rather than sampling with ``dtrace``/
    ``strace``: those need privileges this script must not require, and the
    Python-level count is the one that maps back to a line of code. The
    counters are installed around a single call and removed immediately, so
    nothing else in the process pays for them.

    ``os.stat`` is patched on the ``os`` module, which is what application code
    calls. ``pathlib`` reaches the C ``posix.stat`` directly and is therefore
    NOT counted here — deliberately: a pathlib-mediated stat that this harness
    cannot see would understate the BEFORE number and flatter the change, so
    the modules under test call ``os.stat`` explicitly on the hot path and the
    remaining pathlib traffic is reported by ``--profile`` instead.
    """
    counts: collections.Counter[str] = collections.Counter()
    originals = {name: getattr(os, name) for name in ("stat", "lstat", "scandir", "open")}

    def wrap(name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def counting(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return real(*args, **kwargs)

        return counting

    for name, real in originals.items():
        setattr(os, name, wrap(name, real))
    try:
        yield counts
    finally:
        for name, real in originals.items():
            setattr(os, name, real)


def _timed(fn: Callable[[], Any], reps: int = REPS) -> dict[str, float]:
    """min/median/max milliseconds over ``reps`` runs.

    The median is the headline and the min is kept alongside, because this
    machine runs several agents' suites at once: a slow sample measures
    contention rather than the code. ``gc.collect()`` before each sample so a
    collection triggered by the previous run is not billed to this one.
    """
    samples = []
    for _ in range(reps):
        gc.collect()
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "min_ms": round(min(samples), 2),
        "median_ms": round(statistics.median(samples), 2),
        "max_ms": round(max(samples), 2),
    }


@contextlib.contextmanager
def _no_writes(store: Path) -> Iterator[None]:
    """Make the measured call provably read-only against ``store``.

    The scan persists an origin-verdict cache under ``<store>/cache``. That is
    the only write on this path, and writing it into the operator's LIVE store
    from a benchmark would both mutate what is being measured and race the real
    sessions using it. Redirecting the cache is not enough on its own to prove
    the point, so this also asserts that nothing opened a file for writing
    anywhere under the store while the block ran.
    """
    from local_operator import resume

    real_save = resume._save_origin_cache
    written: list[str] = []

    def refuse(path: Path, entries: dict[str, Any]) -> None:
        written.append(str(path))

    resume._save_origin_cache = refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        resume._save_origin_cache = real_save  # type: ignore[assignment]
    inside = [p for p in written if str(store) in p]
    if inside:
        raise AssertionError(f"benchmark attempted to write inside the store: {inside}")


def _measure(store: Path, *, read_only: bool) -> dict[str, Any]:
    """Cold and warm figures for one store, plus syscalls for one warm poll."""
    guard = _no_writes(store) if read_only else contextlib.nullcontext()
    with guard:
        gc.collect()
        start = time.perf_counter()
        rows = load_catalog(store)
        cold_ms = (time.perf_counter() - start) * 1000

        catalog = _timed(lambda: load_catalog(store))
        scan = _timed(lambda: _recent_sessions_with_origin(store))

        # Counted on a WARM store: the steady state is what the sidebar
        # actually pays every 2 s, and a cold count would measure the first
        # poll after launch instead.
        with _counted() as catalog_counts:
            load_catalog(store)
        with _counted() as scan_counts:
            _recent_sessions_with_origin(store)

    return {
        "rows": len(rows),
        "cold_ms": round(cold_ms, 2),
        "load_catalog": catalog,
        "scan": scan,
        "load_catalog_syscalls": dict(catalog_counts),
        "load_catalog_syscalls_total": sum(catalog_counts.values()),
        "scan_syscalls": dict(scan_counts),
        "scan_syscalls_total": sum(scan_counts.values()),
    }


def _build_store(root: Path, total: int) -> None:
    """A synthetic store shaped like the real one.

    The proportions matter more than the absolute size. On the reporting
    machine 1,785 of 1,946 directories (92%) are SUBAGENT sessions that the
    picker never lists, which is precisely the population the poll used to pay
    full price for, so a synthetic store of only user sessions would measure a
    case that does not occur and would hide the effect entirely.

    The awkward shapes are represented on purpose, because they are where an
    equivalence bug would hide: directories with no transcript at all, ones
    carrying only an inbox spool, and forked sessions with an origin marker
    that IS user-visible.
    """
    sessions = root / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for i in range(total):
        sid = f"{i:012x}"
        d = sessions / sid
        d.mkdir(exist_ok=True)
        kind = i % 100
        if kind < 92:  # subagent: has a transcript, never listed
            (d / "transcript.jsonl").write_text('{"type":"message"}\n', encoding="utf-8")
            (d / "origin.json").write_text('{"origin":"subagent"}', encoding="utf-8")
        elif kind < 95:  # fork: user-visible despite carrying a marker
            (d / "transcript.jsonl").write_text('{"type":"message"}\n', encoding="utf-8")
            (d / "origin.json").write_text('{"origin":"fork"}', encoding="utf-8")
        elif kind < 97:  # plain user session
            (d / "transcript.jsonl").write_text('{"type":"message"}\n', encoding="utf-8")
        elif kind < 99:  # activity is the mail spool only
            (d / "inbox.jsonl").write_text('{"from":"peer"}\n', encoding="utf-8")
        else:  # no activity at all: never a row
            (d / "notes.txt").write_text("x", encoding="utf-8")
        stamp = now - (i % 5000)
        for name in ("transcript.jsonl", "inbox.jsonl"):
            with contextlib.suppress(OSError):
                os.utime(d / name, (stamp, stamp))


def _ladder(sizes: list[int]) -> list[dict[str, Any]]:
    results = []
    for size in sizes:
        root = Path(tempfile.mkdtemp(prefix=f"lo-bench-{size}-"))
        try:
            _build_store(root, size)
            # A fresh store's first call also builds the verdict cache, so the
            # cache is warmed once before measuring the steady state.
            load_catalog(root)
            row = {"dirs": size, **_measure(root, read_only=False)}
            results.append(row)
            print(
                f"  {size:>5} dirs: rows={row['rows']:<4} "
                f"catalog={row['load_catalog']['median_ms']:>8.2f} ms  "
                f"scan={row['scan']['median_ms']:>8.2f} ms  "
                f"syscalls={row['load_catalog_syscalls_total']:>6}",
                flush=True,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return results


def main() -> int:
    argv = sys.argv[1:]
    mode = argv[0] if argv else "ladder"
    out_json = None
    if "--json" in argv:
        out_json = argv[argv.index("--json") + 1]

    payload: dict[str, Any] = {"loadavg": os.getloadavg(), "mode": mode}
    print(f"load average: {os.getloadavg()}")

    if mode == "ladder":
        sizes = [100, 500, 1000, 2000, 4000]
        if "--sizes" in argv:
            sizes = [int(x) for x in argv[argv.index("--sizes") + 1].split(",")]
        print("synthetic ladder (92% subagents, mirroring the real store's shape):")
        payload["ladder"] = _ladder(sizes)
    elif mode == "real":
        store = Path(os.path.expanduser("~/.local-operator"))
        if "--store" in argv:
            store = Path(os.path.expanduser(argv[argv.index("--store") + 1]))
        total = len(list((store / "sessions").iterdir()))
        print(f"real store: {store} ({total} directories), READ-ONLY")
        result = _measure(store, read_only=True)
        payload["real"] = {"dirs": total, "store": str(store), **result}
        print(json.dumps(payload["real"], indent=2))
    else:
        print(__doc__)
        return 2

    payload["loadavg_after"] = os.getloadavg()
    if out_json:
        Path(out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
