#!/usr/bin/env python3
"""Overhead benchmark for the tokens-per-second analytics change.

WHAT THIS MEASURES. The change adds a decode window (first/last output delta and
a counter) to ``SessionStreamFn._record_stream`` and three aggregate columns
(``decode_us``/``decode_tokens``/``decode_calls``) to the ``calls`` ledger and the
``session_daily`` rollup, then serves a new ``model_rates()`` read. Its claim,
stated so it can be falsified, is:

  1. HOT PATH -- the added cost is at most ONE ``time.monotonic()`` per output
     delta, with no allocation, no lock and no I/O, and the two per-event
     ``getattr(event, "type", "")`` lookups collapse into one.
  2. MIGRATION -- the three ``session_daily`` ALTERs are metadata-only: O(1) in
     row count, so a first open of the real ledger costs what it cost before.
  3. READS -- the four pre-existing reads are UNCHANGED, and ``model_rates()``
     is the only new read cost, paid only by its own endpoint.

Arm 1 is measured two ways and the second is the decisive one:

  * ``hot``   -- wall time per event of a synthetic stream, N = 50/500/5000
    mixed ``text_delta``/``reasoning_delta``/``tool_call_delta``, driven exactly
    as ``tests/unit/analytics/test_stream_recording.py::_drain`` drives it
    (``object.__new__(SessionStreamFn)`` + an async generator, recorder pointed
    at a temp-path store via ``reset_recorder_for_test``). No network.
  * ``clock`` -- the SAME stream with ``local_operator.model.configure.time``
    replaced by a counting proxy, so the number of ``time.monotonic()`` calls is
    COUNTED rather than inferred. This is load-independent: "at most one clock
    read per output delta" is a count, not a duration, and a count does not care
    what else is running on the host. The wall-time arm is the one that a loaded
    machine can move; this arm cannot be moved by load at all.

Arm 2 times ``_connect()`` -- and, isolated from it, the real ``_migrate()`` and
``_migrate_session_daily()`` bodies -- on identical ``cp -c`` clones of a
read-only copy of the operator's real ledger. Every child process prints
``PRAGMA table_info`` / ``page_count`` / row counts before and after, so
"metadata-only" is shown rather than asserted.

Arm 3 times the reads on a migrated clone and records
``store.last_aggregate_source`` / ``last_aggregate_refusal`` beside every
``aggregate()`` sample, so the reader knows which path was measured.

HOW TO REPRODUCE. One root per invocation, so the same script drives both arms
and neither can accidentally import the other's tree. Run each checkout's OWN
venv (provenance is asserted, not assumed -- see ``_claim_root``):

    BASE=/Users/damian/lo-tps-base
    HEAD=/Users/damian/lo-analytics-tps
    ISO=$(mktemp -d)
    run() { env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" \\
              PATH="$PATH" TERM=xterm-256color "$@"; }
    run $BASE/.venv/bin/python bench_tps_overhead.py --root $BASE --save base.json
    run $HEAD/.venv/bin/python bench_tps_overhead.py --root $HEAD --save head.json
    run $HEAD/.venv/bin/python bench_tps_overhead.py --root $HEAD \\
        --compare base.json head.json

The fixture is the prepared read-only ledger copy; point ``--fixture`` at it.
Nothing is written to it: every arm works on ``cp -c`` clones under a temp dir.

A NOTE ON LOAD. This host runs a fleet of concurrent sessions, so a wall time
here is a measurement WITH ITS CONDITIONS, never a guarantee. Every run records
the 1/5/15-minute load average at start and at end, and the report prints both.
For that reason the A/B is meant to be run INTERLEAVED (base, head, base, head,
...) and reduced by median across sessions, and the ``clock`` arm exists to give
a load-independent answer to the part of the claim that is a count.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

#: The tree under test is named by ``--root`` and inserted BEFORE anything else,
#: so ``import local_operator`` resolves to it and nothing else. When this file
#: is run out of a checkout's own ``scripts/`` directory that is also the
#: default.
_DEFAULT_ROOT = Path(__file__).resolve().parent
if not (_DEFAULT_ROOT / "local_operator").is_dir():
    _DEFAULT_ROOT = Path(__file__).resolve().parents[1]

#: Repeats for the bare-clock calibration: enough samples that the floor is the
#: clock call and not the scheduler.
_CLOCK_SAMPLES = 5
_CLOCK_BATCH = 200_000


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def _claim_root(root: Path) -> str:
    """Import ``local_operator`` from ``root`` and refuse to measure otherwise.

    An editable install resolves through a finder holding ONE hard-coded source
    root, so a venv from another worktree imports another tree while every
    observable (version banner, status bar) says otherwise. That failure is
    silent, so this asserts the resolved module path is under ``root`` and dies
    loudly if it is not.
    """
    root = root.resolve()
    sys.path.insert(0, str(root))
    import local_operator  # noqa: PLC0415

    resolved = Path(local_operator.__file__).resolve()
    if root not in resolved.parents:
        raise SystemExit(
            f"provenance failure: local_operator resolved to {resolved}, "
            f"which is not under --root {root}. Wrong venv."
        )
    return str(resolved)


def _git(root: Path, *argv: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *argv],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:  # noqa: BLE001
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _refuse_live_home(allow: bool) -> None:
    """Refuse to run against the operator's real config root.

    AGENTS.md is explicit that a run without a redirected ``HOME`` reads the
    operator's live ``~/.local-operator``, and that *writing* to it has already
    done damage on this machine. This benchmark does not mean to write there --
    every store it opens is an explicit temp path -- but the recorder resolves a
    default store from the config root when handed ``None``, and one future edit
    is all that separates "reads a fixture" from "writes the live ledger". The
    check is therefore a hard stop, with an override for someone who really does
    mean it.
    """
    if allow:
        return
    import pwd  # noqa: PLC0415

    home = os.environ.get("HOME") or ""
    config_dir = os.environ.get("LOCAL_OPERATOR_CONFIG_DIR") or ""
    real_home = ""
    try:
        real_home = pwd.getpwuid(os.getuid()).pw_dir
    except Exception:  # noqa: BLE001
        real_home = ""
    redirected = bool(config_dir) and os.path.abspath(config_dir) != os.path.join(
        real_home, ".local-operator"
    )
    if real_home and os.path.abspath(home) == real_home and not redirected:
        raise SystemExit(
            "refusing to run against the live home. Use a fresh HOME and "
            "LOCAL_OPERATOR_CONFIG_DIR (see the module docstring), or pass "
            "--allow-live-home if you really mean to measure the operator's own "
            "config root."
        )


def _load() -> dict[str, float]:
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:
        return {"load1": -1.0, "load5": -1.0, "load15": -1.0}
    return {"load1": one, "load5": five, "load15": fifteen}


def _provenance(root: Path) -> dict[str, Any]:
    import local_operator  # noqa: PLC0415

    dirty = _git(root, "status", "--porcelain")
    return {
        "root": str(root),
        "module_file": str(Path(local_operator.__file__).resolve()),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "venv_prefix": sys.prefix,
        "git_head": _git(root, "rev-parse", "HEAD"),
        "git_branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_subject": _git(root, "log", "-1", "--format=%s"),
        "git_dirty_files": len(dirty.splitlines()) if dirty else 0,
        "cpus": os.cpu_count(),
    }


# --------------------------------------------------------------------------
# arm 1: hot path
# --------------------------------------------------------------------------


def _request():
    """The request the stream-recording tests use: enough shape to be real."""
    from local_operator.harness.types import (  # noqa: PLC0415
        AgentTool,
        ChatRequest,
        Message,
        ModelSpec,
        TextContent,
        ToolResult,
    )

    async def _noop(tool_call_id: str, *_args: object) -> ToolResult:
        return ToolResult(tool_call_id=tool_call_id, tool_name="stub", content=[])

    return ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="claude-opus-5"),
        system_blocks=["Persona.", "## Available tools\ntools", "env"],
        messages=[Message(role="user", content=[TextContent(text="hi " * 50)])],
        tools=[
            AgentTool(
                name="bash",
                description="run",
                parameters={"type": "object", "properties": {"command": {"type": "string"}}},
                execute=_noop,
            )
        ],
    )


#: A three-step cycle so every output-delta type is in every stream: the
#: reasoning branch and the ``_OUTPUT_DELTA_TYPES`` branch in the loop are
#: different code, and a single-type stream would measure only one of them.
_CYCLE = ("text_delta", "reasoning_delta", "tool_call_delta")


def _events(n: int) -> list[Any]:
    """``n`` mixed output deltas, deterministic and identical for every root."""
    from local_operator.harness.types import (  # noqa: PLC0415
        StreamReasoningDelta,
        StreamTextDelta,
        StreamToolCallDelta,
    )

    out: list[Any] = []
    for i in range(n):
        kind = _CYCLE[i % len(_CYCLE)]
        if kind == "text_delta":
            out.append(StreamTextDelta(delta="tok "))
        elif kind == "reasoning_delta":
            out.append(StreamReasoningDelta(delta="think "))
        else:
            out.append(StreamToolCallDelta(index=0, name="bash", argument_delta="x"))
    return out


class _CountingEvent:
    """A duck-typed stand-in for the pydantic stream events that COUNTS the
    attribute lookups the hot loop performs on each event.

    The real events are pydantic models, which is the right thing to time and
    the wrong thing to instrument: ``.type`` is a real field and resolves through
    ``object.__getattribute__``, while ``.usage``/``.stop_reason``/``.error``/
    ``.served_provider`` are UNSET optional fields and take pydantic's
    ``__getattr__`` miss path. Neither reports how often it was asked. This class
    answers the same five names with the same presence/absence and counts each
    read, so the loop's dispatch is measured rather than argued about.

    It is used ONLY in a count phase, never in a timing phase: ``__getattribute__``
    makes every read slower, and the point of the phase is the count.
    """

    #: The five names ``_record_stream`` reads off every event.
    _WATCHED = ("type", "usage", "stop_reason", "error", "served_provider")
    counts: dict[str, int] = {}

    def __init__(self, kind: str) -> None:
        d = object.__getattribute__(self, "__dict__")
        d["type"] = kind
        d["delta"] = "tok "
        d["index"] = 0
        d["name"] = "bash"
        d["argument_delta"] = "x"

    def __getattribute__(self, name: str) -> Any:
        if name in _CountingEvent._WATCHED:
            counts = _CountingEvent.counts
            counts[name] = counts.get(name, 0) + 1
        # ``usage``/``stop_reason``/``error``/``served_provider`` are not in the
        # instance dict, so this raises AttributeError exactly as an unset
        # pydantic field does, and ``getattr(event, name, None)`` sees None.
        return object.__getattribute__(self, name)


def _events_counting(n: int) -> list[Any]:
    """``n`` mixed deltas as counting duck types, same cycle as ``_events``."""
    return [_CountingEvent(_CYCLE[i % len(_CYCLE)]) for i in range(n)]


def _fresh_recorder(db_path: Path):
    """Point the singleton recorder at a temp store, as the unit tests do."""
    from local_operator.analytics.recorder import (  # noqa: PLC0415
        reset_recorder_for_test,
    )
    from local_operator.analytics.store import AnalyticsStore  # noqa: PLC0415

    store = AnalyticsStore(db_path)
    return reset_recorder_for_test(store), store


async def _drain(fn: Any, request: Any, events: list[Any]) -> int:
    """Exactly ``test_stream_recording.py::_drain``: count events, keep none."""

    async def stream():
        for ev in events:
            yield ev

    seen = 0
    async for _ev in fn._record_stream(request, stream()):
        seen += 1
    return seen


def _make_fn(session_id: str):
    from local_operator.model.configure import SessionStreamFn  # noqa: PLC0415

    fn = object.__new__(SessionStreamFn)
    fn._session_id = session_id
    fn._counts_as_child_request = False
    return fn


async def _hot_async(sizes: list[int], repeats: int, tmp: Path) -> dict[str, Any]:
    import time as _time  # noqa: PLC0415

    request = _request()
    recorder, store = _fresh_recorder(tmp / "hot.db")
    results: dict[str, Any] = {}
    for n in sizes:
        events = _events(n)
        fn = _make_fn(f"bench-{n}")
        # Warmup: starts the recorder's writer thread, populates the store's
        # schema, and touches every branch once. Discarded.
        await _drain(fn, request, events)
        await asyncio.sleep(0.05)
        samples: list[int] = []
        for _ in range(repeats):
            t0 = _time.perf_counter_ns()
            seen = await _drain(fn, request, events)
            samples.append(_time.perf_counter_ns() - t0)
            if seen != n:
                raise SystemExit(f"drain forwarded {seen} events, expected {n}")
        samples.sort()
        results[str(n)] = {
            "events": n,
            "repeats": repeats,
            "forwarded": n,
            "per_event_ns_median": statistics.median(samples) / n,
            "per_event_ns_best": samples[0] / n,
            "total_ns_median": statistics.median(samples),
            "total_ns_best": samples[0],
            "samples_ns": samples,
        }
    store.close()
    recorder.close()
    return results


def _slopes(hot: dict[str, Any]) -> dict[str, Any]:
    """Per-event cost with the per-DRAIN fixed cost removed.

    ``total/N`` still carries the per-drain setup (component-char snapshot,
    ``CallSnapshot`` construction, generator start, ``record_call`` enqueue) --
    all real, all identical in both arms, and all invisible in the per-event
    figure once it is divided by N. The slope between two sizes cancels it,
    which is what makes a ~50 ns/event difference measurable at all.
    """
    out: dict[str, Any] = {}
    if "500" in hot and "5000" in hot:
        lo, hi = hot["500"], hot["5000"]
        dn = hi["events"] - lo["events"]
        out["slope_500_5000_ns_median"] = (
            statistics.median(hi["samples_ns"]) - statistics.median(lo["samples_ns"])
        ) / dn
        out["slope_500_5000_ns_best"] = (hi["total_ns_best"] - lo["total_ns_best"]) / dn
    return out


def measure_hot(sizes: list[int], repeats: int, tmp: Path) -> dict[str, Any]:
    started = _load()
    hot = asyncio.run(_hot_async(sizes, repeats, tmp))
    return {"load_start": started, "load_end": _load(), "sizes": hot, **_slopes(hot)}


# --------------------------------------------------------------------------
# arm 1b: counted clock reads (structural; load-independent)
# --------------------------------------------------------------------------


def _clock_child_arm(root: Path, sizes: list[int], repeats: int, tmp: Path) -> dict[str, Any]:
    """Count ``time.monotonic()`` calls in the loop, one child per stream size.

    ``local_operator.model.configure`` does ``import time`` and calls
    ``time.monotonic()``, so replacing that module's ``time`` attribute with a
    proxy that counts is exact and local to this measurement -- other modules
    keep the real module. Run in a child so the proxy cannot perturb the timing
    arm.
    """
    out: dict[str, Any] = {"sizes": {}}
    for n in sizes:
        out["sizes"][str(n)] = _child("count", [], root=root, tmp=tmp, events=n, repeats=repeats)
    return out


def _count_child(root: Path, events: int, repeats: int, tmp: Path) -> dict[str, Any]:
    import local_operator.model.configure as configure  # noqa: PLC0415

    real = time

    class _Counting:
        calls = 0

        def monotonic(self) -> float:
            _Counting.calls += 1
            return real.monotonic()

        def __getattr__(self, name: str) -> Any:
            return getattr(real, name)

    async def run() -> dict[str, Any]:
        request = _request()
        recorder, store = _fresh_recorder(Path(tmp) / f"count-{events}.db")
        evs = _events(events)
        fn = _make_fn(f"count-{events}")
        await _drain(fn, request, evs)
        await asyncio.sleep(0.05)
        configure.time = _Counting()  # type: ignore[assignment]
        try:
            for _ in range(repeats):
                await _drain(fn, request, evs)
        finally:
            configure.time = real  # type: ignore[assignment]
        monotonic = {
            "monotonic_calls": _Counting.calls,
            "monotonic_calls_per_drain": _Counting.calls / repeats,
            "monotonic_calls_per_output_delta": (_Counting.calls / repeats) / events,
        }
        # Second phase: the SAME loop over duck-typed events that count their own
        # attribute reads. The pydantic events cannot report how many times the
        # loop looked at ``.type``, and the design's other half -- "the existing
        # two getattr(event, "type", "") lookups collapse into one" -- is exactly
        # a claim about that count. Counting it is load-independent, like the
        # clock read above.
        _CountingEvent.counts = {}
        counted = _events_counting(events)
        await _drain(fn, request, counted)
        lookups = dict(_CountingEvent.counts)
        per_drain = {k: v / 1 for k, v in lookups.items()}
        store.close()
        recorder.close()
        return {
            "events": events,
            "repeats": repeats,
            **monotonic,
            "attr_lookups_per_drain": per_drain,
            "type_lookups_per_drain": per_drain.get("type", 0),
            "type_lookups_per_output_delta": per_drain.get("type", 0) / events,
            "miss_lookups_per_drain": {k: v for k, v in per_drain.items() if k != "type"},
        }

    return asyncio.run(run())


# --------------------------------------------------------------------------
# arm 2: migration
# --------------------------------------------------------------------------


def _free_mb(path: Path) -> float:
    stat = os.statvfs(str(path))
    return stat.f_bavail * stat.f_frsize / 1e6


def _copy_fixture(fixture: Path, dest: Path, allow_plain: bool) -> dict[str, Any]:
    """APFS clonefile where available, so a 631 MB ledger copies in O(1).

    A clonefile SHARES blocks with the fixture and materialises only what is
    written, which is why this is the only safe way to copy a 631 MB ledger on a
    host that may have well under a gigabyte free -- measured here: ``cp -c`` of
    the fixture moved free space by 0.4 MB, and the first write to the clone by
    1.9 MB. A plain ``shutil.copyfile`` needs the full 631 MB and will either
    fail or fill the volume, so it is refused unless ``--allow-plain-copy`` is
    passed on purpose.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    free_before = _free_mb(dest.parent)
    cp = shutil.which("cp")
    used_clone = False
    if cp and sys.platform == "darwin":
        proc = subprocess.run([cp, "-c", str(fixture), str(dest)], capture_output=True, text=True)
        used_clone = proc.returncode == 0
        if not used_clone and not allow_plain:
            raise SystemExit(
                f"clonefile failed on {fixture}: {proc.stderr.strip()}. Refusing to make a plain "
                f"631 MB copy with {free_before:.0f} MB free; pass --allow-plain-copy to override."
            )
    if not used_clone:
        if not allow_plain:
            raise SystemExit(
                "clonefile is unavailable on this platform, so copying the fixture needs its full "
                f"{fixture.stat().st_size / 1e6:.0f} MB. Pass --allow-plain-copy to override."
            )
        if free_before < fixture.stat().st_size / 1e6 + 1500:
            raise SystemExit(
                f"refusing a plain copy: {free_before:.0f} MB free, need "
                f"{fixture.stat().st_size / 1e6 + 1500:.0f} MB."
            )
        shutil.copyfile(fixture, dest)
    return {
        "clone": used_clone,
        "bytes_before": dest.stat().st_size,
        "free_mb_before": free_before,
        "free_mb_after_copy": _free_mb(dest.parent),
    }


def _db_files(path: Path) -> dict[str, int]:
    sizes = {}
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(path) + suffix)
        sizes[suffix or "db"] = p.stat().st_size if p.exists() else -1
    return sizes


def _connect_child(root: Path, db: Path, passes: int) -> dict[str, Any]:
    """Time ``_connect()`` on the real code path. One child per measurement, so
    "first" means the process's first connection, not the fixture's."""
    from local_operator.analytics.store import AnalyticsStore  # noqa: PLC0415

    before = _db_files(db)
    times: list[float] = []
    # Declared BEFORE the loop: a zero-pass run (or a future early exit) must
    # still produce a report, and an unbound name here would crash the arm
    # rather than say "nothing was measured".
    calls_rows: int | None = None
    sd_rows: int | None = None
    sd_cols: list[str] = []
    calls_cols: list[str] = []
    has_rollup: bool | None = None
    page_count: int | None = None
    journal: str | None = None
    for i in range(passes):
        # A fresh store object per pass: the connection is per-thread and cached
        # on ``self._local``, so this is a genuine connect each time.
        store = AnalyticsStore(db)
        t0 = time.perf_counter()
        conn = store._connect()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if conn is None:
            raise SystemExit(f"connect failed for {db}")
        times.append(elapsed_ms)
        if i == 0:
            calls_rows = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
            sd_rows = conn.execute("SELECT COUNT(*) FROM session_daily").fetchone()[0]
            sd_cols = [str(r[1]) for r in conn.execute("PRAGMA table_info(session_daily)")]
            calls_cols = [str(r[1]) for r in conn.execute("PRAGMA table_info(calls)")]
            has_rollup = store._has_session_daily  # noqa: SLF001 -- private by design
            page_count = conn.execute("PRAGMA page_count").fetchone()[0]
            journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        store.close()
    after = _db_files(db)
    return {
        "passes": passes,
        "connect_ms": times,
        "first_connect_ms": times[0],
        "second_connect_ms": times[1] if len(times) > 1 else None,
        "calls_rows": calls_rows,
        "session_daily_rows": sd_rows,
        "session_daily_decode_cols": [c for c in sd_cols if c.startswith("decode_")],
        "calls_decode_cols": [c for c in calls_cols if c.startswith("decode_")],
        "has_session_daily": has_rollup,
        "page_count": page_count,
        "journal_mode": journal,
        "files_before": before,
        "files_after": after,
    }


def _raw_child(root: Path, db: Path, which: str) -> dict[str, Any]:
    """Time the REAL migration bodies on a raw connection, isolated from the
    schema script and the file open.

    ``which='session_daily'`` calls the change's own
    ``AnalyticsStore._migrate_session_daily`` -- the new pass, alone.
    ``which='migrate'`` calls ``AnalyticsStore._migrate`` -- every ALTER of a
    first open, including that pass. Each is run TWICE: the second run is the
    idempotent no-op floor.

    ``which='warm'`` decomposes the cost, because the first write to a
    freshly-cloned 631 MB file is expensive on its own and would otherwise be
    charged to whichever statement happened to be first. It times
    ``PRAGMA journal_mode=WAL`` (the write every open pays, before any ALTER),
    then each ALTER statement individually, reading the statements from the
    module's OWN column lists so the decomposition cannot drift from the code.
    """
    from local_operator import analytics as analytics_pkg  # noqa: PLC0415
    from local_operator.analytics import store as store_mod  # noqa: PLC0415
    from local_operator.analytics.store import AnalyticsStore  # noqa: PLC0415

    store = AnalyticsStore(db)  # not connected; only used as the method carrier
    conn = sqlite3.connect(str(db))
    try:
        pre_cols = [str(r[1]) for r in conn.execute("PRAGMA table_info(session_daily)")]
        pre_calls = [str(r[1]) for r in conn.execute("PRAGMA table_info(calls)")]
        pre_pages = conn.execute("PRAGMA page_count").fetchone()[0]
        pre_rows = conn.execute("SELECT COUNT(*) FROM session_daily").fetchone()[0]
        if which == "warm":
            t0 = time.perf_counter()
            conn.execute("PRAGMA journal_mode=WAL")
            wal_ms = (time.perf_counter() - t0) * 1000.0
            statements: list[dict[str, Any]] = []
            calls_cols = set(pre_calls)
            sd_cols = set(pre_cols)
            for name, definition in getattr(store_mod, "_MIGRATION_COLUMNS", ()):
                if name in calls_cols:
                    continue
                t0 = time.perf_counter()
                conn.execute(f"ALTER TABLE calls ADD COLUMN {name} {definition}")
                statements.append(
                    {"table": "calls", "column": name, "ms": (time.perf_counter() - t0) * 1000.0}
                )
            for name, definition in getattr(store_mod, "_SESSION_DAILY_MIGRATION_COLUMNS", ()):
                if name in sd_cols:
                    continue
                t0 = time.perf_counter()
                conn.execute(f"ALTER TABLE session_daily ADD COLUMN {name} {definition}")
                statements.append(
                    {
                        "table": "session_daily",
                        "column": name,
                        "ms": (time.perf_counter() - t0) * 1000.0,
                    }
                )
            conn.commit()
            post_pages = conn.execute("PRAGMA page_count").fetchone()[0]
            post_rows = conn.execute("SELECT COUNT(*) FROM session_daily").fetchone()[0]
            post_cols = [str(r[1]) for r in conn.execute("PRAGMA table_info(session_daily)")]
            post_calls = [str(r[1]) for r in conn.execute("PRAGMA table_info(calls)")]
            # A second WAL switch on the now-warm file, as the reference that
            # says how much of the first one was the write and how much was the
            # clone materialising.
            t0 = time.perf_counter()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            checkpoint_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "which": which,
                "present": True,
                "journal_mode_ms": wal_ms,
                "statements": statements,
                "alter_total_ms": sum(s["ms"] for s in statements),
                "checkpoint_ms": checkpoint_ms,
                "added_session_daily": sorted(set(post_cols) - set(pre_cols)),
                "added_calls": sorted(set(post_calls) - set(pre_calls)),
                "page_count_before": pre_pages,
                "page_count_after": post_pages,
                "session_daily_rows_before": pre_rows,
                "session_daily_rows_after": post_rows,
                "analytics_has_data": bool(getattr(analytics_pkg, "__file__", "")),
            }
        if which == "session_daily" and not hasattr(AnalyticsStore, "_migrate_session_daily"):
            # Pre-change tree: the rollup ALTER pass does not exist at all, which
            # is a structural fact about the arm, not a failure to measure it.
            return {"which": which, "present": False}
        times: list[float] = []
        for _ in range(2):
            t0 = time.perf_counter()
            if which == "session_daily":
                AnalyticsStore._migrate_session_daily(conn)
            else:
                store._migrate(conn)
            times.append((time.perf_counter() - t0) * 1000.0)
        conn.commit()
        post_cols = [str(r[1]) for r in conn.execute("PRAGMA table_info(session_daily)")]
        post_calls = [str(r[1]) for r in conn.execute("PRAGMA table_info(calls)")]
        post_pages = conn.execute("PRAGMA page_count").fetchone()[0]
        post_rows = conn.execute("SELECT COUNT(*) FROM session_daily").fetchone()[0]
        return {
            "which": which,
            "present": True,
            "first_ms": times[0],
            "second_ms": times[1],
            "alter_ms": times[0] - times[1],
            "added_session_daily": sorted(set(post_cols) - set(pre_cols)),
            "added_calls": sorted(set(post_calls) - set(pre_calls)),
            "page_count_before": pre_pages,
            "page_count_after": post_pages,
            "session_daily_rows_before": pre_rows,
            "session_daily_rows_after": post_rows,
        }
    finally:
        conn.close()


def measure_migration(
    root: Path, fixture: Path, tmp: Path, passes: int, allow_plain: bool
) -> dict[str, Any]:
    """One clone at a time, deleted as soon as it is measured.

    A clonefile shares blocks with the fixture, but a host running a fleet of
    sessions can be down to a few hundred megabytes free, so peak usage is held
    at ONE clone rather than however many the arm happens to name.
    """
    out: dict[str, Any] = {"load_start": _load(), "passes": passes, "free_mb_start": _free_mb(tmp)}
    c1 = tmp / "connect-first.db"
    out["connect"] = {
        **_copy_fixture(fixture, c1, allow_plain),
        **_child("connect", [str(c1)], root=root, passes=passes),
    }
    out["connect"]["files_after_open"] = _db_files(c1)
    out["connect"]["free_mb_after_open"] = _free_mb(tmp)
    c1.unlink(missing_ok=True)
    for name in ("-wal", "-shm"):
        Path(str(c1) + name).unlink(missing_ok=True)
    for key, which in (("raw_session_daily", "session_daily"), ("raw_migrate", "migrate")):
        clone = tmp / f"raw-{which}.db"
        meta = _copy_fixture(fixture, clone, allow_plain)
        cell = {**meta, **_child("raw", [str(clone), which], root=root)}
        cell["files_after"] = _db_files(clone)
        out[key] = cell
        clone.unlink(missing_ok=True)
        for name in ("-wal", "-shm"):
            Path(str(clone) + name).unlink(missing_ok=True)
    warm = tmp / "raw-warm.db"
    out["raw_warm"] = {
        **_copy_fixture(fixture, warm, allow_plain),
        **_child("raw", [str(warm), "warm"], root=root),
    }
    out["raw_warm"]["files_after"] = _db_files(warm)
    warm.unlink(missing_ok=True)
    for name in ("-wal", "-shm"):
        Path(str(warm) + name).unlink(missing_ok=True)
    out["free_mb_end"] = _free_mb(tmp)
    out["load_end"] = _load()
    return out


# --------------------------------------------------------------------------
# arm 3: reads
# --------------------------------------------------------------------------


def _busiest_session(db: Path) -> tuple[str, int]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT session_id, COUNT(*) AS n FROM calls"
            " GROUP BY session_id ORDER BY n DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return (str(row[0]), int(row[1])) if row else ("", 0)


def _timed(fn, repeats: int) -> dict[str, Any]:
    samples: list[float] = []
    last: Any = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    payload: dict[str, Any] = {
        "runs": repeats,
        "ms_median": statistics.median(samples),
        "ms_best": samples[0],
        "samples_ms": samples,
    }
    try:
        payload["rows"] = len(last) if last is not None else 0
    except TypeError:
        payload["rows"] = -1
    return payload


def measure_reads(
    root: Path,
    fixture: Path,
    tmp: Path,
    fast: int,
    slow: int,
    allow_plain: bool,
    skip_new_reads: bool = False,
) -> dict[str, Any]:
    import time as _time  # noqa: PLC0415

    from local_operator.analytics.store import AnalyticsStore  # noqa: PLC0415

    db = tmp / "reads.db"
    meta = _copy_fixture(fixture, db, allow_plain)
    busy_id, busy_calls = _busiest_session(db)
    store = AnalyticsStore(db)
    store._connect()  # the migration this arm reads AFTER
    out: dict[str, Any] = {
        **meta,
        "load_start": _load(),
        "busiest_session": busy_id,
        "busiest_calls": busy_calls,
        "has_model_rates": hasattr(store, "model_rates"),
        "has_session_daily": store._has_session_daily,  # noqa: SLF001
    }

    def aggregate_all() -> Any:
        return store.aggregate()

    agg = _timed(aggregate_all, fast)
    agg["last_aggregate_source"] = store.last_aggregate_source
    agg["last_aggregate_refusal"] = store.last_aggregate_refusal
    out["aggregate_all_time"] = agg

    out["daily_series_30"] = _timed(lambda: store.daily_series(30), fast)
    out["series_totals_30"] = _timed(lambda: store.series_totals(daily_days=30), fast)
    out["session_report_busiest"] = _timed(lambda: store.session_report(busy_id), slow)

    if hasattr(store, "model_rates") and not skip_new_reads:
        now_ms = int(_time.time() * 1000)
        since = now_ms - 30 * 24 * 60 * 60 * 1000
        out["model_rates_unbounded"] = _timed(lambda: store.model_rates(), slow)
        out["model_rates_30d"] = _timed(lambda: store.model_rates(since_ms=since), slow)
    store.close()
    out["load_end"] = _load()
    out["free_mb_end"] = _free_mb(tmp)
    db.unlink(missing_ok=True)
    for name in ("-wal", "-shm"):
        Path(str(db) + name).unlink(missing_ok=True)
    return out


# --------------------------------------------------------------------------
# child dispatch
# --------------------------------------------------------------------------


def _child(mode: str, argv: list[str], **kw: Any) -> Any:
    """Run one measurement in a fresh interpreter of the SAME venv.

    A child rather than an in-process call because both remaining arms need a
    state that exists once per process: a first ``_connect()``, and an unshared
    ``configure.time``.
    """
    env = _child_env(Path(kw["root"]))
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--root",
        str(kw["root"]),
        "--_child",
        mode,
        *argv,
        "--_tmp",
        str(kw.get("tmp", "")),
    ]
    if kw.get("passes"):
        cmd += ["--_passes", str(kw["passes"])]
    if kw.get("events"):
        cmd += ["--_events", str(kw["events"])]
    if kw.get("repeats"):
        cmd += ["--_repeats", str(kw["repeats"])]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(kw["root"]))
    if proc.returncode != 0:
        raise SystemExit(f"child {mode} failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _child_env(root: Path) -> dict[str, str]:
    """Isolated env for a child: fresh HOME-derived roots, this tree on the path.

    Built from the parent's environment, which the caller has already stripped
    with ``env -i`` plus a fresh ``HOME``/``LOCAL_OPERATOR_CONFIG_DIR``. The
    ``CMUX_*``/``LOP_*`` prefixes a parent ``lop`` exports are dropped here too,
    because the child product reads them (a child runtime would otherwise
    inherit another session's provider and model).

    The notification gate goes through ``agent_shell.harness_child_env`` rather
    than being set here by hand, for the reason ``bench_base_overhead.py`` gives:
    a bench child that drives a real session can otherwise put a mock
    completion on the operator's lock screen, and one helper is what stops the
    next bench from remembering half of it. It also declares the child a harness
    child, so a bench run from inside an agent's own shell measures the arm
    instead of failing on the agent-shell marker.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("CMUX_") and not k.startswith("LOP_")
    }
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TERM", "xterm-256color")
    sys.path.insert(0, str(root))
    try:
        from local_operator.agent_shell import harness_child_env

        return harness_child_env(env)
    except Exception:  # noqa: BLE001 — a bench must not fail on a missing gate helper
        # Fail SAFE rather than open: without the helper the gate is set by hand,
        # because the one failure mode that matters here is a child putting a
        # notification on the operator's screen.
        env["LOCAL_OPERATOR_NO_NOTIFICATIONS"] = "1"
        return env


def _child_main(mode: str, args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    _claim_root(root)
    tmp = Path(args._tmp)
    if mode == "count":
        payload = _count_child(root, args._events, args._repeats, tmp)
    elif mode == "connect":
        payload = _connect_child(root, Path(args.argv[0]), args._passes)
    elif mode == "raw":
        payload = _raw_child(root, Path(args.argv[0]), args.argv[1])
    else:
        raise SystemExit(f"unknown child mode {mode}")
    print(json.dumps(payload))
    return 0


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _clock_ns() -> dict[str, Any]:
    best = None
    samples = []
    for _ in range(_CLOCK_SAMPLES):
        t0 = time.perf_counter_ns()
        for _ in range(_CLOCK_BATCH):
            time.monotonic()
        elapsed = time.perf_counter_ns() - t0
        samples.append(elapsed / _CLOCK_BATCH)
        best = elapsed if best is None else min(best, elapsed)
    samples.sort()
    return {
        "samples_ns": samples,
        "ns_median": statistics.median(samples),
        "ns_best": (best / _CLOCK_BATCH) if best is not None else None,
        "samples": _CLOCK_SAMPLES,
        "batch": _CLOCK_BATCH,
    }


def _fmt_row(label: str, *cells: str, widths: tuple[int, ...]) -> str:
    line = f"{label:<{widths[0]}}"
    for cell, width in zip(cells, widths[1:]):
        line += f"{cell:>{width}}"
    return line


HOT_W = (14, 9, 16, 16, 16)


def report_hot(result: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    sizes = result["hot"]["sizes"]
    old = (baseline or {}).get("hot", {}).get("sizes", {})
    print()
    print("arm 1: hot path -- SessionStreamFn._record_stream, synthetic stream, no network")
    print(
        _fmt_row(
            "events",
            "repeats",
            "per-event ns",
            "per-event ns",
            "slope ns/ev",
            widths=HOT_W,
        )
    )
    print(
        _fmt_row(
            "",
            "",
            "median",
            "best",
            "500->5000",
            widths=HOT_W,
        )
    )
    print("-" * sum(HOT_W))
    for key in sorted(sizes, key=int):
        cell = sizes[key]
        slope = ""
        if key == "5000":
            hot = result["hot"]
            value = hot.get("slope_500_5000_ns_best")
            if value is None:
                value = hot.get("slope_500_5000_ns_median", float("nan"))
            slope = f"{value:.1f}"
        print(
            _fmt_row(
                key,
                str(cell["repeats"]),
                f"{cell['per_event_ns_median']:.1f}",
                f"{cell['per_event_ns_best']:.1f}",
                slope,
                widths=HOT_W,
            )
        )
    clock = result["clock_ns"]
    print(
        f"bare time.monotonic(): median {clock['ns_median']:.1f} ns, "
        f"best {clock['ns_best']:.1f} ns ({clock['samples']} x {clock['batch']} calls)"
    )
    if baseline:
        print()
        # One calibration for both arms: the bare clock is a property of the host
        # and the moment, not of either tree, so a delta is expressed against the
        # MEAN of the two runs' measured clock rather than against whichever one
        # happens to be the baseline.
        clocks = [
            c
            for c in (
                baseline.get("clock_ns", {}).get("ns_median"),
                result.get("clock_ns", {}).get("ns_median"),
            )
            if c
        ]
        ref = statistics.mean(clocks) if clocks else float("nan")
        print(
            f"arm 1 delta (this run - baseline), ns per event  " f"[median clock ref {ref:.1f} ns]"
        )
        best_clocks = [
            c
            for c in (
                baseline.get("clock_ns", {}).get("ns_best"),
                result.get("clock_ns", {}).get("ns_best"),
            )
            if c
        ]
        ref_best = statistics.mean(best_clocks) if best_clocks else float("nan")
        print(
            _fmt_row(
                "events",
                "median ns",
                "best ns",
                "best / clock(best)",
                "slope delta ns",
                widths=(14, 16, 16, 22, 18),
            )
        )
        print(
            f"  columns 1-2 are per-event deltas; column 3 divides the BEST-based delta by "
            f"{ref_best:.1f} ns (mean of both runs' bare-clock bests)"
        )
        print("-" * 86)
        clock = ref
        for key in sorted(sizes, key=int):
            if key not in old:
                continue
            d_med = sizes[key]["per_event_ns_median"] - old[key]["per_event_ns_median"]
            d_best = sizes[key]["per_event_ns_best"] - old[key]["per_event_ns_best"]
            slope = ""
            if key == "5000":
                # Prefer the min-based slope when the run carries one (a merged
                # run does): on this host the median-based slope is dominated by
                # descheduled samples, and a delta of two noisy slopes says
                # nothing.
                def _slope(blob: dict[str, Any] | None) -> float | None:
                    hot = (blob or {}).get("hot") or {}
                    return hot.get("slope_500_5000_ns_best") or hot.get("slope_500_5000_ns_median")

                now_slope, old_slope = _slope(result), _slope(baseline)
                if now_slope is not None and old_slope is not None:
                    slope = f"{now_slope - old_slope:+.1f}"
            print(
                _fmt_row(
                    key,
                    f"{d_med:+.1f}",
                    f"{d_best:+.1f}",
                    f"{d_best / ref_best:.2f} x clock",
                    slope,
                    widths=(14, 16, 16, 22, 18),
                )
            )


def report_clock(result: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    """The decisive arm for the clock-read half of the claim.

    "At most ONE ``time.monotonic()`` per output delta" is a COUNT. A count does
    not move when the host is at load average 200, which is why this table is the
    one to read when the wall-time table looks noisy -- and why the two are
    reported side by side instead of one being trusted.
    """
    print()
    print("arm 1b: counted time.monotonic() calls in the loop (load-independent)")
    old = (baseline or {}).get("clock", {}).get("sizes", {})
    print(
        _fmt_row(
            "events",
            "repeats",
            "calls/drain",
            "calls/delta",
            "baseline calls/delta",
            widths=(14, 9, 16, 16, 22),
        )
    )
    print("-" * 77)
    for key in sorted(result["clock"]["sizes"], key=int):
        cell = result["clock"]["sizes"][key]
        before = old.get(key, {}).get("monotonic_calls_per_output_delta")
        print(
            _fmt_row(
                key,
                str(cell["repeats"]),
                f"{cell['monotonic_calls_per_drain']:.2f}",
                f"{cell['monotonic_calls_per_output_delta']:.4f}",
                f"{before:.4f}" if before is not None else "",
                widths=(14, 9, 16, 16, 22),
            )
        )
    if not any("type_lookups_per_drain" in c for c in result["clock"]["sizes"].values()):
        return
    print()
    print("arm 1c: counted getattr(event, 'type', '') lookups (same run, duck-typed events)")
    print(
        _fmt_row(
            "events",
            "type/drain",
            "type/delta",
            "miss/drain",
            "baseline type/delta",
            widths=(14, 14, 16, 14, 22),
        )
    )
    print("-" * 80)
    for key in sorted(result["clock"]["sizes"], key=int):
        cell = result["clock"]["sizes"][key]
        if "type_lookups_per_drain" not in cell:
            continue
        before = old.get(key, {}).get("type_lookups_per_output_delta")
        # Already per-drain: the counting phase drains ONCE, so no division.
        misses = sum(cell.get("miss_lookups_per_drain", {}).values())
        print(
            _fmt_row(
                key,
                f"{cell['type_lookups_per_drain']:.0f}",
                f"{cell['type_lookups_per_output_delta']:.4f}",
                f"{misses:.0f}",
                f"{before:.4f}" if before is not None else "",
                widths=(14, 14, 16, 14, 22),
            )
        )


MIG_W = (34, 14, 14, 14)


def _mig_rows(result: dict[str, Any]) -> list[tuple[str, float, float, float]]:
    mig = result["migration"]
    conn = mig["connect"]
    rows = [
        ("connect first (schema+ALTERs)", conn["first_connect_ms"], 0.0, 0.0),
        ("connect second (idempotent)", conn["second_connect_ms"] or 0.0, 0.0, 0.0),
        ("_migrate() first (isolated)", mig["raw_migrate"]["first_ms"], 0.0, 0.0),
        ("_migrate() second (no-op)", mig["raw_migrate"]["second_ms"], 0.0, 0.0),
    ]
    sd = mig["raw_session_daily"]
    if sd.get("present"):
        rows.append(("_migrate_session_daily() first", sd["first_ms"], 0.0, 0.0))
        rows.append(("_migrate_session_daily() second", sd["second_ms"], 0.0, 0.0))
    return rows


def report_migration(result: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    mig = result["migration"]
    conn = mig["connect"]
    print()
    print("arm 2: migration -- first open of identical cp -c clones of the real ledger")
    print(_fmt_row("measurement", "this run ms", "delta ms", "x", widths=(34, 14, 14, 14)))
    print("-" * sum(MIG_W))
    old_rows = {r[0]: r[1] for r in _mig_rows(baseline)} if baseline else {}
    for label, ms, _a, _b in _mig_rows(result):
        before = old_rows.get(label)
        if before is None:
            print(_fmt_row(label, f"{ms:.2f}", "", "", widths=MIG_W))
        else:
            print(
                _fmt_row(
                    label,
                    f"{ms:.2f}",
                    f"{ms - before:+.2f}",
                    f"{(ms / before if before else 0):.2f}",
                    widths=MIG_W,
                )
            )
    print()
    print("  isolated ALTER evidence (real migration bodies, raw connection)")
    for which in ("raw_migrate", "raw_session_daily"):
        cell = mig[which]
        if not cell.get("present"):
            print(f"    {which:<34} ABSENT on this root (pass does not exist)")
            continue
        print(
            f"    {which:<34} added session_daily={cell['added_session_daily']} "
            f"added calls={len(cell['added_calls'])}"
        )
        print(
            f"    {'':<34} alter ms={cell['alter_ms']:+.3f} "
            f"rows {cell['session_daily_rows_before']}->{cell['session_daily_rows_after']} "
            f"pages {cell['page_count_before']}->{cell['page_count_after']}"
        )
    warm = mig.get("raw_warm")
    if warm and warm.get("present"):
        print()
        print("  per-statement decomposition after the WAL switch (warm first write)")
        print(f"    PRAGMA journal_mode=WAL (first write)   {warm['journal_mode_ms']:8.2f} ms")
        for stmt in warm["statements"]:
            print(f"    ALTER {stmt['table']}.{stmt['column']:<20}       {stmt['ms']:8.2f} ms")
        print(f"    {'ALTER total':<40}{warm['alter_total_ms']:8.2f} ms")
        print(
            f"    rows {warm['session_daily_rows_before']}->{warm['session_daily_rows_after']}  "
            f"pages {warm['page_count_before']}->{warm['page_count_after']}  "
            f"added calls={len(warm['added_calls'])}"
        )
    fa = conn["files_after"]
    fb = conn["files_before"]
    print()
    print(
        f"  clone size {fb['db'] / 1e6:.2f} MB -> {fa['db'] / 1e6:.2f} MB "
        f"(delta {(fa['db'] - fb['db']) / 1e6:+.2f} MB), "
        f"wal={fa['-wal']} shm={fa['-shm']} cloned={mig['connect']['clone']}"
    )
    print(
        f"  free space: start {mig['free_mb_start']:.0f} MB, after connect clone "
        f"{conn.get('free_mb_after_open', float('nan')):.0f} MB, end {mig['free_mb_end']:.0f} MB"
    )
    print(
        f"  decode columns after open: session_daily={conn['session_daily_decode_cols']} "
        f"calls={conn['calls_decode_cols']} has_session_daily={conn['has_session_daily']} "
        f"journal={conn['journal_mode']} rows={conn['calls_rows']}"
    )


READ_W = (30, 8, 14, 14, 22)


def _read_cells(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    r = result["reads"]
    cells = [
        ("aggregate() all-time", r["aggregate_all_time"]),
        ("  last_aggregate_source", {"extra": r["aggregate_all_time"]["last_aggregate_source"]}),
        ("  last_aggregate_refusal", {"extra": r["aggregate_all_time"]["last_aggregate_refusal"]}),
        ("daily_series(30)", r["daily_series_30"]),
        ("series_totals(30)", r["series_totals_30"]),
        ("session_report(busiest)", r["session_report_busiest"]),
        ("model_rates() unbounded", r.get("model_rates_unbounded", {})),
        ("model_rates(since 30d)", r.get("model_rates_30d", {})),
    ]
    return cells


def report_reads(result: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    r = result["reads"]
    old = {label: cell for label, cell in _read_cells(baseline)} if baseline else {}
    print()
    print(
        f"arm 3: reads -- migrated clone of the real ledger (busiest session "
        f"{r['busiest_session']}, {r['busiest_calls']} calls)"
    )
    print(_fmt_row("measurement", "runs", "median ms", "best ms", "delta median ms", widths=READ_W))
    print("-" * sum(READ_W))
    for label, cell in _read_cells(result):
        if "extra" in cell:
            print(_fmt_row(label, "", "", "", f"{cell['extra'] or '(none)'}", widths=READ_W))
            continue
        if not cell:
            print(_fmt_row(label, "", "absent", "", "", widths=READ_W))
            continue
        before = old.get(label, {}).get("ms_median")
        delta = f"{cell['ms_median'] - before:+.2f}" if before is not None else ""
        print(
            _fmt_row(
                label,
                str(cell["runs"]),
                f"{cell['ms_median']:.2f}",
                f"{cell['ms_best']:.2f}",
                delta,
                widths=READ_W,
            )
        )
    print(f"  has_model_rates={r['has_model_rates']} has_session_daily={r['has_session_daily']}")


def report(result: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    prov = result["provenance"]
    print(
        f"bench_tps_overhead  root={prov['root']}\n"
        f"  python {prov['python']}  venv={prov['venv_prefix']}\n"
        f"  module={prov['module_file']}\n"
        f"  git {prov['git_branch']} {prov['git_head'][:12]} "
        f"({prov['git_subject']})  dirty_files={prov['git_dirty_files']}"
    )
    print(
        f"  load average start {result['load_start']['load1']:.2f} "
        f"{result['load_start']['load5']:.2f} {result['load_start']['load15']:.2f}"
        f"  end {result['load_end']['load1']:.2f} "
        f"{result['load_end']['load5']:.2f} {result['load_end']['load15']:.2f}"
        f"  cpus={prov['cpus']}"
    )
    if "fixture" in result:
        fx = result["fixture"]
        print(
            f"  fixture {fx['path']}  bytes={fx['bytes']}  sha256={fx['sha256'][:16]}  "
            f"calls={fx['calls']}  session_daily={fx['session_daily']}"
        )
    if baseline:
        print(
            f"  baseline: {baseline['provenance']['root']} "
            f"{baseline['provenance']['git_head'][:12]}"
        )
    for arm in result["arms"]:
        if arm == "hot":
            report_hot(result, baseline)
        elif arm == "clock":
            report_clock(result, baseline)
        elif arm == "migration":
            report_migration(result, baseline)
        elif arm == "reads":
            report_reads(result, baseline)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _fixture_meta(fixture: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with fixture.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    conn = sqlite3.connect(f"file:{fixture}?mode=ro", uri=True)
    try:
        calls = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        sd = conn.execute("SELECT COUNT(*) FROM session_daily").fetchone()[0]
    finally:
        conn.close()
    return {
        "path": str(fixture),
        "bytes": fixture.stat().st_size,
        "sha256": digest.hexdigest(),
        "calls": calls,
        "session_daily": sd,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT, help="checkout root to measure")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("/Users/damian/lo-tps-evidence/ledger-copy.db"),
        help="read-only analytics ledger copy; never written to",
    )
    parser.add_argument(
        "--arms",
        default="hot,clock,migration,reads",
        help="comma list of hot,clock,migration,reads",
    )
    parser.add_argument("--events", default="50,500,5000", help="stream sizes for the hot arm")
    parser.add_argument("--repeats", type=int, default=9, help="repeats per stream size")
    parser.add_argument("--connect-passes", type=int, default=2, help="connects timed per clone")
    parser.add_argument("--read-fast", type=int, default=5, help="repeats for the cheap reads")
    parser.add_argument("--read-slow", type=int, default=3, help="repeats for the slow reads")
    parser.add_argument("--keep-tmp", action="store_true", help="keep the clones for inspection")
    parser.add_argument(
        "--tmpdir",
        type=Path,
        default=None,
        help="where the clones live (default: the system temp dir)",
    )
    parser.add_argument(
        "--skip-new-reads",
        action="store_true",
        help="omit model_rates() from the reads arm, to spend the repeats on the four "
        "pre-existing reads instead",
    )
    parser.add_argument(
        "--allow-plain-copy",
        action="store_true",
        help="permit a full 631 MB copy when clonefile is unavailable",
    )
    parser.add_argument("--save", type=Path, default=None, help="write this run as JSON")
    parser.add_argument("--baseline", type=Path, default=None, help="compare against a saved run")
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        default=None,
        metavar=("BEFORE", "AFTER"),
        help="print a paired report from two saved runs and exit",
    )
    parser.add_argument(
        "--allow-live-home",
        action="store_true",
        help="skip the refusal to run against the operator's real config root",
    )
    parser.add_argument("--_child", default="", help=argparse.SUPPRESS)
    parser.add_argument("--_tmp", default="", help=argparse.SUPPRESS)
    parser.add_argument("--_passes", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--_events", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_repeats", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("argv", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._child:
        return _child_main(args._child, args)

    _refuse_live_home(args.allow_live_home)

    if args.compare:
        before = json.loads(args.compare[0].read_text())
        after = json.loads(args.compare[1].read_text())
        print("=" * 78)
        print(f"BEFORE {args.compare[0]}  ->  AFTER {args.compare[1]}")
        print("=" * 78)
        report(after, before)
        return 0

    root = Path(args.root).resolve()
    _claim_root(root)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    sizes = [int(s) for s in args.events.split(",") if s.strip()]
    result: dict[str, Any] = {
        "provenance": _provenance(root),
        "arms": arms,
        "load_start": _load(),
    }
    if "migration" in arms or "reads" in arms:
        result["fixture"] = _fixture_meta(args.fixture)

    tmp_root = Path(
        tempfile.mkdtemp(prefix="lo-bench-tps-", dir=str(args.tmpdir) if args.tmpdir else None)
    )
    try:
        if "hot" in arms or "clock" in arms:
            result["clock_ns"] = _clock_ns()
        if "hot" in arms:
            result["hot"] = measure_hot(sizes, args.repeats, tmp_root)
        if "clock" in arms:
            # The counting proxy is exactly why the child is per-size: every
            # other cell in this file is measured with the real ``time``.
            result["clock"] = _clock_child_arm(root, sizes, args.repeats, tmp_root)
        if "migration" in arms:
            result["migration"] = measure_migration(
                root, args.fixture, tmp_root, args.connect_passes, args.allow_plain_copy
            )
        if "reads" in arms:
            result["reads"] = measure_reads(
                root,
                args.fixture,
                tmp_root,
                args.read_fast,
                args.read_slow,
                args.allow_plain_copy,
                args.skip_new_reads,
            )
        result["load_end"] = _load()
        report(result, json.loads(args.baseline.read_text()) if args.baseline else None)
        if args.save:
            args.save.write_text(json.dumps(result, indent=2))
            print(f"\nsaved to {args.save}")
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            print(f"clones kept under {tmp_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
