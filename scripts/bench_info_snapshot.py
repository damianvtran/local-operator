#!/usr/bin/env python3
"""The `/info` host read: what one panel open costs, and where it goes.

`/info` is the same read on both surfaces — the TUI's `/info` screen and the
desktop panel's `GET /v1/desktop/info` both call
``local_operator.info.collect.collect_snapshot`` — so this measures that call,
the HTTP route that wraps it, and the probes underneath. It exists because the
read was reported as "analytics almost 30 s, session and info panels very
slow": on macOS a single ``top -l1`` dump inside the session listing cost
2,344 ms of a 3,133 ms read.

What is measured, and why each number is separate rather than one total:

1. **Probes** — the batched ``ps``, the ``top`` dump, and (where the tree under
   test has it) the direct per-pid ``proc_pid_rusage`` read. A total cannot
   tell you which of the two mechanisms the tree is actually spending, and the
   whole point of the change is that one of them stops being spent.
2. **``collect_snapshot``** — the real call, first invocation and steady state
   separately. THE FIRST INVOCATION IS THE REPORTED SYMPTOM ("the first open
   after app start"), so it is never averaged into the warm runs and no cache
   may be allowed to hide it.
3. **``GET /v1/desktop/info``** — through the real FastAPI app, and with
   ``--http`` through a real uvicorn server on a loopback socket, first request
   after boot included. The in-process ASGI request is the default because it
   needs no port; the socket measurement is opt-in because it spawns a daemon.
4. **Equivalence** — the direct read's ``ri_phys_footprint`` against the ``top``
   MEM column for the SAME real pids, because the change is only allowed to be
   faster, not different. The differences are reported rather than asserted
   away: what is checked is that every pid the fixture asked about gets a
   number, and that the two readers agree on the quantity (see ``--tolerance``).

The fixture is real: it spawns N processes, publishes real session records for
them into a temp config root through the registry's own ``publish``, and then
measures the real reader over them. It never touches ``~/.local-operator``.
Run it against the previous commit to get a before column — ``--tree`` points
the imports AND the spawned uvicorn at that checkout, and the script prints the
``local_operator`` file it resolved so the tree is never in doubt:

    .venv/bin/python scripts/bench_info_snapshot.py --json bench/info-snapshot-after.json
    git worktree add --detach /tmp/lo-base <previous-sha>
    .venv/bin/python scripts/bench_info_snapshot.py --tree /tmp/lo-base \\
        --json bench/info-snapshot-before.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


#: How far ``ri_resident_size`` and ``ps`` RSS may diverge on a PARKED process
#: before the layout proof is called broken: a page or two of drift is the two
#: readers sampling a moment apart, while a wrong offset would be off by the
#: distance to another field (hundreds of KB to tens of MB).
_RESIDENT_TOLERANCE_BYTES = 64 * 1024

#: Above this 1-minute load per CPU the latency ceilings below are NOT applied,
#: and the artifact says so. A millisecond ceiling calibrated on a quiet laptop
#: is a bet on the machine rather than a statement about the code once the box is
#: oversubscribed several times over — this repo is worked from many concurrent
#: worktrees and has been measured at 15-20x its CPU count, where every wall time
#: here inflates by a multiple (``AGENTS.md``, "Calibrate ceilings from CI, never
#: from your laptop"). The STRUCTURAL checks are applied either way: they are
#: facts about what the read did, not how long the machine took to do it.
_QUIET_LOAD_PER_CPU = 2.0


class CheckFailed(Exception):
    """The measurement itself is void — wrong tree, no daemon, no response."""


#: Verdicts on a measurement that DID happen, in order. A failure here is a
#: finding about the tree under test, not a broken run: the `before` tree is
#: supposed to fail the latency checks, and the artifact it produces is the
#: whole point of running it. So these are collected, written to the artifact
#: and reported on stderr, and the process still exits non-zero.
_CHECKS: list[str] = []
_FAILURES: list[str] = []


def require(condition: bool, message: str) -> None:
    """Stop: without this the numbers below describe something else."""
    if not condition:
        raise CheckFailed(message)


def check(condition: bool, ok: str, fail: str | None = None) -> None:
    """Record a verdict. False is a finding, not an aborted run.

    ``ok`` is what a reader of a passing run should see, ``fail`` what a broken
    one should — two spellings because the honest sentence differs ("12 of 12
    fixture sessions got a footprint" is not the sentence you want printed when
    the answer was 7).
    """
    if condition:
        _CHECKS.append(ok)
    else:
        message = fail or ok
        _CHECKS.append(f"FAILED: {message}")
        _FAILURES.append(message)


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 2)


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 2) if values else 0.0


def _display_path(path: str | Path) -> str:
    """A path as it goes into the artifact: ``~/...`` under the home directory.

    These artifacts travel — published on the PR that cites them, never committed
    (``AGENTS.md`` §7) — and the repo's convention for anything that travels is a
    home-relative path. What the artifact has to record is WHICH CHECKOUT was
    measured — the caller's own layout is not part of the measurement.
    """
    resolved = Path(path).resolve()
    try:
        return "~/" + str(resolved.relative_to(Path.home()))
    except ValueError:
        return str(resolved)


def _select_tree(tree: Path) -> str:
    """Put ``tree`` first on ``sys.path`` and report which file won.

    ``PYTHONPATH``-style insertion happens here rather than by editing the
    capture environment so a before/after pair is one flag apart, and the
    resolved path is returned (and printed) because a benchmark that measured
    the wrong tree is worse than no benchmark.
    """
    sys.path.insert(0, str(tree))
    import local_operator

    resolved = Path(local_operator.__file__).resolve()
    require(
        resolved.is_relative_to(tree.resolve()),
        f"--tree {tree} did not win the import: local_operator resolved to {resolved}",
    )
    return _display_path(resolved)


class Fixture:
    """N real processes with real session records in a temp config root."""

    def __init__(self, count: int) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="lo-info-bench-"))
        (self.root / "run" / "mobile").mkdir(parents=True, exist_ok=True)
        self.processes: list[subprocess.Popen[bytes]] = []
        for index in range(count):
            # stdout/stderr to DEVNULL: an inherited pipe would keep this
            # script's own caller waiting for EOF long after the benchmark ends.
            process = subprocess.Popen(
                # An hour: long enough that no section can outlive the fixture on a
                # loaded box (the runs below add up to minutes under load), and
                # ``close`` kills them regardless.
                [sys.executable, "-c", "import time; time.sleep(3600)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.processes.append(process)
            self._publish(index, process.pid)

    def _publish(self, index: int, pid: int) -> None:
        from local_operator.session.runtime.registry import publish
        from local_operator.session.runtime.types import SessionRecord

        # A record whose owner is not heartbeat-ing classifies as ``wedged``, and
        # ``wedged`` rows are NOT probed — the pid list comes from the records
        # classified ``live``. ``publish`` writes a fresh heartbeat itself, which
        # is why ``refresh`` is a re-publish rather than a timestamp edit: the
        # registry's own write path is the one rule for "this owner is alive".
        publish(
            SessionRecord(
                pid=pid,
                kind="tui",
                session_id=f"{index:012x}",
                conversation_name=f"bench fixture {index}",
                cwd=str(self.root),
                model_label="bench/fixture",
                control_port=43000 + index,
                control_key="0" * 64,
            ),
            root=self.root,
        )

    @property
    def pids(self) -> list[int]:
        return [process.pid for process in self.processes]

    def refresh(self) -> None:
        """Re-publish every record so the fleet is ``live`` for the next section.

        A record whose owner has not reported for ``HEARTBEAT_TIMEOUT_S`` (45 s)
        classifies as ``wedged``, and WEDGED ROWS ARE NOT PROBED — the pid list
        comes from the records classified ``live``. These fixture processes do
        not heartbeat, and the sections below add up to minutes, so without this
        call every section after the first 45 s silently measures a read with
        nothing to measure. That failure is invisible in a duration (it looks
        FAST) and it is why each route measurement also reports how many fixture
        rows came back in the response.
        """
        for index, process in enumerate(self.processes):
            self._publish(index, process.pid)

    def close(self) -> None:
        for process in self.processes:
            process.kill()
        for process in self.processes:
            process.wait(timeout=10)
        shutil.rmtree(self.root, ignore_errors=True)


def time_call(fn: Any, runs: int) -> tuple[list[float], Any]:
    """Run ``fn`` ``runs`` times, returning every duration in ms and the last result."""
    durations: list[float] = []
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        durations.append(_ms(time.perf_counter() - start))
    return durations, result


def measure_probes(fixture: Fixture, runs: int) -> dict[str, Any]:
    """Each probe, timed on its own. This is where the 2.3 s went."""
    from local_operator.mobile.resources import session_resource_usage

    fixture.refresh()

    pids = fixture.pids
    csv = ",".join(str(pid) for pid in pids)

    def run_ps() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["ps", "-o", "pid=,rss=", "-p", csv], capture_output=True, timeout=30, check=False
        )

    ps_durations, _ = time_call(run_ps, runs)

    top_durations: list[float] = []
    top_available = sys.platform == "darwin"
    if top_available:

        def run_top() -> subprocess.CompletedProcess[bytes]:
            # The exact argv ``mobile.resources`` uses, so the number is the
            # dump this read used to pay for rather than a similar-looking one.
            return subprocess.run(
                ["top", "-l1", "-stats", "pid,mem", "-ncols", "2"],
                capture_output=True,
                timeout=60,
                check=False,
            )

        top_durations, _ = time_call(run_top, min(runs, 3))

    try:
        from local_operator.mobile.resources import (
            _darwin_footprint_bytes as direct_probe,
        )
    except ImportError:  # a tree older than the direct read: nothing to time
        direct_probe = None
    direct: dict[str, Any] = {"available": direct_probe is not None}
    if direct_probe is not None and sys.platform == "darwin":
        per_pid_us: list[float] = []
        for pid in pids:
            start = time.perf_counter()
            direct_probe(pid)
            per_pid_us.append(round((time.perf_counter() - start) * 1_000_000, 1))
        direct.update(
            {
                "per_pid_us_median": _median(per_pid_us),
                "per_pid_us_max": round(max(per_pid_us), 1),
                "pids_read": sum(1 for pid in pids if direct_probe(pid) is not None),
                "pids_total": len(pids),
            }
        )

    usage_durations, usage = time_call(lambda: session_resource_usage(list(pids)), runs)
    return {
        "ps_batched_ms": {
            "first": ps_durations[0],
            "median": _median(ps_durations),
            "runs": ps_durations,
        },
        "top_dump_ms": {
            "available": top_available,
            "first": top_durations[0] if top_durations else None,
            "median": _median(top_durations),
            "runs": top_durations,
        },
        "direct_rusage": direct,
        "session_resource_usage_ms": {
            "first": usage_durations[0],
            "median": _median(usage_durations),
            "runs": usage_durations,
        },
        "footprint_by_pid": {
            str(pid): usage[pid].footprint_bytes for pid in sorted(usage) if pid in set(pids)
        },
    }


def measure_snapshot(fixture: Fixture, runs: int) -> dict[str, Any]:
    from local_operator.info.collect import LiveState, collect_snapshot

    fixture.refresh()

    durations: list[float] = []
    snapshot = None
    for _ in range(runs):
        start = time.perf_counter()
        snapshot = collect_snapshot(LiveState(), root=fixture.root)
        durations.append(_ms(time.perf_counter() - start))
    assert snapshot is not None
    rows = snapshot.sessions.lines
    fixture_pids = set(fixture.pids)
    measured = {row.pid: row.footprint_bytes for row in rows if row.pid in fixture_pids}
    return {
        "collect_snapshot_ms": {
            "first": durations[0],
            "median": _median(durations),
            "min": round(min(durations), 2),
            "max": round(max(durations), 2),
            "runs": durations,
        },
        "session_rows": len(rows),
        "fixture_rows": len(measured),
        "fixture_rows_with_footprint": sum(1 for value in measured.values() if value),
        "rss_by_pid": {str(row.pid): row.rss_bytes for row in rows if row.pid in fixture_pids},
        "footprint_by_pid": {str(pid): value for pid, value in sorted(measured.items())},
    }


def measure_blocks(fixture: Fixture, runs: int) -> dict[str, Any]:
    """Each collector on its own, so "should these overlap?" has a number.

    ``collect_snapshot`` runs its blocks one after another, and the question that
    follows every latency fix is whether overlapping the independent ones is
    worth a thread pool. What such a pool could save is the sum of the blocks
    minus the longest of them, so both are recorded rather than an opinion.

    Warm, not cold, and deliberately: ``measure_snapshot`` has already run these
    blocks, so the per-call costs here are the steady-state ones a panel
    re-open pays. The one-time import graph shows up as the difference between
    ``collect_snapshot``'s first invocation and its median.

    Also worth stating in the artifact: the blocks are NOT all independent.
    ``process`` and ``agents`` are built FROM the sessions block's output, so the
    real ceiling is lower than the arithmetic below.
    """
    from local_operator.info.collect import (
        LiveState,
        collect_agents,
        collect_env,
        collect_install,
        collect_process,
        collect_sessions,
    )

    fixture.refresh()
    live = LiveState()
    sessions = collect_sessions(fixture.root, self_pid=os.getpid())
    self_line = next((line for line in sessions.lines if line.is_self), None)

    def time_block(fn: Any) -> list[float]:
        durations: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            fn()
            durations.append(_ms(time.perf_counter() - start))
        return durations

    durations = {
        "install": time_block(lambda: collect_install([])),
        "sessions": time_block(lambda: collect_sessions(fixture.root, self_pid=os.getpid())),
        "process": time_block(
            lambda: collect_process(live, self_line=self_line, errors=[], root=fixture.root)
        ),
        "agents": time_block(lambda: collect_agents(live, [], sessions)),
        "env": time_block(lambda: collect_env(live, [])),
    }
    medians = {name: _median(values) for name, values in durations.items()}
    # `install` and `env` read nothing the session block produces, so those three
    # could overlap one another today; `process`/`agents` must wait for sessions.
    independent = ["install", "sessions", "env"]
    independent_sum = round(sum(medians[name] for name in independent), 2)
    independent_max = max(medians[name] for name in independent)
    return {
        "warm": True,
        "medians_ms": medians,
        "runs": durations,
        "serial_sum_ms": round(sum(medians.values()), 2),
        "independent_blocks": independent,
        "independent_sum_ms": independent_sum,
        "independent_max_ms": independent_max,
        "concurrency_ceiling_ms": round(independent_sum - independent_max, 2),
    }


def _top_mem_by_pid() -> dict[int, int]:
    """The ``top`` MEM column for every pid, as bytes, from one real dump."""
    from local_operator.mobile.resources import _parse_top_footprint

    out = subprocess.run(
        ["top", "-l1", "-stats", "pid,mem", "-ncols", "2"],
        capture_output=True,
        timeout=60,
        check=False,
    ).stdout.decode("utf-8", "replace")
    every_pid = {
        int(line.split()[0])
        for line in out.splitlines()
        if line.split() and line.split()[0].isdigit()
    }
    return _parse_top_footprint(out, every_pid)


def _rusage_resident_bytes(pid: int) -> int | None:
    """``ri_resident_size`` from the SAME struct the module reads.

    Read here for one reason: the module hardcodes ``ri_phys_footprint``'s byte
    offset, and an offset that is right today is exactly the kind of constant a
    future macOS can move underneath us. ``ri_resident_size`` is the field
    immediately before it, and it is the one that can be checked against a
    second, independent reader — ``ps`` RSS — byte for byte. If that equality
    ever breaks, every footprint this module reports is suspect, and this row is
    what says so rather than a plausible-looking number.
    """
    import ctypes

    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_pid_rusage.restype = ctypes.c_int
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    buffer = ctypes.create_string_buffer(2048)
    if lib.proc_pid_rusage(pid, 2, ctypes.byref(buffer)) != 0:
        return None
    return int.from_bytes(buffer.raw[64:72], "little")


def _ps_rss_by_pid(pids: list[int]) -> dict[int, int]:
    """The ``ps`` RSS column in bytes, for the layout check's other side."""
    if not pids:
        return {}
    out = subprocess.run(
        ["ps", "-o", "pid=,rss=", "-p", ",".join(str(pid) for pid in pids)],
        capture_output=True,
        timeout=30,
        check=False,
    ).stdout.decode("utf-8", "replace")
    values: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            values[int(parts[0])] = int(parts[1]) * 1024
    return values


def _process_label(comm: str) -> str:
    """A pid's process, in a form that is safe to publish.

    ``ps`` returns either a path or, for a branded runtime, mac's ``Local
    Operator [session] id=<hex>``. The id half names one of the operator's own
    sessions and these artifacts are published on a PR, so it is dropped; the
    path half is reduced to its basename for the same reason (the operator's home
    and worktree layout is not part of the measurement). What is kept is what the
    row is for: whether the pid is a fixture interpreter or a real session
    process.
    """
    head = comm.split(" id=")[0].strip()
    if "/" in head:
        return head.rsplit("/", 1)[-1]
    return head or "?"


def _process_names(pids: list[int]) -> dict[int, str]:
    """A redacted ``comm`` for each pid, for context in the artifact."""
    if not pids:
        return {}
    out = subprocess.run(
        ["ps", "-o", "pid=,comm=", "-p", ",".join(str(pid) for pid in pids)],
        capture_output=True,
        timeout=30,
        check=False,
    ).stdout.decode("utf-8", "replace")
    names: dict[int, str] = {}
    for line in out.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            names[int(parts[0])] = _process_label(parts[1].strip())
    return names


def measure_equivalence(
    fixture: Fixture, tolerance_pct: float, extra_pids: list[int]
) -> dict[str, Any]:
    """``proc_pid_rusage`` vs the ``top`` MEM column, for the same real pids.

    Only meaningful on a tree that has the direct probe; on an older tree this
    reports ``available: false`` and the check is skipped rather than faked.
    """
    from local_operator.mobile import resources as resources_module
    from local_operator.mobile.resources import session_resource_usage

    fixture.refresh()
    if not hasattr(resources_module, "_darwin_footprint_bytes"):
        return {"available": False, "reason": "this tree has no direct probe"}

    pids = fixture.pids + extra_pids
    usage = session_resource_usage(list(pids))
    top_values = _top_mem_by_pid()
    names = _process_names(pids)

    comparisons: list[dict[str, Any]] = []
    for pid in pids:
        direct = usage[pid].footprint_bytes
        topped = top_values.get(pid)
        row: dict[str, Any] = {
            "pid": pid,
            "comm": names.get(pid, ""),
            "direct_bytes": direct,
            "top_bytes": topped,
        }
        if direct is not None and topped:
            row["delta_bytes"] = direct - topped
            row["delta_pct"] = round((direct - topped) / topped * 100, 3)
        comparisons.append(row)

    # The layout proof, on the FIXTURE pids only. ``ri_resident_size`` — the
    # field whose neighbour the module reads — is the same quantity ``ps`` RSS
    # reports, so an independent reader can confirm the struct still sits where
    # the hardcoded offset assumes. It is gated on fixture pids because those are
    # parked processes: for a RUNNING session the two readers sample at different
    # instants, so a large gap there measures the session's own allocation rate
    # rather than the layout (seen on a real session: 165 MB against 247 MB after
    # it allocated for a second).
    fixture_pids = set(fixture.pids)
    rss_values = _ps_rss_by_pid(sorted(fixture_pids))
    resident_values = {pid: _rusage_resident_bytes(pid) for pid in sorted(fixture_pids)}
    checked_pids = [pid for pid in sorted(fixture_pids) if resident_values[pid] is not None]
    resident_deltas: dict[int, int] = {}
    for pid in checked_pids:
        resident = resident_values[pid]
        rss = rss_values.get(pid)
        if resident is not None and rss is not None:
            resident_deltas[pid] = resident - rss
    layout_mismatches = [
        pid for pid, delta in resident_deltas.items() if abs(delta) > _RESIDENT_TOLERANCE_BYTES
    ]

    comparable = [row for row in comparisons if "delta_pct" in row]
    # The CHECK reads the fixture rows only. A fixture process is a sleeping
    # interpreter with a stable footprint, while the extra pids are real running
    # sessions whose footprint genuinely moves between `top`'s sampling instant
    # and the direct read — that is a property of the measurement, not a
    # disagreement between the readers, so it is reported and not gated on.
    fixture_comparable = [row for row in comparable if row["pid"] in fixture_pids]
    worst = max((abs(row["delta_pct"]) for row in fixture_comparable), default=None)
    worst_any = max((abs(row["delta_pct"]) for row in comparable), default=None)
    # Same quantity, different rounding: `top` prints three significant digits,
    # so a 100 MB figure is quantised to ~0.5 MB (~0.5%). The bound is stated in
    # the artifact rather than hidden in the check, and the per-pid rows above
    # are what a reader uses to see a real disagreement.
    return {
        "available": True,
        "rusage_layout_checked_pids": len(checked_pids),
        "rusage_layout_mismatches": layout_mismatches,
        "rusage_layout_max_resident_delta_bytes": max(
            (abs(delta) for delta in resident_deltas.values()), default=None
        ),
        "direct_probe_read_pids": sum(1 for row in comparisons if row["direct_bytes"]),
        "direct_probe_read_fixture_pids": sum(
            1 for row in comparisons if row["pid"] in fixture_pids and row["direct_bytes"]
        ),
        "top_read_pids": sum(1 for row in comparisons if row["top_bytes"]),
        "comparable_pids": len(comparable),
        "fixture_comparable_pids": len(fixture_comparable),
        "worst_abs_delta_pct": worst,
        "worst_abs_delta_pct_any_pid": worst_any,
        "tolerance_pct": tolerance_pct,
        "comparisons": comparisons,
    }


async def _route_request(client: Any, path: str) -> tuple[float, dict[str, Any]]:
    start = time.perf_counter()
    response = await client.get(path)
    elapsed = _ms(time.perf_counter() - start)
    require(response.status_code == 200, f"{path} returned {response.status_code}")
    return elapsed, response.json()["result"]["data"]


def _fixture_rows(payload: dict[str, Any], pids: list[int]) -> dict[str, Any]:
    """What the route's OWN response says about the fixture's sessions.

    A route measurement whose response carries no fixture rows measured the
    warm-up and not the read, so the count travels with the timing instead of
    being assumed from the fixture's existence.
    """
    wanted = set(pids)
    lines = payload.get("sessions", {}).get("lines") or []
    rows = [line for line in lines if line.get("pid") in wanted]
    return {
        "fixture_rows": len(rows),
        "fixture_rows_with_footprint": sum(1 for line in rows if line.get("footprint_bytes")),
        "footprint_by_pid": {
            str(line["pid"]): line.get("footprint_bytes")
            for line in sorted(rows, key=lambda line: line["pid"])
        },
    }


def measure_route_in_process(fixture: Fixture, runs: int) -> dict[str, Any]:
    """`GET /v1/desktop/info` through the real ASGI app, no socket.

    The app is wired the way the daemon wires it (desktop token in the
    environment, config dir pointed at the fixture) and driven through httpx's
    ASGI transport, which runs the middleware, the handler, the JSON encoding
    and the response validation in this process.
    """
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from local_operator.config import ConfigManager
    from local_operator.server.routes import capabilities, desktop_catalogues

    fixture.refresh()
    token = "bench-desktop-token"  # noqa: S105 — a throwaway for a temp config root
    os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"] = token
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(fixture.root)
    app = FastAPI()
    app.include_router(capabilities.router)
    app.include_router(desktop_catalogues.router)
    app.state.config_manager = ConfigManager(fixture.root)

    async def drive() -> tuple[list[float], dict[str, Any]]:
        durations: list[float] = []
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            payload: dict[str, Any] = {}
            for _ in range(runs):
                elapsed, payload = await _route_request(client, "/v1/desktop/info")
                durations.append(elapsed)
        if getattr(app.state, "desktop_auth", None):
            await app.state.desktop_auth.close()
        return durations, payload

    durations, payload = asyncio.run(drive())
    return {
        "route": "/v1/desktop/info",
        "transport": "asgi",
        "first_ms": durations[0],
        "median_ms": _median(durations),
        "runs": durations,
        **_fixture_rows(payload, fixture.pids),
    }


def measure_route_http(
    fixture: Fixture, tree: Path, runs: int, boot_timeout: float
) -> dict[str, Any]:
    """The same route over a real socket, on a daemon started for this run.

    An isolated ``HOME`` AND config dir (both), because the config override alone
    does not redirect the model-cache root — the rule ``AGENTS.md`` states for
    isolating a run. The socket measurement exists because the ASGI transport
    shares this process' already-imported modules, and "the first open after app
    start" is a claim about a process that has just booted.
    """
    import urllib.error
    import urllib.request

    fixture.refresh()
    token = "bench-desktop-token"  # noqa: S105 — throwaway, temp HOME and config root
    home = Path(tempfile.mkdtemp(prefix="lo-info-bench-home-"))
    port = _free_port()
    env = {
        **os.environ,
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(fixture.root),
        "LOCAL_OPERATOR_DESKTOP_TOKEN": token,
        # The daemon must import the tree under test, not the editable install
        # this script was launched from.
        "PYTHONPATH": str(tree),
    }
    # Strip what a parent `lop` session exported: `CMUX_*` renames the
    # operator's real cmux workspaces from a headless child, and `LOP_*` makes a
    # child inherit this session's provider/model (AGENTS.md, "Isolating a run").
    for name in [key for key in env if key.startswith(("CMUX_", "LOP_"))]:
        env.pop(name)

    # Verify which tree the daemon is about to import BEFORE booting it: an
    # editable install of another worktree would otherwise serve this
    # measurement silently, and the row below is what makes that visible.
    probe = subprocess.run(
        [sys.executable, "-c", "import local_operator, sys; print(local_operator.__file__)"],
        cwd=str(tree),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    resolved = (probe.stdout or "").strip().splitlines()[-1] if probe.stdout else ""
    require(
        bool(resolved) and Path(resolved).resolve().is_relative_to(tree),
        f"the daemon would import {resolved or probe.stderr.strip()[:200]}, not {tree}",
    )

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "local_operator.server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(tree),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    def get(path: str) -> tuple[int, float, bytes]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", headers={"Authorization": f"Bearer {token}"}
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
                return response.status, _ms(time.perf_counter() - start), body
        except urllib.error.HTTPError as error:
            return error.code, _ms(time.perf_counter() - start), error.read()

    try:
        boot_start = time.perf_counter()
        deadline = time.time() + boot_timeout
        status = 0
        while time.time() < deadline:
            if server.poll() is not None:
                stderr = (server.stderr.read() if server.stderr else b"").decode("utf-8", "replace")
                raise CheckFailed(
                    f"the daemon exited during boot (rc={server.returncode}): {stderr[-800:]}"
                )
            try:
                status, _, _ = get("/health")
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.25)
        require(status == 200, "the daemon did not answer /health before the boot timeout")
        boot_ms = _ms(time.perf_counter() - boot_start)

        durations: list[float] = []
        payload: dict[str, Any] = {}
        for _ in range(runs):
            code, elapsed, body = get("/v1/desktop/info")
            require(code == 200, f"/v1/desktop/info returned {code}")
            durations.append(elapsed)
            payload = json.loads(body)["result"]["data"]
        return {
            "route": "/v1/desktop/info",
            "transport": "http",
            "server_local_operator": _display_path(resolved),
            "boot_ms_to_health": boot_ms,
            "first_ms": durations[0],
            "median_ms": _median(durations),
            "runs": durations,
            **_fixture_rows(payload, fixture.pids),
        }
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover - teardown safety
            server.kill()
            server.wait(timeout=10)
        shutil.rmtree(home, ignore_errors=True)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _independent_label(blocks: dict[str, Any]) -> str:
    """The independent block names, for the printed line."""
    return "+".join(blocks["independent_blocks"])


def _print_report(report: dict[str, Any]) -> None:
    print(f"tree: {report['tree']} (local_operator from {report['local_operator']})")
    print(f"fixture: {report['fixture']['processes']} processes, {report['fixture']['root']}")
    probes = report["probes"]
    print("\nprobes")
    print(f"  batched ps           {probes['ps_batched_ms']['median']:>9.1f} ms")
    if probes["top_dump_ms"]["available"]:
        print(f"  top -l1 dump         {probes['top_dump_ms']['median']:>9.1f} ms")
    else:
        print("  top -l1 dump         not measured (not macOS)")
    direct = probes["direct_rusage"]
    if direct.get("available"):
        print(
            f"  proc_pid_rusage      {direct['per_pid_us_median']:>9.1f} us/pid "
            f"({direct['pids_read']}/{direct['pids_total']} pids read)"
        )
    else:
        print("  proc_pid_rusage      absent from this tree")
    print(f"  session_resource_usage {probes['session_resource_usage_ms']['median']:>7.1f} ms")
    if "blocks" in report:
        blocks = report["blocks"]
        print("\nblocks, warm (ms)")
        for name, value in sorted(blocks["medians_ms"].items(), key=lambda item: -item[1]):
            print(f"  {name:<20} {value:>9.1f}")
        print(f"  serial sum           {blocks['serial_sum_ms']:>9.1f}")
        print(
            f"  overlap ceiling      {blocks['concurrency_ceiling_ms']:>9.1f} "
            f"(sum {blocks['independent_sum_ms']} - max {blocks['independent_max_ms']} "
            f"over {_independent_label(blocks)})"
        )
    snapshot = report["snapshot"]["collect_snapshot_ms"]
    print("\ncollect_snapshot (real call, real registry fixture)")
    print(f"  first invocation     {snapshot['first']:>9.1f} ms")
    print(f"  median of {len(snapshot['runs'])}           {snapshot['median']:>9.1f} ms")
    if "route_in_process" in report:
        route = report["route_in_process"]
        print("\nGET /v1/desktop/info (in-process ASGI)")
        print(f"  first request        {route['first_ms']:>9.1f} ms")
        print(f"  median of {len(route['runs'])}           {route['median_ms']:>9.1f} ms")
        print(
            f"  fixture rows in the response "
            f"{route['fixture_rows']} ({route['fixture_rows_with_footprint']} with a footprint)"
        )
    if "route_http" in report:
        route = report["route_http"]
        print("\nGET /v1/desktop/info (real daemon over loopback)")
        print(f"  boot to /health      {route['boot_ms_to_health']:>9.1f} ms")
        print(f"  first request        {route['first_ms']:>9.1f} ms")
        print(f"  median of {len(route['runs'])}           {route['median_ms']:>9.1f} ms")
        print(
            f"  fixture rows in the response "
            f"{route['fixture_rows']} ({route['fixture_rows_with_footprint']} with a footprint)"
        )
    equivalence = report["equivalence"]
    if equivalence.get("available"):
        print("\nequivalence (proc_pid_rusage vs top MEM, same pids)")
        print(f"  comparable pids      {equivalence['comparable_pids']}")
        print(
            f"  worst |delta|        {equivalence['worst_abs_delta_pct']}% "
            f"(tolerance {equivalence['tolerance_pct']}%)"
        )
    for line in report["checks"]:
        print(f"  check: {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--tree", default=str(REPO), help="checkout of local-operator to measure")
    parser.add_argument(
        "--processes", type=int, default=12, help="fixture sessions (12 is the reported case)"
    )
    parser.add_argument("--runs", type=int, default=5, help="timed runs per measurement")
    parser.add_argument(
        "--tolerance", type=float, default=1.0, help="allowed |delta| %% between the two readers"
    )
    parser.add_argument(
        "--http", action="store_true", help="also boot a real uvicorn daemon and time the socket"
    )
    parser.add_argument(
        "--boot-timeout", type=float, default=180.0, help="seconds to wait for the daemon"
    )
    parser.add_argument(
        "--no-route", action="store_true", help="skip the in-process route measurement"
    )
    parser.add_argument(
        "--extra-pids",
        default="",
        help=(
            "comma-separated additional pids for the equivalence section — real "
            "long-lived processes beside the fixture's, so the comparison covers "
            "footprints of tens to hundreds of MB rather than parked interpreters"
        ),
    )
    parser.add_argument("--json", help="write the full report to this path")
    args = parser.parse_args()

    tree = Path(args.tree).resolve()
    report: dict[str, Any] = {
        "tree": _display_path(tree),
        "platform": sys.platform,
        # Wall time on a shared box tracks load, so the load this run saw is part
        # of the artifact rather than something a reader has to guess at: this
        # repo is worked from many concurrent worktrees and 1-minute load in the
        # tens is normal here.
        "loadavg": [round(value, 2) for value in os.getloadavg()],
        "load_per_cpu": round(os.getloadavg()[0] / (os.cpu_count() or 1), 2),
        "processes": args.processes,
        "runs": args.runs,
        "tolerance_pct": args.tolerance,
    }
    fixture = Fixture(args.processes)
    try:
        report["local_operator"] = _select_tree(tree)
        report["fixture"] = {"processes": len(fixture.pids), "root": str(fixture.root)}
        report["probes"] = measure_probes(fixture, args.runs)
        report["snapshot"] = measure_snapshot(fixture, args.runs)
        report["blocks"] = measure_blocks(fixture, args.runs)
        extra_pids = [int(part) for part in args.extra_pids.split(",") if part.strip()]
        report["equivalence"] = measure_equivalence(fixture, args.tolerance, extra_pids)
        if not args.no_route:
            report["route_in_process"] = measure_route_in_process(fixture, args.runs)
        if args.http:
            report["route_http"] = measure_route_http(fixture, tree, args.runs, args.boot_timeout)

        # EVERY pid the fixture asked about must come back with a reading: the
        # module's contract is an entry per pid, and a fixture process is one
        # this user owns, so a reader that can see the machine at all can see
        # these. This is the check that fails if the fast path silently returns
        # nothing and the fallback does not cover it.
        quiet = report.get("load_per_cpu", 0.0) <= _QUIET_LOAD_PER_CPU
        # ALWAYS gated, quiet box or not: this is the regression this change
        # removes, and it is two to three orders of magnitude deep (the macOS
        # `top` dump alone measures 2.5-12 s under load against 11-30 ms for the
        # whole probe now), so no amount of machine noise can hide it coming
        # back. The route and snapshot ceilings below have no such margin and are
        # therefore only applied in the quiet regime.
        probe_median = report["probes"]["session_resource_usage_ms"]["median"]
        check(
            probe_median < 1000,
            f"session_resource_usage median {probe_median} ms (under 1000 ms)",
            f"session_resource_usage median took {probe_median} ms: the whole-system "
            "`top` dump is back on the macOS path",
        )
        snapshot_measure = report["snapshot"]
        check(
            snapshot_measure["fixture_rows"] == args.processes,
            f"{args.processes}/{args.processes} fixture sessions were listed",
            f"{snapshot_measure['fixture_rows']} of {args.processes} fixture sessions were listed",
        )
        if quiet:
            check(
                snapshot_measure["collect_snapshot_ms"]["median"] < 1000,
                f"collect_snapshot median {snapshot_measure['collect_snapshot_ms']['median']} ms "
                "(under 1000 ms)",
                f"collect_snapshot median took "
                f"{snapshot_measure['collect_snapshot_ms']['median']} ms (target: under 1000 ms)",
            )
        else:
            check(
                True,
                f"collect_snapshot median "
                f"{snapshot_measure['collect_snapshot_ms']['median']} ms — NOT gated: load was "
                f"{report['load_per_cpu']}x the CPU count",
            )
        check(
            snapshot_measure["fixture_rows_with_footprint"] == args.processes,
            f"{args.processes}/{args.processes} fixture sessions came back with a footprint",
            f"only {snapshot_measure['fixture_rows_with_footprint']} of {args.processes} "
            "fixture sessions got a footprint",
        )

        equivalence = report["equivalence"]
        if equivalence.get("available"):
            mismatches = equivalence["rusage_layout_mismatches"]
            check(
                not mismatches,
                "rusage layout holds: ri_resident_size at offset 64 matches ps RSS on "
                f"{equivalence['rusage_layout_checked_pids']} parked fixture pids "
                f"(worst {equivalence['rusage_layout_max_resident_delta_bytes']} bytes)",
                "ri_resident_size (the field before the one this module reads) no longer "
                f"equals ps RSS on fixture pids {mismatches}: the hardcoded offset is wrong",
            )
            read = equivalence["direct_probe_read_fixture_pids"]
            check(
                read == args.processes,
                f"the direct probe read every fixture pid ({read}/{args.processes})",
                f"the direct probe read only {read} of {args.processes} fixture pids",
            )
            worst = equivalence["worst_abs_delta_pct"]
            check(
                worst is not None and worst <= args.tolerance,
                f"direct footprint agrees with the top MEM column within {worst}% "
                f"(tolerance {args.tolerance}%)",
                f"the two readers disagree by {worst}% on a fixture pid "
                f"(tolerance {args.tolerance}%)",
            )
        else:
            check(True, "direct probe absent from this tree: equivalence not measured")

        for name in ("route_in_process", "route_http"):
            if name in report:
                route = report[name]
                # The route must have READ THE FIXTURE, not merely answered: a
                # response with no fixture rows is a warm-up measurement, and it
                # looks fast for the wrong reason.
                rows = route["fixture_rows"]
                with_footprint = route["fixture_rows_with_footprint"]
                check(
                    rows == args.processes and with_footprint == args.processes,
                    f"{name}: {args.processes} fixture rows with footprints in the response, "
                    f"first request {route['first_ms']} ms",
                    f"{name} answered with {rows} fixture rows ({with_footprint} with a "
                    "footprint): the route did not read the fixture",
                )
                if quiet:
                    check(
                        route["first_ms"] < 1000,
                        f"{name} first request {route['first_ms']} ms (under 1000 ms)",
                        f"{name} first request took {route['first_ms']} ms "
                        "(target: under 1000 ms)",
                    )
                else:
                    check(
                        True,
                        f"{name} first request {route['first_ms']} ms — NOT gated: load was "
                        f"{report['load_per_cpu']}x the CPU count",
                    )
        report["checks"] = list(_CHECKS)
        report["failures"] = list(_FAILURES)
    finally:
        fixture.close()

    report["loadavg_end"] = [round(value, 2) for value in os.getloadavg()]
    if report.get("load_per_cpu", 0) > 2:
        # Not a failure — a warning, and one the artifact carries too: this
        # repo is worked from many concurrent worktrees, and wall time on a
        # loaded box inflates by a multiple. A before/after pair is still
        # comparable (both columns were measured minutes apart on the same
        # machine), but a reader needs to know which regime they are reading.
        print(
            f"\nWARNING: 1-minute load was {report['loadavg'][0]} "
            f"({report['load_per_cpu']}x the {os.cpu_count()} CPUs); wall times below "
            "are inflated by whatever else this machine is running",
            file=sys.stderr,
        )

    _print_report(report)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) failed:", file=sys.stderr)
        for failure in _FAILURES:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckFailed as failure:
        print(f"FAILED: {failure}", file=sys.stderr)
        raise SystemExit(1) from None
