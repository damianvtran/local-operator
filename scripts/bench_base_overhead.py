#!/usr/bin/env python3
"""Base-overhead benchmark: what a session costs BEFORE it does any work.

Startup cost is the one latency the user pays on every single invocation,
including the ones that do nothing (``local-operator --version``, a shell
completion, a scheduler tick). It is also the easiest cost to regress by
accident, because adding one module-level ``import`` in a module the CLI
already touches is invisible in review.

This script measures four things, all in FRESH SUBPROCESSES so nothing is
pre-warmed by the benchmark harness itself:

  1. cold import wall time for ``local_operator.cli`` (the console-script
     entry point) and ``local_operator.session_factory`` (the composition
     root every front end funnels through), reported as min/median over
     ``--runs`` interpreters, net of bare-interpreter startup.
  2. peak RSS after that import, and peak RSS after building a real session
     against the mock provider (``--hosting test --model mock-model``, which
     needs no network and no credentials).
  3. the top ``--top`` heaviest imports by SELF time from ``-X importtime``,
     so a reduction has a concrete target list rather than a vibe.
  4. wall time and peak RSS for a no-op ``exec`` run end to end, through the
     real argv path.

Deterministic output: fixed table layout, numbers only, no adjectives.
``--save FILE`` writes the run as JSON and ``--baseline FILE`` prints a
before/after delta column against a previously saved run, which is how the
"we cut N%" claim is meant to be reproduced.

Run:
    .venv/bin/python scripts/bench_base_overhead.py
    .venv/bin/python scripts/bench_base_overhead.py --save /tmp/before.json
    .venv/bin/python scripts/bench_base_overhead.py --baseline /tmp/before.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Modules whose cold import cost is the headline number. ``cli`` is what the
#: console script pays; ``session_factory`` is what every non-CLI host (server,
#: scheduler, exec worker) pays.
TARGET_MODULES = ("local_operator.cli", "local_operator.session_factory")

#: ru_maxrss is bytes on Darwin and kibibytes on Linux. getrusage does not tell
#: you which, so the platform has to; guessing produces a 1024x error that looks
#: like a real regression.
_RSS_DIVISOR = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0


def _rss_mb(ru_maxrss: int) -> float:
    return ru_maxrss / _RSS_DIVISOR


# Child probe: import one module in a virgin interpreter and report the wall
# time plus peak RSS as JSON. perf_counter brackets ONLY the import, so the
# number excludes interpreter boot; the caller subtracts the boot cost from the
# RSS figure separately using the same probe with an empty module name.
_IMPORT_PROBE = """
import json, resource, sys, time
name = sys.argv[1]
t0 = time.perf_counter()
if name:
    __import__(name)
elapsed = time.perf_counter() - t0
print(json.dumps({
    "seconds": elapsed,
    "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "modules": len(sys.modules),
}))
"""

# Child probe: build a real session against the mock provider and report peak
# RSS. This is the "loaded and wired, but zero turns run" state — the floor a
# long-lived session sits at. Everything is torn down before reporting so a
# leaked SQLite handle shows up as a crash here rather than as flakiness in the
# test suite later.
_SESSION_PROBE = """
import argparse, asyncio, json, os, resource, sys, time
config_dir = os.environ["LO_BENCH_CONFIG_DIR"]
from pathlib import Path

t0 = time.perf_counter()
from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.session_factory import create_session

args = argparse.Namespace(
    hosting="test", model="mock-model", agent_name=None, agent_id=None,
    yolo=True, train=False,
)


async def build():
    session = await create_session(
        args,
        ConfigManager(Path(config_dir)),
        CredentialManager(Path(config_dir)),
        AgentRegistry(Path(config_dir)),
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    await session.dispose()
    return rss


rss = asyncio.run(build())
print(json.dumps({
    "seconds": time.perf_counter() - t0,
    "ru_maxrss": rss,
    "modules": len(sys.modules),
}))
"""

# Child probe: boot the REAL TUI headlessly and report its peak RSS. The
# interactive surface is what a human actually runs, so measuring only `exec`
# would report the cheap half of the product. Textual's own import graph is the
# bulk of the difference, which is the point: the gap between this cell and the
# `exec` cell below is the evidence that Textual stays LAZY and headless runs
# never pay for it. A regression that moves `import textual` to module scope in
# `cli.py` shows up here as the two cells converging.
#
# `run_test` drives the app through Textual's own harness rather than a pty, so
# there is no terminal to own and nothing to restore on the way out.
_TUI_PROBE = """
import argparse, asyncio, json, os, resource, sys, time
from pathlib import Path
config_dir = os.environ["LO_BENCH_CONFIG_DIR"]
os.environ["LOCAL_OPERATOR_NO_SHIMMER"] = "1"

t0 = time.perf_counter()
from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.session_factory import create_session
from local_operator.tui.app import OperatorApp

args = argparse.Namespace(
    hosting="test", model="mock-model", agent_name=None, agent_id=None,
    yolo=True, train=False, cwd=None,
)


async def boot():
    async def factory():
        return await create_session(
            args,
            ConfigManager(Path(config_dir)),
            CredentialManager(Path(config_dir)),
            AgentRegistry(Path(config_dir)),
            has_ui=True,
        )

    app = OperatorApp(factory, "dark", None)
    async with app.run_test(size=(100, 30)) as pilot:
        # Type without submitting: the goal is a fully mounted, fully styled
        # frame with a live session behind it, not a provider round trip.
        await pilot.pause()
        await pilot.press(*"hello")
        await pilot.pause()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


rss = asyncio.run(boot())
print(json.dumps({
    "seconds": time.perf_counter() - t0,
    "ru_maxrss": rss,
    "modules": len(sys.modules),
}))
"""

# Child probe: run the console entry point end to end as its OWN child, then
# report that child's peak RSS. RUSAGE_CHILDREN is a high-water mark across
# every child a process has reaped, so it is only attributable when the
# reporting process is freshly spawned and reaps exactly one child — hence the
# extra process layer instead of calling getrusage from the benchmark itself.
_EXEC_PROBE = """
import json, resource, subprocess, sys, time
argv = sys.argv[1:]
t0 = time.perf_counter()
proc = subprocess.run(argv, capture_output=True, text=True)
elapsed = time.perf_counter() - t0
print(json.dumps({
    "seconds": elapsed,
    "ru_maxrss": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    "returncode": proc.returncode,
    "stdout": proc.stdout[-400:],
    "stderr": proc.stderr[-400:],
}))
"""


def _child_env(config_dir: Path) -> dict[str, str]:
    """Environment for every probe: an isolated config dir so the benchmark
    never reads or writes the developer's real ``~/.local-operator``.

    Bytecode caching is deliberately left ON. Disabling it would make every
    sample pay source compilation for the whole dependency graph, which is not
    what a real invocation pays; instead ``_repeat`` discards a warmup run so
    ``__pycache__`` is populated before the first measured sample and every
    sample is warm — the state a real invocation is in after the first. Drop
    that warmup discard and run #1 becomes cold while the rest are warm, which
    skews min and median in opposite directions."""
    env = dict(os.environ)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    env["LO_BENCH_CONFIG_DIR"] = str(config_dir)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_probe(code: str, argv: list[str], env: dict[str, str]) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", code, *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"probe failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    # The wrapper interpreter's returncode only proves the PROBE ran. A probe
    # that spawns the thing under test (_EXEC_PROBE) exits 0 even when that
    # child crashed, and a crash is FAST — a failing `exec noop` would be
    # reported as a 45% improvement instead of a regression. Probes that shell
    # out report the measured command's own returncode; honour it.
    measured_rc = payload.get("returncode", 0)
    if measured_rc:
        raise RuntimeError(
            f"measured command failed ({measured_rc}): {argv}\n"
            f"--- stderr ---\n{payload.get('stderr', '')}\n"
            f"--- stdout ---\n{payload.get('stdout', '')}"
        )
    return payload


def _repeat(code: str, argv: list[str], env: dict[str, str], runs: int) -> dict:
    """Run a probe ``runs`` times and reduce to min/median.

    The first run is discarded. A freshly checked-out tree has no
    ``__pycache__`` for the modules under test, so run #1 pays source
    compilation for the entire dependency graph — tens of milliseconds no real
    invocation pays after the first, and enough to swamp the difference this
    benchmark exists to detect.

    Min is the signal (least interference from the OS), median is the guard
    against a single lucky run; reporting only a mean would let one scheduler
    hiccup masquerade as a regression.
    """
    _run_probe(code, argv, env)  # warmup: populates __pycache__, result discarded
    samples = [_run_probe(code, argv, env) for _ in range(runs)]
    secs = sorted(s["seconds"] for s in samples)
    rss = sorted(s["ru_maxrss"] for s in samples)
    return {
        "min_ms": secs[0] * 1000.0,
        "median_ms": statistics.median(secs) * 1000.0,
        "min_rss_mb": _rss_mb(rss[0]),
        "median_rss_mb": _rss_mb(statistics.median(rss)),
        "modules": samples[-1].get("modules", 0),
        "runs": runs,
    }


def _importtime_self(
    module: str, env: dict[str, str], top: int, runs: int
) -> list[tuple[str, int]]:
    """Parse ``-X importtime`` and return the ``top`` costliest modules by SELF
    microseconds, taking the MINIMUM across ``runs`` interpreters.

    Self time (not cumulative) is the right ranking for optimisation: a parent
    with a huge cumulative number is usually just the module that happened to
    import the expensive leaf first, and moving the parent buys nothing.

    Min-of-N rather than a single run because ``-X importtime`` attributes the
    whole wall clock of an import to that import, including time the process
    spent descheduled. On a loaded machine one unlucky sample promotes an
    innocent 200us module to the top of the list and sends the optimisation
    work at the wrong target; the floor across runs is the module's real cost.
    """
    best: dict[str, int] = {}
    for _ in range(runs):
        proc = subprocess.run(
            [sys.executable, "-X", "importtime", "-c", f"import {module}"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO),
        )
        for line in proc.stderr.splitlines():
            if not line.startswith("import time:"):
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            raw_self = parts[0].split(":", 1)[1].strip()
            if not raw_self.isdigit():  # the header row
                continue
            name, usec = parts[2].strip(), int(raw_self)
            if usec < best.get(name, 1 << 62):
                best[name] = usec
    rows = sorted(best.items(), key=lambda r: r[1], reverse=True)
    return rows[:top]


def measure(runs: int, top: int, config_dir: Path) -> dict:
    env = _child_env(config_dir)

    # Bare interpreter boot: subtracted from RSS so the reported megabytes are
    # the harness's own, not Python's. Import time is already boot-free
    # (perf_counter starts after boot), so only RSS needs the correction.
    baseline = _repeat(_IMPORT_PROBE, [""], env, runs)

    result: dict = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "runs": runs,
        "baseline_interpreter": baseline,
        "imports": {},
        "importtime_top": {},
    }
    for module in TARGET_MODULES:
        result["imports"][module] = _repeat(_IMPORT_PROBE, [module], env, runs)
        result["importtime_top"][module] = _importtime_self(module, env, top, runs)

    result["session_build"] = _repeat(_SESSION_PROBE, [], env, max(1, runs // 2) or 1)
    result["tui_boot"] = _repeat(_TUI_PROBE, [], env, max(1, runs // 2) or 1)

    exec_argv = [
        sys.executable,
        "-m",
        "local_operator.cli",
        "--hosting",
        "test",
        "--model",
        "mock-model",
        "exec",
        "noop",
    ]
    result["exec_noop"] = _repeat(_EXEC_PROBE, exec_argv, env, max(1, runs // 2) or 1)
    return result


def _delta(now: float, before: float | None) -> str:
    if before is None or before == 0:
        return ""
    diff = now - before
    pct = diff / before * 100.0
    return f"{diff:+8.1f} ({pct:+6.1f}%)"


def _dig(blob: dict | None, *keys):
    cur: object = blob
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _rows(result: dict, baseline: dict | None) -> list[tuple]:
    """Flatten a run into ``(label, unit, min, median, baseline_min)`` rows.

    Both the current run and the baseline are projected through the SAME
    function, so a table cell and its delta can never be computed from
    different expressions — the bug that makes an "improvement" column lie.
    """
    base_rss = result["baseline_interpreter"]["min_rss_mb"]
    old_base_rss = _dig(baseline, "baseline_interpreter", "min_rss_mb")

    def rss_pair(cell: dict, old: dict | None) -> tuple[float, float, float | None]:
        # RSS is reported NET of the bare interpreter, and the baseline uses
        # its own floor: comparing a raw ru_maxrss across machines or Python
        # patch releases measures CPython, not this harness.
        old_min = _dig(old, "min_rss_mb")
        before = None
        if old_min is not None and old_base_rss is not None:
            before = old_min - old_base_rss
        return cell["min_rss_mb"] - base_rss, cell["median_rss_mb"] - base_rss, before

    rows: list[tuple] = []
    for module in TARGET_MODULES:
        cell = result["imports"][module]
        old = _dig(baseline, "imports", module)
        short = module.replace("local_operator.", "lo.")
        rows.append(
            ("cold import " + short, "ms", cell["min_ms"], cell["median_ms"], _dig(old, "min_ms"))
        )
        lo_rss, med_rss, before = rss_pair(cell, old)
        rows.append((f"  RSS after import {short}", "MB", lo_rss, med_rss, before))
        mods = float(cell["modules"])
        rows.append((f"  sys.modules after {short}", "", mods, mods, _dig(old, "modules")))

    sess, old_sess = result["session_build"], _dig(baseline, "session_build")
    rows.append(
        ("session build (mock)", "ms", sess["min_ms"], sess["median_ms"], _dig(old_sess, "min_ms"))
    )
    lo_rss, med_rss, before = rss_pair(sess, old_sess)
    rows.append(("  RSS after session build", "MB", lo_rss, med_rss, before))

    tui, old_tui = result["tui_boot"], _dig(baseline, "tui_boot")
    rows.append(
        (
            "TUI boot (mock, headless)",
            "ms",
            tui["min_ms"],
            tui["median_ms"],
            _dig(old_tui, "min_ms"),
        )
    )
    lo_rss, med_rss, before = rss_pair(tui, old_tui)
    rows.append(("  RSS after TUI boot", "MB", lo_rss, med_rss, before))
    tui_mods = float(tui["modules"])
    rows.append(("  sys.modules after TUI boot", "", tui_mods, tui_mods, _dig(old_tui, "modules")))

    ex, old_ex = result["exec_noop"], _dig(baseline, "exec_noop")
    rows.append(
        ("exec noop end-to-end", "ms", ex["min_ms"], ex["median_ms"], _dig(old_ex, "min_ms"))
    )
    # The exec child's RSS is NOT floor-corrected: it is the whole process the
    # user's machine actually pays for, which is the number that matters here.
    rows.append(
        (
            "  peak RSS of exec child",
            "MB",
            ex["min_rss_mb"],
            ex["median_rss_mb"],
            _dig(old_ex, "min_rss_mb"),
        )
    )
    return rows


def report(result: dict, baseline: dict | None, top: int) -> None:
    base_rss = result["baseline_interpreter"]["min_rss_mb"]
    print(f"python {result['python']} on {result['platform']}, {result['runs']} runs per cell")
    print(f"bare interpreter RSS floor: {base_rss:.1f} MB (subtracted from harness RSS below)")
    print()

    label_w = 36
    header = f"{'measurement':<{label_w}}{'min':>10}{'median':>10}  {'unit':<4}"
    if baseline:
        header += f"{'delta vs baseline':>24}"
    print(header)
    print("-" * len(header))
    for label, unit, lo, med, before in _rows(result, baseline):
        line = f"{label:<{label_w}}{lo:>10.1f}{med:>10.1f}  {unit:<4}"
        if baseline:
            line += f"{_delta(lo, before):>24}"
        print(line)

    for module in TARGET_MODULES:
        print()
        print(f"top {top} imports by SELF time — {module}")
        print(f"{'module':<52}{'self us':>10}")
        print("-" * 62)
        for name, usec in result["importtime_top"][module]:
            print(f"{name:<52}{usec:>10}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=7, help="interpreters per import measurement")
    parser.add_argument("--top", type=int, default=15, help="how many self-time offenders to list")
    parser.add_argument("--save", type=Path, default=None, help="write this run as JSON")
    parser.add_argument("--baseline", type=Path, default=None, help="compare against a saved run")
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text()) if args.baseline else None

    with tempfile.TemporaryDirectory(prefix="lo-bench-overhead-") as tmp:
        config_dir = Path(tmp) / ".local-operator"
        config_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        result = measure(args.runs, args.top, config_dir)
        result["wall_seconds"] = time.time() - started

    report(result, baseline, args.top)
    if args.save:
        args.save.write_text(json.dumps(result, indent=2))
        print(f"\nsaved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
