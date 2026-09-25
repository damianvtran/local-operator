#!/usr/bin/env python3
"""One harness for the four latencies an operator actually feels.

``scripts/`` already carries ~46 single-purpose benchmarks. Each one is right
about its own axis and none of them answers the question this file exists for:
*did this branch make the product feel faster, on the paths a person waits on,
without trading one of them for another?* The four paths are the four complaints
they came from:

1. **startup** — process launch to a useful answer, for the cheap verbs and for
   a real session build;
2. **prompt** — a prompt submitted to the model request being issued;
3. **switch** — picking another session to a usable transcript;
4. **fan-out** — the same work while N sessions are live, which is where a
   harness stops being fast in isolation and starts being slow in aggregate.

WHY THIS IS NOT A FIFTH MICRO-BENCHMARK. Every cell here reports two numbers,
and they are not interchangeable:

* **CPU ms** — the primary metric. This host runs ~25 concurrent agent sessions
  and sits at load 100-190; a wall-clock reading at that load is weather. CPU is
  what the change actually removed, and it is comparable across runs.
* **wall ms** — reported *with the load average that produced it*, and only ever
  as an interleaved A/B inside one process, because that is the only shape in
  which two wall numbers on this host mean anything.

A cell that cannot be measured honestly is reported as ``skipped`` with the
reason, never as a zero. ``--baseline`` diffs against a saved run; a cell whose
shape changed between the two runs is marked ``INCOMPARABLE`` rather than
deltas, so a schema change cannot silently read as a speedup.

Usage::

    .venv/bin/python scripts/bench_harness_perf.py --save /tmp/before.json
    .venv/bin/python scripts/bench_harness_perf.py --baseline /tmp/before.json

Every subprocess runs under an isolated ``HOME`` and config root of its own, and
never touches the operator's live store. ``--runs`` controls the sample count
per cell (default 5, minimum 3).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent

#: The cells, in the order the report prints them. Each is a callable that
#: returns ``(cpu_ms, wall_ms, detail)`` or raises ``_Skip`` with its reason.
_SKIP = "skipped"


class Skipped(Exception):
    """A cell that cannot be measured on this host, with the reason why."""


def _load() -> str:
    """The 1-minute load average, as the string every wall reading carries."""
    try:
        return f"{os.getloadavg()[0]:.1f}"
    except OSError:  # pragma: no cover - platform without loadavg
        return "n/a"


def _isolated_env(root: Path) -> dict[str, str]:
    """A child environment with no inherited lop/CMUX state.

    ``env -i``-equivalent in-process: the two prefixes below are read by the
    CHILD PRODUCT, not merely by a terminal, and an inherited ``LOP_*`` silently
    changes what a child runtime is (see AGENTS.md "Isolating a run").
    """
    keep = ("PATH", "TERM", "LANG", "LC_ALL", "TMPDIR", "SHELL", "USER")
    env = {name: os.environ[name] for name in keep if name in os.environ}
    env["HOME"] = str(root)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(root / ".local-operator")
    env["TERM"] = env.get("TERM", "xterm-256color")
    env["PYTHONPATH"] = str(REPO)
    return env


def _run_probe(
    code: str,
    argv: list[str],
    env: dict[str, str],
    cwd: Path,
    timeout: float = 600.0,
) -> tuple[float, float, dict[str, Any]]:
    """Run ``code`` in a fresh interpreter; return (child CPU ms, wall ms, payload).

    Child CPU comes from ``RUSAGE_CHILDREN`` deltas rather than the child's own
    ``process_time``, so the fork/exec and the interpreter's own startup are
    inside the number: that is the point, because it is what a user waits for.
    """
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, *argv, "-c", code],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall_ms = (time.perf_counter() - start) * 1000
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_ms = ((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)) * 1000
    if proc.returncode != 0:
        raise Skipped(f"probe exited {proc.returncode}: {proc.stderr.strip()[-400:]}")
    payload: dict[str, Any] = {}
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
    return cpu_ms, wall_ms, payload


# --------------------------------------------------------------------------
# Cell 1: startup. The cheap verbs first, because they are paid on EVERY
# invocation (a shell completion, a scheduler tick, an agent's status poll),
# then a real session build.
# --------------------------------------------------------------------------

_IMPORT_PROBE = """
import json, os, sys, time
t0 = time.process_time()
import local_operator.cli  # noqa: F401
print(json.dumps({"cpu_ms": (time.process_time() - t0) * 1000, "modules": len(sys.modules)}))
"""

#: ``cli.main()`` reads ``sys.argv`` itself (it takes no argument), so the probe
#: sets argv rather than passing one — driving the same entry the console script
#: drives is the whole point of this cell.
_VERSION_PROBE = """
import json, sys, time
sys.argv = ["local-operator", "--version"]
from local_operator.cli import main
t0 = time.process_time()
try:
    main()
except SystemExit:
    pass
print(json.dumps({"cpu_ms": (time.process_time() - t0) * 1000}))
"""

_SESSION_PROBE = """
import argparse, asyncio, json, time
from pathlib import Path
config_dir = __import__("os").environ["LOCAL_OPERATOR_CONFIG_DIR"]
from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.session_factory import create_session

args = argparse.Namespace(
    hosting="test", model="mock-model", agent_name=None, agent_id=None,
    yolo=True, train=False,
)


async def build():
    t0 = time.process_time()
    session = await create_session(
        args, ConfigManager(Path(config_dir)), AgentRegistry(Path(config_dir))
    )
    cpu = (time.process_time() - t0) * 1000
    await session.dispose()
    return cpu


print(json.dumps({"cpu_ms": asyncio.run(build())}))
"""


def cell_startup(runs: int, root: Path, work: Path) -> dict[str, Any]:
    env = _isolated_env(root)
    probes: dict[str, tuple[str, list[str]]] = {
        # cwd is a small private directory on purpose: `sys.path[0]` is the CWD
        # for `-c`, and importlib.metadata walks every sys.path entry, so a big
        # cwd would measure the parent directory rather than this branch.
        "import local_operator.cli": (_IMPORT_PROBE, []),
        "lop --version": (_VERSION_PROBE, []),
        "session build (mock provider)": (_SESSION_PROBE, []),
    }
    cells: dict[str, Any] = {}
    for name, (code, argv) in probes.items():
        samples: list[tuple[float, float]] = []
        detail: dict[str, Any] = {}
        for _ in range(runs):
            try:
                cpu, wall, payload = _run_probe(code, argv, env, work)
            except (Skipped, subprocess.TimeoutExpired) as exc:
                cells[name] = {"status": _SKIP, "reason": str(exc)[:300]}
                break
            samples.append((cpu, wall))
            detail.update(payload)
        else:
            cpus = sorted(cpu for cpu, _ in samples)
            walls = sorted(wall for _, wall in samples)
            cells[name] = {
                "status": "ok",
                "cpu_ms_min": round(cpus[0], 2),
                "cpu_ms_median": round(statistics.median(cpus), 2),
                "wall_ms_min": round(walls[0], 1),
                "wall_ms_median": round(statistics.median(walls), 1),
                "runs": len(samples),
                **{k: v for k, v in detail.items() if k != "cpu_ms"},
            }
    return {"cells": cells, "load": _load()}


# --------------------------------------------------------------------------
# Cell 2: prompt. One prompt through the real exec path against the mock
# provider. The interesting quantity is not the model turn (there is none) but
# how much CPU a fresh process spends before the request is issued.
# --------------------------------------------------------------------------

_EXEC_PROBE = """
import json, sys, time
sys.argv = ["local-operator", "exec", "--hosting", "test", "--model", "mock",
            "--yolo", "bench prompt"]
from local_operator.cli import main
t0 = time.process_time()
try:
    main()
except SystemExit:
    pass
print(json.dumps({"cpu_ms": (time.process_time() - t0) * 1000}))
"""


def cell_prompt(runs: int, root: Path, work: Path) -> dict[str, Any]:
    env = _isolated_env(root)
    samples: list[tuple[float, float]] = []
    for _ in range(runs):
        try:
            cpu, wall, _payload = _run_probe(_EXEC_PROBE, [], env, work)
        except (Skipped, subprocess.TimeoutExpired) as exc:
            return {
                "cells": {"exec cold prompt": {"status": _SKIP, "reason": str(exc)[:300]}},
                "load": _load(),
            }
        samples.append((cpu, wall))
    cpus = sorted(cpu for cpu, _ in samples)
    walls = sorted(wall for _, wall in samples)
    return {
        "cells": {
            "exec cold prompt": {
                "status": "ok",
                "cpu_ms_min": round(cpus[0], 2),
                "cpu_ms_median": round(statistics.median(cpus), 2),
                "wall_ms_min": round(walls[0], 1),
                "wall_ms_median": round(statistics.median(walls), 1),
                "runs": len(samples),
            }
        },
        "load": _load(),
    }


# --------------------------------------------------------------------------
# Cell 3: switch. Delegated to the tree's own switch harness, which already
# measures the structural counts (displays/layouts/mounts) that make a switch
# comparable across load. Re-implementing it here would be a second opinion
# about the same code, so this cell runs it and lifts its headline numbers.
# --------------------------------------------------------------------------

_SWITCH = REPO / "scripts" / "bench_session_switch.py"


def cell_switch(runs: int, root: Path, work: Path) -> dict[str, Any]:
    if not _SWITCH.exists():
        return {
            "cells": {
                "session switch": {"status": _SKIP, "reason": "bench_session_switch.py absent"}
            },
            "load": _load(),
        }
    out = work / "switch.json"
    env = _isolated_env(root)
    proc = subprocess.run(
        [sys.executable, str(_SWITCH), "--output", str(out), "--samples", str(max(1, runs // 2))],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not out.exists():
        return {
            "cells": {
                "session switch": {
                    "status": _SKIP,
                    "reason": f"harness rc={proc.returncode}: {proc.stderr.strip()[-300:]}",
                }
            },
            "load": _load(),
        }
    data = json.loads(out.read_text())
    records: list[dict[str, Any]] = []
    raw = data.get("records", {})
    for group in raw.values():
        for record in group if isinstance(group, list) else [group]:
            if isinstance(record, dict) and record.get("ok"):
                records.append(record)
    if not records:
        return {
            "cells": {"session switch": {"status": _SKIP, "reason": "no ok records"}},
            "load": _load(),
        }
    loop_cpu = sorted(float(r.get("loop_cpu_ms") or 0.0) for r in records)
    displays = sorted(int(r.get("displays") or 0) for r in records)
    layouts = sorted(int(r.get("layouts") or 0) for r in records)
    return {
        "cells": {
            "session switch (cold, small transcript)": {
                "status": "ok",
                "loop_cpu_ms_median": round(statistics.median(loop_cpu), 2),
                "loop_cpu_ms_min": round(loop_cpu[0], 2),
                # Structural counts, not times: identical on an idle laptop and
                # a wedged runner, which is why they are quoted next to the CPU.
                "displays_median": displays[len(displays) // 2],
                "layouts_median": layouts[len(layouts) // 2],
                "records": len(records),
                "source": "scripts/bench_session_switch.py",
            }
        },
        "load": _load(),
    }


# --------------------------------------------------------------------------
# Cell 4: fan-out. N live sessions in one process, which is where per-session
# costs that are invisible at N=1 show up as a slope. Reported as CPU per
# session so the SHAPE is the result, not any single time.
# --------------------------------------------------------------------------

_FANOUT_PROBE = """
import argparse, asyncio, json, os, time
from pathlib import Path
config_dir = os.environ["LOCAL_OPERATOR_CONFIG_DIR"]
from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.session_factory import create_session

count = int(os.environ["LO_FANOUT_N"])
args = argparse.Namespace(
    hosting="test", model="mock-model", agent_name=None, agent_id=None,
    yolo=True, train=False,
)


async def build_all():
    sessions = []
    t0 = time.process_time()
    for _ in range(count):
        sessions.append(
            await create_session(
                args, ConfigManager(Path(config_dir)), AgentRegistry(Path(config_dir))
            )
        )
    cpu = (time.process_time() - t0) * 1000
    for session in sessions:
        await session.dispose()
    return cpu


print(json.dumps({"cpu_ms": asyncio.run(build_all()), "n": count}))
"""


def cell_fanout(runs: int, root: Path, work: Path) -> dict[str, Any]:
    env = _isolated_env(root)
    cells: dict[str, Any] = {}
    for n in (1, 5):
        samples: list[float] = []
        for _ in range(max(2, runs // 2)):
            env["LO_FANOUT_N"] = str(n)
            try:
                cpu, _wall, _payload = _run_probe(_FANOUT_PROBE, [], env, work)
            except (Skipped, subprocess.TimeoutExpired) as exc:
                cells[f"{n} sessions in one process"] = {
                    "status": _SKIP,
                    "reason": str(exc)[:300],
                }
                samples = []
                break
            samples.append(cpu)
        if samples:
            best = min(samples)
            cells[f"{n} sessions in one process"] = {
                "status": "ok",
                "cpu_ms_min": round(best, 2),
                "cpu_ms_per_session_min": round(best / n, 2),
                "runs": len(samples),
            }
    return {"cells": cells, "load": _load()}


CELLS: dict[str, Callable[[int, Path, Path], dict[str, Any]]] = {
    "startup": cell_startup,
    "prompt": cell_prompt,
    "switch": cell_switch,
    "fanout": cell_fanout,
}


def _flatten(section: str, data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f"{section}/{name}": cell for name, cell in data.get("cells", {}).items()}


def _print_run(result: dict[str, Any]) -> None:
    print(f"host load (1m): {result['load']}   python {sys.version.split()[0]}")
    print(f"repo: {REPO}")
    for section in ("startup", "prompt", "switch", "fanout"):
        data = result["sections"].get(section)
        if not data:
            continue
        print(f"\n== {section} ==  (load {data.get('load', '?')})")
        for name, cell in data.get("cells", {}).items():
            if cell.get("status") != "ok":
                print(f"  {name:44s} {_SKIP}: {cell.get('reason', '')[:80]}")
                continue
            cpu = cell.get("cpu_ms_min")
            percpu = cell.get("cpu_ms_per_session_min")
            wall = cell.get("wall_ms_min")
            parts = []
            if cpu is not None:
                parts.append(f"cpu {cpu:8.1f} ms")
            if percpu is not None:
                parts.append(f"({percpu:.1f}/session)")
            if wall is not None:
                parts.append(f"wall {wall:8.1f} ms")
            if "loop_cpu_ms_median" in cell:
                parts.append(f"loop-cpu {cell['loop_cpu_ms_median']:.1f} ms")
                parts.append(f"displays {cell.get('displays_median')}")
            print(f"  {name:44s} " + "  ".join(parts))


def _print_delta(result: dict[str, Any], baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text())
    before = {}
    for section in ("startup", "prompt", "switch", "fanout"):
        before.update(_flatten(section, baseline["sections"].get(section, {})))
    after = {}
    for section in ("startup", "prompt", "switch", "fanout"):
        after.update(_flatten(section, result["sections"].get(section, {})))
    print(f"\n== delta vs {baseline_path} ==")
    print(f"  baseline load {baseline.get('load')}   this run load {result.get('load')}")
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        if old is None or new is None:
            print(f"  {name:44s} only in one run -> NOT COMPARED")
            continue
        if old.get("status") != "ok" or new.get("status") != "ok":
            print(f"  {name:44s} skipped on one side -> NOT COMPARED")
            continue
        key = None
        for candidate in ("cpu_ms_min", "loop_cpu_ms_median"):
            if candidate in old and candidate in new:
                key = candidate
                break
        if key is None:
            print(f"  {name:44s} INCOMPARABLE (no shared metric)")
            continue
        base, now = float(old[key]), float(new[key])
        if base <= 0:
            print(f"  {name:44s} INCOMPARABLE (baseline 0)")
            continue
        pct = (now - base) / base * 100
        verdict = "faster" if pct < -1 else ("slower" if pct > 1 else "unchanged")
        print(f"  {name:44s} {base:9.2f} -> {now:9.2f} ms  {pct:+6.1f}%  {verdict}  [{key}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=5, help="samples per cell (minimum 3)")
    parser.add_argument(
        "--only", default="", help="comma-separated sections: startup,prompt,switch,fanout"
    )
    parser.add_argument("--save", type=Path, default=None, help="write this run as JSON")
    parser.add_argument("--baseline", type=Path, default=None, help="compare against a saved run")
    args = parser.parse_args(argv)

    runs = max(3, args.runs)
    wanted = [s.strip() for s in args.only.split(",") if s.strip()] or list(CELLS)
    unknown = [s for s in wanted if s not in CELLS]
    if unknown:
        parser.error(f"unknown section(s): {', '.join(unknown)}")

    result: dict[str, Any] = {
        "harness": "bench_harness_perf",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "repo": str(REPO),
        "runs": runs,
        "load": _load(),
        "sections": {},
    }
    root = Path(tempfile.mkdtemp(prefix="lop-bench-"))
    try:
        work = root / "cwd"
        work.mkdir(parents=True, exist_ok=True)
        for section in wanted:
            result["sections"][section] = CELLS[section](runs, root, work)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    _print_run(result)
    if args.baseline is not None:
        if args.baseline.exists():
            _print_delta(result, args.baseline)
        else:
            print(f"\nbaseline {args.baseline} does not exist; nothing to compare")
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"\nsaved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
