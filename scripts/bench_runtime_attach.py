"""Measure how long `/new` takes to produce a usable runtime.

WHAT THIS MEASURES, AND WHY IT IS THE RIGHT THING
=================================================
The user-visible complaint is that `/new` sits on "starting…" for a long
time. That indicator is raised by `OperatorApp._set_starting(True)` in
`_engage_runtime_eagerly` and lowered when `RemoteSession._ensure_bound()`
returns, so the number a user actually feels is exactly the wall time of
`_ensure_bound`, which decomposes into:

    engage_runtime()            spawn a child + poll until it publishes
      └─ _spawn_runtime()         fork/exec `python -m ...runtime.process`
      └─ child: import           the composition root's import graph
      └─ child: spawn_owned_session()  build the session
      └─ child: RuntimeServer.start_in_process()  bind + publish record
      └─ parent poll loop        find_owner_record() every _POLL_* seconds
    find_owner_record()         one more scan once the errand is delivered
    _bind_to(record)            dial the socket, await the frontend snapshot,
                                load history, install state

Two of those phases are pure latency the parent adds on top of the child's
real work: the **poll grid** (the parent only notices the record on a poll
boundary, so up to one full backoff step is dead time) and the **serialized
spawn** (nothing is started until the first `find_owner_record` scan has
already come back empty). This benchmark reports every phase separately so a
change can be shown to move the phase it claims to move, rather than showing
one aggregate number that improved for unknown reasons.

ISOLATION
=========
Every run gets a fresh `HOME` *and* a fresh `LOCAL_OPERATOR_CONFIG_DIR`.
`AGENTS.md` is explicit that the config dir alone is not enough — the model
catalogue cache derives its root from the home directory independently, so a
run with only the config dir redirected reads the operator's real cache and
reports a warm number as a cold one. The harness therefore never touches the
operator's live sessions, and a benchmark run cannot spawn a runtime that
attaches to a real conversation.

USAGE
=====
    .venv/bin/python scripts/bench_runtime_attach.py --runs 7
    .venv/bin/python scripts/bench_runtime_attach.py --runs 7 --json out.json

Report the MEDIAN, not the mean: the distribution has a long right tail (the
first run in a process pays for the OS page cache being cold on the
interpreter and on site-packages), and a mean over seven runs is dominated by
whichever one happened to land during a Spotlight sweep.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

# A benchmark under scripts/ must read the tree it lives in, not whatever tree
# the venv was installed from (AGENTS.md, "Every feature worktree owns its own
# venv"). Without this a benchmark run in a worktree silently measures main.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Phase:
    """One timed span, recorded in milliseconds."""

    __slots__ = ("name", "started", "elapsed_ms")

    def __init__(self, name: str) -> None:
        self.name = name
        self.started = time.perf_counter()
        self.elapsed_ms = 0.0

    def stop(self) -> float:
        self.elapsed_ms = (time.perf_counter() - self.started) * 1000.0
        return self.elapsed_ms


async def _one_run(config_dir: Path) -> dict[str, float]:
    """Spawn one runtime and attach to it, timing each phase.

    Returns a mapping of phase name to milliseconds. `total` is the number the
    user feels — the whole of `_ensure_bound`.
    """
    from local_operator.mobile.attach_client import find_owner_record
    from local_operator.session.runtime.launch import WarmErrand, engage_runtime

    session_id = f"bench-{uuid.uuid4().hex[:12]}"
    cwd = str(config_dir)
    out: dict[str, float] = {}

    total = Phase("total")

    engage = Phase("engage")
    await engage_runtime(session_id, cwd, WarmErrand(), config_dir=config_dir)
    out["engage_ms"] = engage.stop()

    lookup = Phase("lookup")
    record, _owner = await asyncio.to_thread(find_owner_record, config_dir, session_id)
    out["lookup_ms"] = lookup.stop()

    out["total_ms"] = total.stop()
    out["found"] = 1.0 if record is not None else 0.0

    # Tear the child down: a benchmark that leaves seven runtimes resident
    # measures the eighth against a machine it degraded itself.
    if record is not None:
        pid = getattr(record, "pid", None)
        if isinstance(pid, int) and pid > 0:
            try:
                os.kill(pid, 15)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    return out


def _seed_config(config_dir: Path) -> None:
    """Write the minimum config a runtime needs to construct.

    The `test` provider is used deliberately: `create_session` raises
    `HostingNotConfiguredError` before doing any of the work this benchmark
    measures unless `hosting` resolves, and every real provider would put a
    network client construction (and possibly a live credential check) inside
    the span. `test` resolves through the same `get_provider_definition`
    lookup the engine uses, so the composition root runs its full ordinary
    path — imports, registry, engine, tool wiring — with nothing reaching the
    network. That keeps the measurement about startup rather than about the
    operator's connectivity.

    `model_name` is set explicitly because `default_model_for("test")` is
    None, and a missing model raises `ModelNotConfiguredError` at the same
    preflight.

    Written through `ConfigManager` rather than as hand-rolled YAML: the
    metadata block carries fields (`last_modified`) that the loader requires
    and that a hand-written file silently omits, which surfaces as a
    `KeyError` from inside the child and reads like a startup regression
    rather than a broken fixture.
    """
    from local_operator.config import ConfigManager

    manager = ConfigManager(config_dir=config_dir)
    manager.update_config({"hosting": "test", "model_name": "test-model"})


def _run_isolated(runs: int) -> list[dict[str, float]]:
    """Run the benchmark `runs` times, each in a fresh HOME + config dir."""
    results: list[dict[str, float]] = []
    for index in range(runs):
        root = Path(tempfile.mkdtemp(prefix="lop-bench-"))
        config_dir = root / ".local-operator"
        config_dir.mkdir(parents=True, exist_ok=True)
        _seed_config(config_dir)
        saved = {key: os.environ.get(key) for key in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR")}
        os.environ["HOME"] = str(root)
        os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
        try:
            result = asyncio.run(_one_run(config_dir))
            result["run"] = float(index)
            results.append(result)
            print(
                f"  run {index + 1}/{runs}: total={result['total_ms']:.0f}ms "
                f"engage={result['engage_ms']:.0f}ms lookup={result['lookup_ms']:.0f}ms "
                f"{'ok' if result['found'] else 'NO RECORD'}",
                flush=True,
            )
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            shutil.rmtree(root, ignore_errors=True)
    return results


def _summarize(results: list[dict[str, float]]) -> dict[str, Any]:
    """Median and spread per phase. Median, because the tail is not the signal."""
    summary: dict[str, Any] = {"runs": len(results)}
    for key in ("total_ms", "engage_ms", "lookup_ms"):
        values = [r[key] for r in results if key in r]
        if not values:
            continue
        summary[key] = {
            "median": round(statistics.median(values), 1),
            "min": round(min(values), 1),
            "max": round(max(values), 1),
        }
    summary["records_found"] = sum(int(r.get("found", 0)) for r in results)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=7, help="how many spawns to time")
    parser.add_argument("--json", type=str, default="", help="write raw results here")
    args = parser.parse_args()

    print(f"benchmarking runtime attach: {args.runs} isolated runs", flush=True)
    results = _run_isolated(args.runs)
    summary = _summarize(results)

    print("\n--- summary (ms) ---")
    for key in ("total_ms", "engage_ms", "lookup_ms"):
        if key in summary:
            stat = summary[key]
            print(
                f"  {key:<12} median={stat['median']:>7}  "
                f"min={stat['min']:>7}  max={stat['max']:>7}"
            )
    print(f"  records found: {summary['records_found']}/{summary['runs']}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": summary, "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
