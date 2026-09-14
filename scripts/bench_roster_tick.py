"""What one roster tick costs the session loop, and what a sync frame costs.

Runs the REAL code path -- no monkeypatched store -- and prints, per tick, the
wall cost, the GC collections, and how many retained rows were frozen. Both
halves of the A/B run this one file, so the comparison is the same inputs on the
same host through the same interpreter:

  # one tree to measure at a time, same file, same interpreter:
  cd /tmp && PYTHONPATH=~/local-operator-worktrees/<tree> \
    ~/local-operator-worktrees/<any-tree-with-a-venv>/.venv/bin/python \
    ~/local-operator-worktrees/<tree-with-this-script>/scripts/bench_roster_tick.py all

Run it from OUTSIDE any tree, and do NOT add the scripts/ self-correction some
siblings carry: ``sys.path[0]`` is the script's directory, so ``PYTHONPATH`` is
what decides which tree's ``local_operator`` is imported, and selecting the tree
under test is the whole point of an A/B. The first line of output names the tree
that actually ran -- check it, because a wrong tree silently produces baseline
numbers for both halves.

The row shape and the workloads are the architect's (arch-resume-wedge, §1/B1-B2):
5 subagents x 500 retained rows x 8 KiB, and a 22-job roster for the sync frame
(the operator's wedged session `835fbcafdc27` carries 22 rows). Rows carry the
REAL append stamp (``harness.jobs.TRAJECTORY_SEQ_KEY``), because that stamp is
the identity the retained-window cache matches on and a fixture without it would
measure the fallback path instead of the shipped one.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Any

os.environ.setdefault("LOCAL_OPERATOR_CONFIG_DIR", "/tmp/frame-cost-cfg")

from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY as SEQ  # noqa: E402
from local_operator.session import frontend_state as fs  # noqa: E402
from local_operator.session.frontend_state import (  # noqa: E402
    FrontendSessionState,
    FrontendStateStore,
    FrontendSync,
    JobState,
    sync_wire_payload,
)

JOBS = 5
ROWS = 500
ROW_TEXT_BYTES = 8192
TICKS = 20

#: Rows frozen since the last reset. ``_freeze_value`` is the single funnel every
#: retained row goes through on the way into canonical state, so counting the
#: ROW-shaped calls to it counts the work the cache is supposed to remove. The
#: spy wraps the module function the same way in both halves of the A/B.
_frozen_rows = 0
_original_freeze_value = fs._freeze_value


def _counting_freeze_value(value: Any) -> Any:
    global _frozen_rows
    if isinstance(value, dict) and SEQ in value:
        _frozen_rows += 1
    return _original_freeze_value(value)


def install_freeze_spy() -> None:
    fs._freeze_value = _counting_freeze_value


def make_row(i: int, text_bytes: int = ROW_TEXT_BYTES) -> dict[str, Any]:
    return {
        "type": "tool_execution_end",
        SEQ: i,
        "tool_call_id": f"call_{i:08d}",
        "tool_name": "bash",
        "intent": "run a command and read what it printed",
        "result": {
            "content": [{"type": "text", "text": "x" * text_bytes}],
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


def build(n_jobs: int = JOBS, rows: int = ROWS, text_bytes: int = ROW_TEXT_BYTES) -> Any:
    jobs = [
        FakeJob(f"job-{k}", [make_row(i, text_bytes) for i in range(rows)]) for k in range(n_jobs)
    ]
    session = FakeSession(jobs)
    state = FrontendSessionState.model_validate(
        {
            "session_id": "frame-cost",
            "epoch": "e1",
            "sequence": 0,
            "jobs": [JobState.from_job(j) for j in jobs],
        }
    )
    return FrontendStateStore(state), session


def advance(session: Any) -> None:
    """The streaming shape: one new row per job per tick, at the 500-row cap."""
    for job in session.jobs.list():
        job.trajectory.append(make_row(job.trajectory[-1][SEQ] + 1))
        del job.trajectory[0]


def measure(label: str, store: Any, session: Any, *, advance_rows: bool) -> float:
    for _ in range(2):
        if advance_rows:
            advance(session)
        store.refresh_jobs(session)
    gc.collect()
    stats0 = gc.get_stats()
    frozen0 = _frozen_rows
    t0 = time.perf_counter()
    for _ in range(TICKS):
        if advance_rows:
            advance(session)
        store.refresh_jobs(session)
    dt = (time.perf_counter() - t0) / TICKS
    stats1 = gc.get_stats()
    collected = sum(b["collections"] - a["collections"] for a, b in zip(stats0, stats1))
    gen2 = stats1[2]["collections"] - stats0[2]["collections"]
    frozen = (_frozen_rows - frozen0) / TICKS
    print(
        f"{label:<34} {dt * 1000:8.2f} ms/tick  "
        f"gc {collected / TICKS:6.1f}/tick (gen2 {gen2 / TICKS:4.2f})  "
        f"rows frozen {frozen:7.1f}/tick  -> {dt * 20:6.1%} of one core at 20 Hz"
    )
    return dt


def ticks() -> None:
    print(f"{JOBS} subagents x {ROWS} retained rows x {ROW_TEXT_BYTES} B/row\n")
    store, session = build()
    base_idle = measure("idle (window not moving)", store, session, advance_rows=False)
    store, session = build()
    base_live = measure("streaming (1 row/job/tick at cap)", store, session, advance_rows=True)
    print(f"{'':34} baseline idle {base_idle * 1000:.2f} ms, live {base_live * 1000:.2f} ms")


def sync() -> None:
    print("sync_wire_payload on the loop thread\n")
    for n_jobs in (5, 22):
        for text_bytes in (2048, 8192):
            rows = [make_row(i, text_bytes) for i in range(ROWS)]
            jobs = [
                JobState.from_job(FakeJob(f"job-{k}", [dict(r) for r in rows]))
                for k in range(n_jobs)
            ]
            state = FrontendSessionState.model_validate(
                {
                    "session_id": "frame-cost-sync",
                    "epoch": "e1",
                    "sequence": 1,
                    "jobs": jobs,
                }
            )
            store = FrontendStateStore(state)
            retained = n_jobs * sum(len(json.dumps(r)) for r in rows)
            frontend = FrontendSync(
                epoch="e1", sequence=1, snapshot=store.state, display_history=None
            )
            gc.collect()
            t0 = time.perf_counter()
            payload = sync_wire_payload(frontend)
            dt = time.perf_counter() - t0
            print(
                f"{n_jobs:2d} jobs x {ROWS} rows x {text_bytes:5d} B "
                f"(retained {retained / 1024 / 1024:6.1f} MiB): "
                f"{dt * 1000:8.1f} ms  wire payload "
                f"{len(json.dumps(payload).encode()) / 1024:7.1f} KiB"
            )
            del state, store, frontend, payload, jobs, rows
            gc.collect()


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(
        f"interpreter {sys.version.split()[0]}  gc thresholds={gc.get_threshold()}\n"
        f"tree imported: {fs.__file__}\n"
        f"load average: {', '.join(f'{v:.2f}' for v in os.getloadavg())}\n"
    )
    install_freeze_spy()
    if what in ("ticks", "all"):
        ticks()
        print()
    if what in ("sync", "all"):
        sync()


if __name__ == "__main__":
    main()
