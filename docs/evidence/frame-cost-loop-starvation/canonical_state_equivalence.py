"""Canonical-state equivalence over one deterministic scenario list.

Runs the SAME scenario against whatever tree it is pointed at and prints a digest
of the canonical state after every step. The two trees must agree on EVERY digest
-- that is the correctness claim for the retained-window memo: it may change what
a roster tick costs, never what canonical state holds. The one deliberate
exception is printed last and labelled: a row revised IN PLACE, which the writer
never does and no fingerprint can see.

  cd /tmp && PYTHONPATH=~/local-operator-worktrees/<pre-change-tree> \
    ~/local-operator-worktrees/frame-cost-loop-starvation/.venv/bin/python \
    docs/evidence/frame-cost-loop-starvation/canonical_state_equivalence.py

Run it from OUTSIDE either tree, so ``PYTHONPATH`` picks which one is imported.
The scenario covers every shape a retained window takes: idle ticks, one-row
appends, a fill to the cap, cap rotation, a burst, a front-only trim, an
unstamped row, a status move, rows dropped, the row LIST replaced by a new
attempt, an epoch move, a job leaving and joining the roster, and both
``refresh_from_session`` modes.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

os.environ.setdefault("LOCAL_OPERATOR_CONFIG_DIR", "/tmp/window-cfg")

from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY as SEQ  # noqa: E402
from local_operator.session.frontend_state import (  # noqa: E402
    FrontendSessionState,
    FrontendStateStore,
    JobState,
)

ROWS = 500
JOBS = 5
ROW_TEXT_BYTES = 2048


def make_row(i: int, *, stamped: bool = True) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "tool_execution_end",
        "tool_call_id": f"call_{i:08d}",
        "tool_name": "bash",
        "intent": "run a command and read what it printed",
        "result": {
            "content": [{"type": "text", "text": "x" * ROW_TEXT_BYTES}],
            "details": {"exit_code": 0, "added": i, "removed": 3},
        },
    }
    if stamped:
        row[SEQ] = i
    return row


class FakeJob:
    def __init__(self, job_id: str, rows: list[dict[str, Any]], counter: int) -> None:
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
        #: The writer's append counter: monotone, never revised, never reused.
        self.counter = counter


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


def append(job: FakeJob, n: int = 1, *, stamped: bool = True, trim: bool = True) -> None:
    for _ in range(n):
        job.trajectory.append(make_row(job.counter, stamped=stamped))
        job.counter += 1
    if trim and len(job.trajectory) > ROWS:
        del job.trajectory[: len(job.trajectory) - ROWS]


def build() -> tuple[Any, FakeSession]:
    jobs = []
    for k in range(JOBS):
        rows = [make_row(i) for i in range(4)]
        jobs.append(FakeJob(f"job-{k}", rows, 4))
    session = FakeSession(jobs)
    state = FrontendSessionState.model_validate(
        {
            "session_id": "window-scenarios",
            "epoch": "e1",
            "sequence": 0,
            "jobs": [JobState.from_job(j) for j in jobs],
        }
    )
    return FrontendStateStore(state), session


def digest(store: Any) -> str:
    payload = store._state.model_dump(mode="json")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def main() -> None:
    store, session = build()
    jobs = session.jobs.list()
    steps: list[tuple[str, Any]] = [
        ("idle-1", None),
        ("idle-2", None),
        ("append-1-each", lambda: [append(j) for j in jobs]),
        ("append-1-each", lambda: [append(j) for j in jobs]),
        ("fill-to-cap", lambda: [append(j, ROWS) for j in jobs]),
        ("rotating-1-append-1-trim", lambda: [append(j) for j in jobs]),
        ("rotating-1-append-1-trim", lambda: [append(j) for j in jobs]),
        ("burst-append-40", lambda: [append(j, 40) for j in jobs]),
        ("front-trim-only", lambda: [j.trajectory.__delitem__(slice(0, 7)) for j in jobs]),
        ("replace-last-row-object", lambda: _replace_tail(jobs)),
        ("append-unstamped", lambda: [append(j, stamped=False) for j in jobs]),
        ("status-move", lambda: [setattr(j, "status", "completed") for j in jobs]),
        ("drop-all-rows", lambda: [j.trajectory.clear() for j in jobs]),
        ("append-after-rewrite", lambda: [append(j) for j in jobs]),
        ("epoch-move", lambda: _epoch(store, jobs)),
        ("remove-a-job", lambda: session.jobs._jobs.pop()),
        (
            "add-a-job",
            lambda: session.jobs._jobs.append(
                FakeJob("job-new", [make_row(i) for i in range(3)], 3)
            ),
        ),
    ]
    for label, action in steps:
        if action is not None:
            action()
        t0 = time.perf_counter()
        store.refresh_jobs(session)
        dt = time.perf_counter() - t0
        rows = [len(j.trajectory) for j in store._state.jobs]
        print(
            f"{label:<26} dt={dt * 1000:7.2f} ms  digest={digest(store)}  rows={rows}",
            flush=True,
        )
    print("\n--- refresh_from_session ---")
    for label in ("plain", "initial"):
        t0 = time.perf_counter()
        store.refresh_from_session(session, initial=(label == "initial"))
        dt = time.perf_counter() - t0
        print(f"{label:<26} dt={dt * 1000:7.2f} ms  digest={digest(store)}", flush=True)
    store.refresh_jobs(session)
    print(f"{'tick after refresh':<26} digest={digest(store)}", flush=True)
    # NOT part of the equivalence set: a row revised IN PLACE. The writer never
    # does this (``_make_relay`` appends a fresh dict per event and never revises
    # one), and it is the one change no fingerprint over stamps, list identity or
    # tail identity can see -- an unchanged count, first and last stamp and the
    # same row objects describe two different windows. Recorded here because it
    # is the boundary of what the cache may be trusted for, and it is pinned by
    # the writer-invariant test rather than by a check inside the cache.
    jobs[0].trajectory[-1]["intent"] = "revised in place, stamp untouched"
    store.refresh_jobs(session)
    print(
        f"{'row revised in place':<26} dt=  n/a  digest={digest(store)}  "
        "(expected to differ from the pre-change tree; see the comment)",
        flush=True,
    )


def _replace_tail(jobs: list[FakeJob]) -> None:
    """Swap the last row for an equal COPY, keeping its stamp."""
    for job in jobs:
        if job.trajectory:
            job.trajectory[-1] = dict(job.trajectory[-1])


def _epoch(store: Any, jobs: list[FakeJob]) -> None:
    payload = store._state.model_dump(mode="json")
    payload["epoch"] = "e2"
    payload["sequence"] = 0
    store.replace(FrontendSessionState.model_validate(payload))


if __name__ == "__main__":
    main()
